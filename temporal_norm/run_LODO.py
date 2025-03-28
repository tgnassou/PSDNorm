# %%
import time
from collections import defaultdict
import copy
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd

from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import torch
from torch import nn
from torch.amp import autocast, GradScaler

from temporal_norm.utils import get_subject_ids, get_dataloader, get_probs
from temporal_norm.utils.architecture import USleepNorm

import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"
# %%
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="ABC")
parser.add_argument("--percent", type=float, default=0.01)
parser.add_argument("--norm", type=str, default="PSDNorm")
parser.add_argument("--use_amp", action="store_true")
parser.add_argument("--balanced", action="store_true")
parser.add_argument("--num_workers", type=int, default=40)

args = parser.parse_args()
use_amp = args.use_amp
if use_amp:
    print("BE CAREFUL! AMP is enabled.")

# %%
dataset_names = [
    "ABC",
    "CHAT",
    "CFS",
    "SHHS",
    "HOMEPAP",
    "CCSHS",
    "MASS",
    "PhysioNet",
    "SOF",
    "MROS",
]
metadata = pd.read_parquet("metadata/metadata_sleep.parquet")

# %%
percentage = args.percent
norm = args.norm
dataset_target = args.dataset

print(f"Percentage: {percentage}")
modules = []

# HPs for the experiment
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
rng = check_random_state(seed)

# dataloader
n_windows = 35
n_windows_stride = 21
batch_size = 64
batch_size_inference = batch_size
num_workers = args.num_workers
pin_memory = True
persistent_workers = False
balanced = args.balanced

# model
in_chans = 2
n_classes = 5
input_size_samples = 3000
lr = 1e-3

if norm == "BatchNorm":
    filter_size = None
    depth_norm = None

elif norm == "PSDNorm":
    filter_size = 16
    depth_norm = 3

print(f"Filter size: {filter_size}, Depth Norm: {depth_norm}, Norm: {norm}")
# training
n_epochs = 3
patience = 5

# idx center of the window
assert (n_windows - n_windows_stride) % 2 == 0, "n_windows - n_windows_stride must be even"
first_window_idx = (n_windows - n_windows_stride) // 2
last_window_idx = first_window_idx + n_windows_stride

# %%
subject_ids = get_subject_ids(metadata, dataset_names)

subject_id_target = subject_ids[dataset_target]

dataset_sources = dataset_names.copy()
dataset_sources.remove(dataset_target)

# %%
subject_ids_train, subject_ids_val = dict(), dict()
n_subject_tot = 0

print("Datasets used for training and validation:")
for dataset_name in dataset_sources:
    subject_ids_all = subject_ids[dataset_name]
    n_subjects = int(percentage * len(subject_ids_all))
    n_subjects = max(n_subjects, 2)
    n_subject_tot += n_subjects

    print(f"Dataset: {dataset_name}, n_subjects: {n_subjects}")

    # Randomly sample the subjects to use
    subject_ids_dataset = rng.choice(subject_ids_all, n_subjects, replace=False)

    # Split into train/val
    subject_ids_train[dataset_name], subject_ids_val[dataset_name] = train_test_split(
        subject_ids_dataset, test_size=0.2, random_state=seed
    )

# %%
probs = get_probs(metadata, dataset_sources, alpha=0.5)

# Source train dataloader
dataloader_train = get_dataloader(
    metadata=metadata,
    dataset_names=dataset_sources,
    subject_ids=subject_ids_train,
    n_windows=n_windows,
    n_windows_stride=n_windows_stride,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=persistent_workers,
    balanced=balanced,
    randomize=True,
)

# Source val dataloader
dataloader_val = get_dataloader(
    metadata=metadata,
    dataset_names=dataset_sources,
    subject_ids=subject_ids_val,
    n_windows=n_windows,
    n_windows_stride=n_windows_stride,
    batch_size=batch_size_inference,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=persistent_workers,
    balanced=balanced,
    randomize=False,
)

# Target dataloader
dataloader_target = get_dataloader(
    metadata=metadata,
    dataset_names=[dataset_target],
    subject_ids={dataset_target: subject_id_target},
    n_windows=n_windows,
    n_windows_stride=n_windows_stride,
    batch_size=batch_size_inference,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=persistent_workers,
    balanced=balanced,
    randomize=False,
)


print()
print(f"Number of source subjects: {n_subject_tot}")
print(f"Number of training subjects: {sum([len(v) for v in subject_ids_train.values()])}")
print(f"Number of validation subjects: {sum([len(v) for v in subject_ids_val.values()])}")
print(f"Number of target subjects: {len(subject_id_target)}")
print()

print(f"Number of training batches: {len(dataloader_train)}")
print(f"Number of validation batches: {len(dataloader_val)}")
print(f"Number of target batches: {len(dataloader_target)}")


# %%

model = USleepNorm(
    n_chans=in_chans,
    sfreq=100,
    depth=12,
    with_skip_connection=True,
    n_outputs=n_classes,
    n_times=input_size_samples,
    filter_size=filter_size,
    depth_norm=depth_norm,
    norm=norm,
)
# model = torch.compile(model)

model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# %%
history = []

print()
print("Start training")
scaler = GradScaler(device=device, enabled=use_amp)
min_val_loss = np.inf
for epoch in range(n_epochs):
    print()
    print(f"Epoch: {epoch}")
    time_start = time.time()
    model.train()
    train_loss = np.zeros(len(dataloader_train))
    y_pred_all, y_true_all = list(), list()

    running_loss = 0.0
    running_window = len(dataloader_train) // 20  # Number of batches for averaging loss
    for i, (batch_X, batch_y, _, _) in enumerate(
        tqdm(dataloader_train, desc="Training", unit="batch")
    ):
        optimizer.zero_grad()
        batch_X = batch_X.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        with autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
            output = model(batch_X)
            loss_batch = criterion(output, batch_y)

        scaler.scale(loss_batch).backward()
        scaler.step(optimizer)
        scaler.update()

        y_pred_all.append(output.argmax(axis=1).detach())
        y_true_all.append(batch_y.detach())
        train_loss[i] = loss_batch.item()

        # Update tqdm progress bar every running_window batches with average loss
        running_loss += loss_batch.item()
        if (i + 1) % running_window == 0:
            avg_loss = running_loss / running_window
            tqdm.write(f"Batch {i+1}/{len(dataloader_train)}, Avg Loss: {avg_loss:.3f}")
            running_loss = 0.0

    y_pred_all = [y.cpu().numpy() for y in y_pred_all]
    y_true_all = [y.cpu().numpy() for y in y_true_all]
    y_pred = np.concatenate(y_pred_all)[:, first_window_idx:last_window_idx]
    y_true = np.concatenate(y_true_all)[:, first_window_idx:last_window_idx]
    perf = accuracy_score(y_true.flatten(), y_pred.flatten())
    f1 = f1_score(
        y_true.flatten(), y_pred.flatten(), average="weighted"
    )

    model.eval()
    with torch.no_grad():
        val_loss = np.zeros(len(dataloader_val))
        y_pred_all, y_true_all = list(), list()
        for i, (batch_X, batch_y, _, _) in enumerate(
            tqdm(dataloader_val, desc="Validation", unit="batch")
        ):
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            with autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                output = model(batch_X)

            loss_batch = criterion(output, batch_y)

            y_pred_all.append(output.argmax(axis=1).detach())
            y_true_all.append(batch_y.detach())
            val_loss[i] = loss_batch.item()

        y_pred_all = [y.cpu().numpy() for y in y_pred_all]
        y_true_all = [y.cpu().numpy() for y in y_true_all]

        y_pred = np.concatenate(y_pred_all)[:, first_window_idx:last_window_idx]
        y_true = np.concatenate(y_true_all)[:, first_window_idx:last_window_idx]
        perf_val = accuracy_score(y_true.flatten(), y_pred.flatten())
        std_val = np.std(perf_val)
        f1_val = f1_score(
            y_true.flatten(), y_pred.flatten(), average="weighted"
        )
        std_f1_val = np.std(f1_val)

    time_end = time.time()
    history.append(
        {
            "epoch": epoch,
            "train_loss": np.mean(train_loss),
            "train_acc": perf,
            "train_f1": f1,
            "val_loss": np.mean(val_loss),
            "val_acc": perf_val,
            "val_std": std_val,
            "val_f1": f1_val,
            "val_f1_std": std_f1_val,
        }
    )

    print(
        "Ep:",
        epoch,
        "Loss:",
        round(np.mean(train_loss), 2),
        "Acc:",
        round(np.mean(perf), 2),
        "LossVal:",
        round(np.mean(val_loss), 2),
        "AccVal:",
        round(np.mean(perf_val), 2),
        "Time:",
        round(time_end - time_start, 2),
    )

    # do early stopping
    if min_val_loss > np.mean(val_loss):
        min_val_loss = np.mean(val_loss)
        patience_counter = 0
        best_model = copy.deepcopy(model)
    else:
        patience_counter += 1
        if patience_counter > patience:
            print("Early stopping")
            break

folder = Path("results_LODO")
folder.mkdir(parents=True, exist_ok=True)
folder_history = folder / "history"
folder_history.mkdir(parents=True, exist_ok=True)
history_path = folder_history / f"history_{norm}_{percentage}_LODO_{dataset_target}.pkl"
df_history = pd.DataFrame(history)
df_history.to_pickle(history_path)

folder_model = folder / "models"
folder_model.mkdir(parents=True, exist_ok=True)
torch.save(best_model, folder_model / f"models_{norm}_{percentage}_LODO_{dataset_target}.pt")
# save optimizer
torch.save(optimizer.state_dict(), folder_model / f"optimizer_{norm}_{percentage}_LODO_{dataset_target}.pt")

results = []
folder_pickle = folder / "pickles"
folder_pickle.mkdir(parents=True, exist_ok=True)
results_path = folder_pickle / f"results_{norm}_{percentage}_LODO_{dataset_target}.pkl"

# Accumulate predictions and targets on GPU per subject
results_by_subject = defaultdict(lambda: {"y_pred": [], "y_true": []})

best_model.eval()
with torch.no_grad():
    for batch_X, batch_y, batch_sub_id, batch_session_id in tqdm(
        dataloader_target, desc="Inference on target", unit="batch"
    ):
        batch_X = batch_X.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        batch_sub_id = batch_sub_id.to(device, non_blocking=True)

        with autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
            output = best_model(batch_X)

        preds = output.argmax(dim=1)

        # Gather predictions per subject
        for y_t, y_p, subj in zip(batch_y, preds, batch_sub_id):
            results_by_subject[int(subj.item())]["y_true"].append(y_t[first_window_idx:last_window_idx])
            results_by_subject[int(subj.item())]["y_pred"].append(y_p[first_window_idx:last_window_idx])

results = []
for subj_id, data in results_by_subject.items():
    y_pred_tensor = torch.cat(data["y_pred"])
    y_true_tensor = torch.cat(data["y_true"])

    results.append(
        {
            "subject": subj_id,
            "seed": seed,
            "dataset": dataset_target,
            "dataset_type": "target",
            "norm": norm,
            "filter_size_input": None,
            "filter_size": filter_size,
            "depth_norm": depth_norm,
            "n_subject_train": n_subject_tot,
            "n_subject_test": len(subject_id_target),
            "n_windows": n_windows,
            "n_windows_stride": n_windows_stride,
            "batch_size": batch_size,
            "batch_size_inference": batch_size_inference,
            "num_workers": num_workers,
            "n_epochs": n_epochs,
            "patience": patience,
            "percentage": percentage,
            "y_pred": y_pred_tensor.cpu().numpy().flatten(),
            "y_true": y_true_tensor.cpu().numpy().flatten(),
        }
    )

try:
    df_results = pd.read_pickle(results_path)
except FileNotFoundError:
    df_results = pd.DataFrame()
df_results = pd.concat((df_results, pd.DataFrame(results)))
df_results.to_pickle(results_path)

print("Target Inference Done")
