# %%
import time
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

from temporal_norm.utils import get_subject_ids, get_dataloader
from temporal_norm.utils.architecture import USleepNorm

import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"
# %%
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="ABC")
parser.add_argument("--percent", type=float, default=0.01)
parser.add_argument("--norm", type=str, default="PSDNorm")
parser.add_argument("--use_amp", action="store_true")

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
n_windows_stride = 10
# batch_size = 512
batch_size = 64
batch_size_inference = batch_size * 16
num_workers = 10
pin_memory = True
persistent_workers = False

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
n_epochs = 10
patience = 5

# %%
subject_ids = get_subject_ids(metadata, dataset_names)

subject_id_target = subject_ids[dataset_target]

dataset_sources = dataset_names.copy()
dataset_sources.remove(dataset_target)

# %%
subject_ids_train, subject_ids_val = dict(), dict()
subject_ids_test = dict()
n_subject_tot = 0
print("Datasets used for training (and validation):")
for dataset_name in dataset_sources:
    subject_ids_train_val, subject_ids_test[dataset_name] = train_test_split(
        subject_ids[dataset_name], test_size=0.2, random_state=rng
    )
    n_subjects = int(percentage * len(subject_ids_train_val))
    n_subjects = 2 if n_subjects < 2 else n_subjects
    n_subject_tot += n_subjects
    print(f"Dataset: {dataset_name}, n_subjects: {n_subjects}")
    subject_ids_dataset = rng.choice(subject_ids_train_val, n_subjects, replace=False)
    subject_ids_train[dataset_name], subject_ids_val[dataset_name] = train_test_split(
        subject_ids_dataset, test_size=0.2, random_state=seed
    )

# %%
dataloader_train = get_dataloader(
    metadata,
    dataset_sources,
    subject_ids_train,
    n_windows,
    n_windows_stride,
    batch_size,
    num_workers,
    pin_memory,
    persistent_workers,
)

dataloader_val = get_dataloader(
    metadata,
    dataset_sources,
    subject_ids_val,
    n_windows,
    n_windows_stride,
    batch_size_inference,
    num_workers,
    pin_memory,
    persistent_workers,
)

print()
print(f"Number of subjects: {n_subject_tot}")
print(f"Number of training subjects: {sum([len(v) for v in subject_ids_train.values()])}")
print(f"Number of validation subjects: {sum([len(v) for v in subject_ids_val.values()])}")
print(f"Number of testing subjects: {sum([len(v) for v in subject_ids_test.values()])}")

print(f"Number of training batches: {len(dataloader_train)}")
print(f"Number of validation batches: {len(dataloader_val)}")

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
    running_window = 100  # Number of batches for averaging loss
    for i, (batch_X, batch_y) in enumerate(
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

        # Update tqdm progress bar every 10 batches with average loss
        running_loss += loss_batch.item()
        if (i + 1) % running_window == 0:
            avg_loss = running_loss / running_window
            tqdm.write(f"Batch {i+1}/{len(dataloader_train)}, Avg Loss: {avg_loss:.3f}")
            running_loss = 0.0

    y_pred_all = [y.cpu().numpy() for y in y_pred_all]
    y_true_all = [y.cpu().numpy() for y in y_true_all]
    y_pred = np.concatenate(y_pred_all)[:, 10:25]
    y_true = np.concatenate(y_true_all)[:, 10:25]
    perf = accuracy_score(y_true.flatten(), y_pred.flatten())
    f1 = f1_score(
        y_true.flatten(), y_pred.flatten(), average="weighted"
    )

    print(f"End epoch train: {time.time() - time_start:.2f}s")

    print(f"Start eval on val...")

    model.eval()
    with torch.no_grad():
        val_loss = np.zeros(len(dataloader_val))
        y_pred_all, y_true_all = list(), list()
        for i, (batch_X, batch_y) in enumerate(tqdm(dataloader_val)):
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

        y_pred = np.concatenate(y_pred_all)[:, 10:25]
        y_true = np.concatenate(y_true_all)[:, 10:25]
        perf_val = accuracy_score(y_true.flatten(), y_pred.flatten())
        std_val = np.std(perf_val)
        f1_val = f1_score(
            y_true.flatten(), y_pred.flatten(), average="weighted"
        )
        std_f1_val = np.std(f1_val)

        print(f"End of eval val: {time.time() - time_start:.2f}s")

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

n_target = len(subject_id_target)
for n_subj in range(n_target):
    dataloader_target = get_dataloader(
        metadata,
        [dataset_target],
        {dataset_target: subject_id_target[n_subj:n_subj+1]},
        n_windows,
        n_windows_stride,
        batch_size_inference,
        num_workers,
    )
    y_pred_all, y_true_all = list(), list()
    #  create one y_pred per moduels

    best_model.eval()
    with torch.no_grad():
        for i, (batch_X, batch_y) in enumerate(dataloader_target):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            with autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                output = best_model(batch_X)

            y_pred_all.append(output.argmax(axis=1).detach())
            y_true_all.append(batch_y.detach())

        y_pred_all = [y.cpu().numpy() for y in y_pred_all]
        y_true_all = [y.cpu().numpy() for y in y_true_all]

        y_pred = np.concatenate(y_pred_all)[:, 10:25].flatten()
        y_t = np.concatenate(y_true_all)[:, 10:25].flatten()
    results.append(
        {
            "subject": n_subj,
            # add hps
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
            # add metrics
            "y_pred": y_pred,
            "y_true": y_t,
        }
    )

try:
    df_results = pd.read_pickle(results_path)
except FileNotFoundError:
    df_results = pd.DataFrame()
df_results = pd.concat((df_results, pd.DataFrame(results)))
df_results.to_pickle(results_path)

print("Target Inference Done")

# %%
results = []
for dataset_source in dataset_sources:
    for n_subj in range(len(subject_ids_test[dataset_source])):
        dataloader_target = get_dataloader(
            metadata,
            [dataset_source],
            {dataset_source: subject_ids_test[dataset_source][n_subj:n_subj+1]},
            n_windows,
            n_windows_stride,
            batch_size,
            num_workers,
        )
        y_pred_all, y_true_all = list(), list()
        best_model.eval()
        with torch.no_grad():
            for i, (batch_X, batch_y) in enumerate(dataloader_target):
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                with autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                    output = best_model(batch_X)

                y_pred_all.append(output.argmax(axis=1).detach())
                y_true_all.append(batch_y.detach())

            y_pred_all = [y.cpu().numpy() for y in y_pred_all]
            y_true_all = [y.cpu().numpy() for y in y_true_all]

            y_pred = np.concatenate(y_pred_all)[:, 10:25].flatten()
            y_t = np.concatenate(y_true_all)[:, 10:25].flatten()

        results.append(
            {
                "subject": n_subj,
                # add hps
                "seed": seed,
                "dataset": dataset_source,
                "dataset_type": "source",
                "norm": norm,
                "filter_size_input": None,
                "filter_size": filter_size,
                "depth_norm": depth_norm,
                "n_subject_train": n_subject_tot,
                "n_subject_test": len(subject_ids_test[dataset_source]),
                "n_windows": n_windows,
                "n_windows_stride": n_windows_stride,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "n_epochs": n_epochs,
                "patience": patience,
                "percentage": percentage,
                "norm": norm,
                # add metrics
                "y_pred": y_pred,
                "y_true": y_t,
            }
        )

try:
    df_results = pd.read_pickle(results_path)
except FileNotFoundError:
    df_results = pd.DataFrame()
df_results = pd.concat((df_results, pd.DataFrame(results)))
df_results.to_pickle(results_path)

print("Source Inference Done")
