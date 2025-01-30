# %%
import time
import copy

import numpy as np
import pandas as pd

from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score

import torch
from torch import nn

from ms3.utils import get_subject_ids, get_dataloader, filter_metadata
from ms3.utils.architecture import USleepTMA

device = "cuda" if torch.cuda.is_available() else "cpu"


# %%

# HPs for the experiment
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
rng = check_random_state(seed)

# dataloader
n_windows = 35
n_windows_stride = 10
batch_size = 64
num_workers = 6
percentage = 1

# model
in_chans = 2
n_classes = 5
input_size_samples = 3000
depth = 12

norm = "BatchNorm"

if norm == "BatchNorm":
    filter_size = None
    depth_norm = None

elif norm == "PSDNorm":
    filter_size = 16
    depth_norm = 3

elif norm == "InstanceNorm":
    filter_size = None
    depth_norm = 3
    norm = "InstanceNorm"

# training
n_epochs = 30
patience = 5
balanced = False

print(f"filter_size: {filter_size}, depth_norm: {depth_norm}, norm: {norm}")

# %%
metadata = pd.read_csv("metadata/metadata_sleep.csv").drop(columns=["Unnamed: 0"])

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
subject_ids = get_subject_ids(metadata, dataset_names)

dataset_targets = ["SOF", "MASS", "CHAT"]
subject_ids_target = {
    dataset_target: subject_ids[dataset_target]
    for dataset_target in dataset_targets
}

dataset_sources = dataset_names.copy()
for dataset_target in dataset_targets:
    dataset_sources.remove(dataset_target)

# %%
print(f"Percentage: {percentage}")
subject_ids_train, subject_ids_val = dict(), dict()
subject_ids_test = dict()
n_subject_tot = 0
for dataset_name in dataset_sources:
    if dataset_name == dataset_target:
        continue
    subject_ids_train_val, subject_ids_test[dataset_name] = train_test_split(
        subject_ids[dataset_name], test_size=0.2, random_state=rng
    )
    n_subjects = int(percentage * len(subject_ids_train_val))
    n_subject_tot += n_subjects
    n_subjects = 2 if n_subjects < 2 else n_subjects
    print(f"Dataset: {dataset_name}, n_subjects: {n_subjects}")
    subject_ids_dataset = rng.choice(subject_ids_train_val, n_subjects, replace=False)
    subject_ids_train[dataset_name], subject_ids_val[dataset_name] = train_test_split(
        subject_ids_dataset, test_size=0.2, random_state=rng
    )

# %%
if balanced:
    metadata_train = filter_metadata(metadata, dataset_sources, subject_ids_train)
    y_train = metadata_train.y.values
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train),
        y=y_train
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
)

dataloader_val = get_dataloader(
    metadata,
    dataset_sources,
    subject_ids_val,
    n_windows,
    n_windows_stride,
    batch_size,
    num_workers,
)
# %%

model = USleepTMA(
    n_chans=in_chans,
    sfreq=100,
    depth=depth,
    with_skip_connection=True,
    n_outputs=n_classes,
    n_times=input_size_samples,
    filter_size=filter_size,
    filter_size_input=None,
    depth_norm=depth_norm,
    norm=norm,
)

model.to(device)
criterion = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights).to(device) if balanced else None)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# %%
history = []

print("Start training")
min_val_loss = np.inf
for epoch in range(n_epochs):
    time_start = time.time()
    model.train()
    train_loss = np.zeros(len(dataloader_train))
    y_pred_all, y_true_all = list(), list()
    for i, (batch_X, batch_y) in enumerate(dataloader_train):
        optimizer.zero_grad()
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        output = model(batch_X)

        loss_batch = criterion(output, batch_y)

        loss_batch.backward()
        optimizer.step()

        y_pred_all.append(output.argmax(axis=1).cpu().detach().numpy())
        y_true_all.append(batch_y.cpu().detach().numpy())
        train_loss[i] = loss_batch.item()

    y_pred = np.concatenate(y_pred_all)[:, 10:25]
    y_true = np.concatenate(y_true_all)[:, 10:25]
    perf = accuracy_score(y_true.flatten(), y_pred.flatten())
    f1 = f1_score(
        y_true.flatten(), y_pred.flatten(), average="weighted"
    )

    model.eval()
    with torch.no_grad():
        val_loss = np.zeros(len(dataloader_val))
        y_pred_all, y_true_all = list(), list()
        for i, (batch_X, batch_y) in enumerate(dataloader_val):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            output = model(batch_X)

            loss_batch = criterion(output, batch_y)

            y_pred_all.append(output.argmax(axis=1).cpu().detach().numpy())
            y_true_all.append(batch_y.cpu().detach().numpy())
            val_loss[i] = loss_batch.item()

        y_pred = np.concatenate(y_pred_all)[:, 10:25]
        y_true = np.concatenate(y_true_all)[:, 10:25]
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
history_path = f"results_all/history/history_{norm}_{percentage}_{balanced}.pkl"
df_history = pd.DataFrame(history)
df_history.to_pickle(history_path)

torch.save(best_model, f"results_all/models/models_{norm}_{percentage}_{balanced}.pt")

# %%
results = []
results_path = f"results_all/pickle/results_{norm}_{percentage}_{balanced}.pkl"
for dataset_target in dataset_targets:
    n_target = len(subject_ids_target[dataset_target])
    for n_subj in range(n_target):
        dataloader_target = get_dataloader(
            metadata,
            [dataset_target],
            {dataset_target: subject_ids_target[dataset_target][n_subj:n_subj+1]},
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

                output = best_model(batch_X)

                y_pred_all.append(output.argmax(axis=1).cpu().detach().numpy())
                y_true_all.append(batch_y.cpu().detach().numpy())

            y_pred = np.concatenate(y_pred_all)[:, 10:25].flatten()
            y_t = np.concatenate(y_true_all)[:, 10:25].flatten()

        results.append(
            {
                "subject": n_subj,
                # add hps
                "seed": seed,
                "dataset": dataset_target,
                "dataset_type": "target",
                "filter_size_input": None,
                "filter_size": filter_size,
                "depth_norm": depth_norm,
                "n_subject_train": n_subject_tot,
                "n_subject_test": len(subject_ids_target[dataset_target]),
                "n_windows": n_windows,
                "n_windows_stride": n_windows_stride,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "n_epochs": n_epochs,
                "patience": patience,
                "percentage": percentage,
                "balanced": balanced,
                "norm": norm,
                # add metrics
                "y_pred": y_pred,
                "y_true": y_t,
            }
        )

# %%
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

                output = best_model(batch_X)

                y_pred_all.append(output.argmax(axis=1).cpu().detach().numpy())
                y_true_all.append(batch_y.cpu().detach().numpy())

            y_pred = np.concatenate(y_pred_all)[:, 10:25].flatten()
            y_t = np.concatenate(y_true_all)[:, 10:25].flatten()

        results.append(
            {
                "subject": n_subj,
                # add hps
                "seed": seed,
                "dataset": dataset_source,
                "dataset_type": "source",
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
                "balanced": balanced,
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
