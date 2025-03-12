# %%
import time
import copy
from tqdm import tqdm

import numpy as np
import pandas as pd

from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from scipy.signal import welch

import torch
from torch import nn

from temporal_norm.utils import get_subject_ids, get_dataloader
from temporal_norm.utils.architecture import USleepTMA

device = "cuda" if torch.cuda.is_available() else "cpu"
# %%

def get_psd(datasets, subjects):
    psd_all = []
    for dataset_name in datasets:
        print(f"Dataset: {dataset_name}")
        psd_dataset = []
        for subject_id in tqdm(subjects[dataset_name]):
            dataloader_train_subject = get_dataloader(
                metadata,
                [dataset_name],
                {dataset_name: [subject_id]},
                n_windows,
                n_windows,
                batch_size,
                num_workers,
                randomize=False,
            )
            X_night = []
            for batch_X, _ in dataloader_train_subject:
                batch_X = batch_X.permute(0, 2, 1, 3)  # (B, C, S, T)
                batch_X = batch_X.flatten(start_dim=2)
                X_night.append(batch_X.numpy())
            X_night = np.concatenate(X_night)
            # flatten
            X_night = np.concatenate(X_night, axis=-1)
            psd = welch(X_night, axis=-1, nperseg=128)[1]
            psd_dataset.append(psd)
        psd_all.append(psd_dataset)
    return psd_all


def get_filters(psd_all, B):
    H_train = []
    for psd_dataset in psd_all:
        H_dataset = []
        for psd in psd_dataset:
            D = np.sqrt(B) / np.sqrt(psd)
            H = np.fft.irfft(D, axis=-1)
            H = np.fft.ifftshift(H, axes=-1)
            H_dataset.append(H)
        H_train.append(H_dataset)
    return H_train


def create_dict_filters(datasets, subjects, filters):
    dict_filters = {}
    for i, dataset_name in enumerate(datasets):
        for j, subject_id in enumerate(subjects[dataset_name]):
            dict_filters[(dataset_name, subject_id)] = filters[i][j]
    return dict_filters

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

tma = "tma_neurips"

if tma == "tma_neurips":
    filter_size = None
    depth_tma = None
    bary_learning = False

# training
n_epochs = 30
patience = 3

# %%
metadata = pd.read_csv("metadata/metadata_sleep.parquet")
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
    print(f"Dataset: {dataset_name}, n_subjects: {n_subjects}")
    subject_ids_dataset = rng.choice(subject_ids_train_val, n_subjects, replace=False)
    subject_ids_train[dataset_name], subject_ids_val[dataset_name] = train_test_split(
        subject_ids_dataset, test_size=0.2, random_state=rng
    )

# %%
psd_all = get_psd(dataset_sources, subject_ids_train)
psd_all_ = np.concatenate(psd_all)
B = np.mean(np.sqrt(psd_all_), axis=0) ** 2
filters = get_filters(psd_all, B)
dict_filters = create_dict_filters(dataset_sources, subject_ids_train, filters)

# %%

dataloader_train = get_dataloader(
    metadata,
    dataset_sources,
    subject_ids_train,
    n_windows,
    n_windows_stride,
    batch_size,
    num_workers,
    randomize=False,
    dict_filters=dict_filters,
)

# %%
psd_all_val = get_psd(dataset_sources, subject_ids_val)
filters_val = get_filters(psd_all_val, B)
dict_filters_val = create_dict_filters(dataset_sources, subject_ids_val, filters_val)

# %%

dataloader_val = get_dataloader(
    metadata,
    dataset_sources,
    subject_ids_val,
    n_windows,
    n_windows_stride,
    batch_size,
    num_workers,
    dict_filters=dict_filters_val,
    randomize=False,
)
# %%

model = USleepTMA(
    n_chans=in_chans,
    sfreq=100,
    depth=12,
    with_skip_connection=True,
    n_outputs=n_classes,
    n_times=input_size_samples,
    filter_size=filter_size,
    filter_size_input=None,
    depth_tma=depth_tma,
    bary_learning=bary_learning,
)

model.to(device)
criterion = nn.CrossEntropyLoss()
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
history_path = f"results_all/history/history_{tma}_{percentage}.pkl"
df_history = pd.DataFrame(history)
df_history.to_pickle(history_path)

torch.save(best_model, f"results_all/models/models_{tma}_{percentage}.pt")

# %%
results = []
results_path = f"results_all/pickle/results_{tma}_{percentage}.pkl"
for dataset_target in dataset_targets:
    n_target = len(subject_ids_target[dataset_target])
    psd_all_target = get_psd([dataset_target], subject_ids_target)
    filters_target = get_filters(psd_all_target, B)
    dict_filters_target = create_dict_filters(
        [dataset_target], subject_ids_target, filters_target
    )
    for n_subj in range(n_target):
        dataloader_target = get_dataloader(
            metadata,
            [dataset_target],
            {dataset_target: subject_ids_target[dataset_target][n_subj:n_subj+1]},
            n_windows,
            n_windows_stride,
            batch_size,
            num_workers,
            randomize=False,
            dict_filters=dict_filters_target,
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
                "tma": tma,
                "filter_size_input": None,
                "filter_size": filter_size,
                "depth_tma": depth_tma,
                "n_subject_train": n_subject_tot,
                "n_subject_test": len(subject_ids_target[dataset_target]),
                "n_windows": n_windows,
                "n_windows_stride": n_windows_stride,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "n_epochs": n_epochs,
                "patience": patience,
                "percentage": percentage,
                # add metrics
                "y_pred": y_pred,
                "y_true": y_t,
            }
        )

# %%
for dataset_source in dataset_sources:
    psd_all_test = get_psd([dataset_source], subject_ids_test)
    filters_test = get_filters(psd_all_test, B)
    dict_filters_test = create_dict_filters(
        [dataset_source], subject_ids_test, filters_test
    )

    for n_subj in range(len(subject_ids_test[dataset_source])):

        dataloader_test = get_dataloader(
            metadata,
            [dataset_source],
            {dataset_source: subject_ids_test[dataset_source][n_subj:n_subj+1]},
            n_windows,
            n_windows_stride,
            batch_size,
            num_workers,
            randomize=False,
            dict_filters=dict_filters_test,
        )

        y_pred_all, y_true_all = list(), list()
        best_model.eval()
        with torch.no_grad():
            for i, (batch_X, batch_y) in enumerate(dataloader_test):
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
                "tma": tma,
                "filter_size_input": None,
                "filter_size": filter_size,
                "depth_tma": depth_tma,
                "n_subject_train": n_subject_tot,
                "n_subject_test": len(subject_ids_test[dataset_source]),
                "n_windows": n_windows,
                "n_windows_stride": n_windows_stride,
                "batch_size": batch_size,
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

# %%
# dataset_target = "MASS"k
# psd_all_target = get_psd([dataset_target], subject_ids_target)

# %%
B = np.load("B.npy")
# #%%
# filters_target = get_filters(psd_all_target, B)
# dict_filters_target = create_dict_filters(
#     [dataset_target], subject_ids_target, filters_target
# )

# %%
# dataset_name = "SOF"
# print(f"Dataset: {dataset_name}")
# subject_id = 1
# dataloader_train_subject = get_dataloader(
#     metadata,
#     [dataset_name],
#     {dataset_name: [subject_id]},
#     n_windows,
#     n_windows,
#     batch_size,
#     num_workers,
#     randomize=False,
# )
# X_night = []
# for batch_X, _ in dataloader_train_subject:
#     batch_test = batch_X
#     batch_X = batch_X.permute(0, 2, 1, 3)  # (B, C, S, T)
#     batch_X = batch_X.flatten(start_dim=2)
#     X_night.append(batch_X.numpy())
# X_night = np.concatenate(X_night)
# # flatten
# X_night = np.concatenate(X_night, axis=-1)
# psd = welch(X_night, axis=-1, nperseg=128)[1]

# # # %%
# # H_0 = dict_filters_target[(dataset_name, subject_id)]
# # # %%
# # from scipy.signal import convolve
# # X = X_night
# # window_size = X.shape[-1]
# # # Reduce the number of dimension to (K, C, N*T)
# # # X = np.concatenate(x_test, axis=-1)
# # C = len(X)
# # # B = psd_all_target[1][3]
# # D = np.sqrt(B) / np.sqrt(psd)
# # H = np.fft.irfft(D, axis=-1)
# # H = np.fft.ifftshift(H, axes=-1)
# # X_norm = [convolve(X[chan], H[chan]) for chan in range(C)]
# # X_norm = np.array(X_norm)

# # # %%
# # psd_0 = welch(X[0], axis=-1, nperseg=128)[1]
# # psd_1 = welch(X_norm[0], axis=-1, nperseg=128)[1]

# # # %%
# # import matplotlib.pyplot as plt
# # plt.plot(psd_0, label="Original")
# # plt.plot(psd_1, label="Filtered")
# # plt.plot(B[0], label="Barycenter")
# # plt.yscale("log")
# # plt.legend()
# # # %%
# # plt.plot(H[0])
# # plt.plot(H_0[0])

# # # %%
# # plt.plot(X[0])
# # plt.plot(X_norm[0])
# # # %%

# # %%

# %%
