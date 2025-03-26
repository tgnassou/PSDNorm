# %%
import time
import copy

import numpy as np
import pandas as pd

from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import torch
from torch import nn

from temporal_norm.utils import get_subject_ids, get_dataloader
from temporal_norm.utils.architecture import USleepNorm

import argparse
import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="ABC")
parser.add_argument("--percent", type=float, default=0.01)
parser.add_argument("--norm", type=str, default="PSDNorm")

args = parser.parse_args()


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
batch_size = 64
num_workers = 40

# model
in_chans = 2
n_classes = 5
input_size_samples = 3000

if norm == "BatchNorm":
    filter_size = None
    depth_norm = None

elif norm == "PSDNorm":
    filter_size = 16
    depth_norm = 3

print(f"Filter size: {filter_size}, Depth Norm: {depth_norm}, Norm: {norm}")
# training
n_epochs = 30
patience = 5

# %%

best_model = torch.load(f"results_LODO/models/models_{norm}_{percentage}_LODO_{dataset_target}.pt")
best_model = best_model.to(device)
results_path = f"results_LODO/pickles/results_{norm}_{percentage}_LODO_{dataset_target}.pkl"

n_target = len(subject_id_target)
for n_subj in tqdm.tqdm(range(n_target)):
    dataloader_target = get_dataloader(
        metadata,
        [dataset_target],
        {dataset_target: subject_id_target[n_subj:n_subj+1]},
        n_windows,
        n_windows_stride,
        batch_size,
        num_workers,
    )
    y_pred_all, y_true_all = list(), list()
    #  create one y_pred per moduels

    best_model.eval()
    with torch.no_grad():
        for i, (batch_X, batch_y) in enumerate(dataloader_target):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            output = best_model(batch_X)

            y_pred_all.append(output.argmax(axis=1).detach())
            y_true_all.append(batch_y.detach())

        y_pred_all = [y.cpu().numpy() for y in y_pred_all]
        y_true_all = [y.cpu().numpy() for y in y_true_all]

        y_pred = np.concatenate(y_pred_all)[:, 10:25].flatten()
        y_t = np.concatenate(y_true_all)[:, 10:25].flatten()
    results = [
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
            "num_workers": num_workers,
            "n_epochs": n_epochs,
            "patience": patience,
            "percentage": percentage,
            # add metrics
            "y_pred": y_pred,
            "y_true": y_t,
        }
    ]

    try:
        df_results = pd.read_pickle(results_path)
    except FileNotFoundError:
        df_results = pd.DataFrame()
    df_results = pd.concat((df_results, pd.DataFrame(results)))
    df_results.to_pickle(results_path)

print("Target Inference Done")