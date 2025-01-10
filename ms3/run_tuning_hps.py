# %%
import copy
import optuna
import time

import numpy as np
import pandas as pd

from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
from torch import nn

from ms3.utils import get_subject_ids, get_dataloader
from ms3.utils.architecture import USleepTMA

device = "cuda" if torch.cuda.is_available() else "cpu"


# %%

# HPs for the experiment

# dataset
n_subjects = 300

# dataloader
n_windows = 35
n_windows_stride = 10
batch_size = 64
num_workers = 6

# model
in_chans = 2
n_classes = 5
input_size_samples = 3000

# training
n_epochs = 30
patience = 5

# %%
metadata = pd.read_csv(
    "metadata/metadata_sleep.csv").drop(columns=["Unnamed: 0"])
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


# %%
def objective(trial):
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    rng = check_random_state(seed)

    # Sample hyperparameters
    print("Trial number: ", trial.number)
    depth_tma = trial.suggest_int("depth_tma", 0, 3)
    if depth_tma > 0:
        filter_size = trial.suggest_int("filter_size", 16, 32)
    else:
        filter_size = None
    filter_size_input = 32
    print("Depth TMA: ", depth_tma)
    print("Filter size: ", filter_size)
    accs = []
    for _ in range(3):
        # select 3 datasets
        dataset_targets = rng.choice(dataset_names, 3, replace=False)
        print("Dataset targets: ", dataset_targets)
        dataset_sources = dataset_names.copy()
        for dataset_target in dataset_targets:
            dataset_sources.remove(dataset_target)

        subject_ids_target = {
            dataset_target: subject_ids[dataset_target]
            for dataset_target in dataset_targets
        }

        subject_ids_train, subject_ids_val = dict(), dict()
        for dataset in dataset_sources:
            subject_ids_dataset = (
                subject_ids[dataset][:n_subjects]
                if n_subjects > 0
                else subject_ids[dataset]
            )
            subject_ids_train[dataset], subject_ids_val[dataset] = (
                train_test_split(
                    subject_ids_dataset, test_size=0.2, random_state=rng
                )
            )

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

        dataloader_target = get_dataloader(
            metadata,
            dataset_targets,
            subject_ids_target,
            n_windows,
            n_windows_stride,
            batch_size,
            num_workers,
        )

        # Initialize the model with sampled hyperparameters
        model = USleepTMA(
            n_chans=in_chans,
            sfreq=100,
            depth=12,
            with_skip_connection=True,
            n_outputs=n_classes,
            n_times=input_size_samples,
            filter_size=filter_size,
            filter_size_input=filter_size_input,
            depth_tma=depth_tma,
        )
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        min_val_loss = np.inf
        patience_counter = 0

        for epoch in range(n_epochs):
            model.train()
            train_loss = []
            time_start = time.time()
            for batch_X, batch_y in dataloader_train:
                optimizer.zero_grad()
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                output = model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())

            # Validation loop
            model.eval()
            val_loss = []
            y_pred_all, y_true_all = [], []
            with torch.no_grad():
                for batch_X, batch_y in dataloader_val:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    output = model(batch_X)
                    loss = criterion(output, batch_y)
                    val_loss.append(loss.item())

            if min_val_loss > np.mean(val_loss):
                min_val_loss = np.mean(val_loss)
                patience_counter = 0
                best_model = copy.deepcopy(model)
            else:
                patience_counter += 1
                if patience_counter > patience:
                    print("Early stopping")
                    break

            time_end = time.time()
            print(
                "Ep:",
                epoch,
                "Loss:",
                round(np.mean(train_loss), 2),
                "LossVal:",
                round(np.mean(val_loss), 2),
                "Time:",
                round(time_end - time_start, 2),
            )
        best_model.eval()
        with torch.no_grad():
            y_pred_all, y_true_all = [], []
            for batch_X, batch_y in dataloader_target:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                output = best_model(batch_X)
                y_pred_all.append(output.argmax(axis=1).cpu().detach().numpy())
                y_true_all.append(batch_y.cpu().detach().numpy())

            y_pred = np.concatenate(y_pred_all)[:, 10:25]
            y_true = np.concatenate(y_true_all)[:, 10:25]
            target_accuracy = accuracy_score(
                y_true.flatten(), y_pred.flatten()
            )

        accs.append(target_accuracy)
    accuracy = np.mean(accs)
    return accuracy


# Create the Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
# Print the best hyperparameters and corresponding performance
print("Best Hyperparameters: ", study.best_params)
print("Best Validation Accuracy: ", study.best_value)

# Save the study results
study.trials_dataframe().to_csv("optuna_study_results_with_layer0.csv")
