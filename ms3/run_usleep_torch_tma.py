# %%
import time
import numpy as np

from braindecode.samplers import SequenceSampler
from braindecode.models import USleep

from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from torch.optim.lr_scheduler import CosineAnnealingLR, ChainedScheduler, LinearLR
from torch.utils.data import DataLoader
from skorch.dataset import Dataset

from monge_alignment.utils import MongeAlignment

from typing import Iterable

import torch
from torch import nn

from ms3.utils._tmanorm import TMANorm, welch_psd
from ms3.utils.architecture import USleepTMA

# import DAtaloader
from tqdm import tqdm

import copy
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(1)


class SequenceDataset(Dataset):
    def __init__(self, X, y, subject_ids, domains, target_transform=None):
        super().__init__(X=X, y=y)
        self.domains = domains
        self.subject_ids = subject_ids
        self.target_transform = target_transform
        self.create_metadata()

    def create_metadata(self):
        self.metadata = pd.DataFrame(
            {
                "target": self.y,
                "subject": self.subject_ids,
                "run": self.domains,
                "i_window_in_trial": np.arange(len(self.y)),
                "i_start_in_trial": np.zeros(len(self.y)),
                "i_stop_in_trial": 3000 * np.ones(len(self.y)),
            }
        )

    def __len__(self):
        return len(self.X)

    def _get_sequence(self, indices):
        X, y = list(), list()
        for ind in indices:
            out_i = super().__getitem__(ind)
            X.append(out_i[0])
            y.append(out_i[1])

        X = np.stack(X, axis=0)
        y = np.array(y)

        return X, y

    def __getitem__(self, idx):
        if isinstance(idx, Iterable):  # Sample multiple windows
            item = self._get_sequence(idx)
        else:
            item = super().__getitem__(idx)
        if self.target_transform is not None:
            item = item[:1] + (self.target_transform(item[1]),) + item[2:]

        return item


# %%
dataset_names = [
    "ABC",
    "CHAT",
    "CFS",
    "SHHS",
    "HOMEPAP",
    "CCSHS",
    "MASS",
    "Physionet",
    "SOF",
    "MROS",
]

data_dict = {}
max_subjects = 50
for dataset_name in dataset_names:
    subject_ids_ = np.load(f"data/{dataset_name}/subject_ids.npy")
    X_ = []
    y_ = []
    subject_selected = []
    for subject in tqdm(subject_ids_):
        X_.append(np.load(f"data/{dataset_name}/X_{subject}.npy"))
        y_.append(np.load(f"data/{dataset_name}/y_{subject}.npy"))
        subject_selected.append(subject)
        if len(X_) == max_subjects:
            break
    data_dict[dataset_name] = [X_, y_, subject_selected]


# %%
module_name = "usleep"
n_windows = 35
n_windows_stride = 10
n_epochs = 100
batch_size = 64
patience = 10
n_jobs = 1
seed = 42
lr = 1e-3
weight = "unbalanced"
use_scheduler = False

for filter_size in [32, 64, 128]:
    for tmatype in ["online1", "online01", "offlineonline1"]:
        if tmatype == "online0":
            tmanorm = "online"
            tmalayer = [0]
        elif tmatype == "online01":
            tmanorm = "online"
            tmalayer = [0, 1]
        elif tmatype == "online1":
            tmanorm = "online"
            tmalayer = [1]
        elif tmatype == "offlineonline1":
            tmanorm = "offline+online"
            tmalayer = [1]
        elif tmatype == "offline":
            tmanorm = "offline"
            tmalayer = None
        else:
            tmanorm = "None"
            tmalayer = None

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        rng = check_random_state(seed)
        for dataset_target in ["MASS", "CHAT"]:
            results_path = (
                f"results/pickle/results_{module_name}_{dataset_target}_{tmatype}.pkl"
            )
            X_target, y_target, subject_ids_target = data_dict[dataset_target]
            X_train, X_val, y_train, y_val = (
                [],
                [],
                [],
                [],
            )
            subjects_train, subjects_val = [], []
            domain_train, domain_val = [], []
            for dataset_source in dataset_names:
                if dataset_source != dataset_target:
                    X_, y_, subjects_ = data_dict[dataset_source]
                    valid_size = 0.2
                    (
                        X_train_,
                        X_val_,
                        y_train_,
                        y_val_,
                        subjects_train_,
                        subjects_val_,
                    ) = train_test_split(X_, y_, subjects_, test_size=valid_size)

                    X_train += X_train_
                    X_val += X_val_
                    y_train += y_train_
                    y_val += y_val_
                    subjects_train += subjects_train_
                    subjects_val += subjects_val_
                    domain_train += [dataset_source] * len(X_train_)
                    domain_val += [dataset_source] * len(X_val_)
                    print(f"Dataset {dataset_source}: {len(X_train_)}" f" train, {len(X_val_)} val")

            if "offline" in tmanorm:
                ma = MongeAlignment(n_jobs=n_jobs)
                X_train = ma.fit_transform(X_train)
                X_val = ma.transform(X_val)
                X_target = ma.transform(X_target)
                del ma

            n_chans, n_time = X_train[0][0].shape
            n_classes = len(np.unique(y_train[0]))

            # Train
            domains = np.concatenate(
                [[domain_train[i]] * len(X_train[i]) for i in range(len(domain_train))]
            )
            subjects = np.concatenate(
                [[subjects_train[i]] * len(X_train[i]) for i in range(len(subjects_train))]
            )
            dataset = SequenceDataset(
                np.concatenate(X_train, axis=0), np.concatenate(y_train, axis=0), subjects, domains
            )

            train_sampler = SequenceSampler(
                dataset.metadata, n_windows, n_windows_stride, random_state=seed, randomize=True
            )

            dataloader = DataLoader(
                dataset,
                batch_size=64,
                sampler=train_sampler,
                num_workers=0,
                pin_memory=True,
                drop_last=True,
            )

            # Validation
            domains_val = np.concatenate(
                [[domain_val[i]] * len(X_val[i]) for i in range(len(domain_val))]
            )
            subjects_val_ = np.concatenate(
                [[subjects_val[i]] * len(X_val[i]) for i in range(len(subjects_val))]
            )
            dataset_val = SequenceDataset(
                np.concatenate(X_val, axis=0),
                np.concatenate(y_val, axis=0),
                subjects_val_,
                domains_val,
            )

            valid_sampler = SequenceSampler(
                dataset_val.metadata, n_windows, n_windows_stride, randomize=False
            )

            in_chans, input_size_samples = dataset[0][0].shape

            dataloader_val = DataLoader(
                dataset_val,
                batch_size=64,
                sampler=valid_sampler,
                num_workers=0,
                pin_memory=True,
                drop_last=True,
            )

            # Target
            domains_target = np.concatenate(
                [[dataset_target] * len(X_target[i]) for i in range(len(X_target))]
            )
            subjects_target = np.concatenate(
                [[subject_ids_target[i]] * len(X_target[i]) for i in range(len(subject_ids_target))]
            )

            dataset_t = SequenceDataset(
                np.concatenate(X_target, axis=0),
                np.concatenate(y_target, axis=0),
                subjects_target,
                domains_target,
            )

            target_sampler = SequenceSampler(
                dataset_t.metadata, n_windows, n_windows_stride, randomize=False
            )

            dataloader_target = DataLoader(
                dataset_t,
                batch_size=64,
                sampler=target_sampler,
                num_workers=0,
                pin_memory=True,
                drop_last=True,
            )

            if "online" in tmanorm:
                if not tmalayer:
                    raise ValueError("TMALayer must be provided")
                model = USleepTMA(
                    n_chans=in_chans,
                    sfreq=100,
                    depth=12,
                    with_skip_connection=True,
                    n_outputs=n_classes,
                    n_times=input_size_samples,
                    tmalayer=tmalayer,
                    filter_size=filter_size,
                )
            else:
                model = USleep(
                    n_chans=in_chans,
                    sfreq=100,
                    depth=12,
                    with_skip_connection=True,
                    n_outputs=n_classes,
                    n_times=input_size_samples,
                )

            model.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            warmup_epochs = 10
            scheduler = ChainedScheduler(
                [
                    LinearLR(
                        optimizer,
                        start_factor=0.3333,
                        end_factor=1.0,
                        total_iters=warmup_epochs,
                    ),
                    CosineAnnealingLR(optimizer, T_max=n_epochs - 1, eta_min=lr / 10),
                ]
            )

            # create a dict for history
            history = []

            print("Start training")
            min_val_loss = np.inf
            for epoch in range(n_epochs):
                time_start = time.time()
                model.train()
                train_loss = np.zeros(len(dataloader))
                y_pred_all, y_true_all = list(), list()
                for i, (batch_X, batch_y) in enumerate(dataloader):
                    optimizer.zero_grad()
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)

                    output = model(batch_X)
                    # check if output is nan

                    loss_batch = criterion(output, batch_y)

                    loss_batch.backward()
                    optimizer.step()

                    y_pred_all.append(output.argmax(axis=1).cpu().detach().numpy())
                    y_true_all.append(batch_y.cpu().detach().numpy())
                    train_loss[i] = loss_batch.item()

                # scheduler.step()
                y_pred = np.concatenate(y_pred_all)
                y_true = np.concatenate(y_true_all)
                perf = accuracy_score(y_true.flatten(), y_pred.flatten())
                f1 = f1_score(y_true.flatten(), y_pred.flatten(), average="weighted")

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
                    f1_val = f1_score(y_true.flatten(), y_pred.flatten(), average="weighted")
                    std_f1_val = np.std(f1_val)

                    y_pred_all, y_true_all = list(), list()
                    for i, (batch_X, batch_y) in enumerate(dataloader_target):
                        batch_X = batch_X.to(device)
                        batch_y = batch_y.to(device)
                        output = model(batch_X)

                        y_pred_all.append(output.argmax(axis=1).cpu().detach().numpy())
                        y_true_all.append(batch_y.cpu().detach().numpy())

                    y_pred = np.concatenate(y_pred_all)[:, 10:25]
                    y_true = np.concatenate(y_true_all)[:, 10:25]
                    perf_target = accuracy_score(y_true.flatten(), y_pred.flatten())
                    std_target = np.std(perf_target)
                    f1_target = f1_score(y_true.flatten(), y_pred.flatten(), average="weighted")
                    std_f1_target = np.std(f1_target)

                    score_train = []
                    f1_train = []
                    for i in range(len(X_train)):
                        X = X_train[i]
                        y = y_train[i]

                        y_pred = (
                            model(torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device))
                            .detach()
                            .cpu()
                            .argmax(axis=1)
                        )[0]

                        score_train.append(accuracy_score(y, y_pred))
                        f1_train.append(f1_score(y, y_pred, average="weighted"))

                    perf_night_train = np.mean(score_train)
                    std_night_train = np.std(score_train)
                    f1_night_train = np.mean(f1_train)
                    f1_night_std_train = np.std(f1_train)

                    score_val = []
                    f1_val = []
                    for i in range(len(X_val)):
                        X = X_val[i]
                        y = y_val[i]

                        y_pred = (
                            model(torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device))
                            .detach()
                            .cpu()
                            .argmax(axis=1)
                        )[0]

                        score_val.append(accuracy_score(y, y_pred))
                        f1_val.append(f1_score(y, y_pred, average="weighted"))

                    perf_night_val = np.mean(score_val)
                    std_night_val = np.std(score_val)
                    f1_night_val = np.mean(f1_val)
                    f1_night_std_val = np.std(f1_val)

                    score_target = []
                    f1_target = []
                    for i in range(len(X_target)):
                        X = X_target[i]
                        y = y_target[i]

                        y_pred = (
                            model(torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device))
                            .detach()
                            .cpu()
                            .argmax(axis=1)
                        )[0]

                        score_target.append(accuracy_score(y, y_pred))
                        f1_target.append(f1_score(y, y_pred, average="weighted"))

                    perf_night_target = np.mean(score_target)
                    std_night_target = np.std(score_target)
                    f1_night_target = np.mean(f1_target)
                    f1_night_std_target = np.std(f1_target)

                time_end = time.time()
                history.append({
                    "epoch": epoch,
                    "train_loss": np.mean(train_loss),
                    "train_acc": perf,
                    "train_acc_night": perf_night_train,
                    "train_acc_night_std": std_night_train,
                    "train_f1": f1,
                    "train_f1_night": f1_night_train,
                    "train_f1_night_std": f1_night_std_train,
                    "val_loss": np.mean(val_loss),
                    "val_acc": perf_val,
                    "val_std": std_val,
                    "val_acc_night": perf_night_val,
                    "val_acc_night_std": std_night_val,
                    "val_f1": f1_val,
                    "val_f1_std": std_f1_val,
                    "val_f1_night": f1_night_val,
                    "val_f1_night_std": f1_night_std_val,
                    "target_acc": perf_target,
                    "target_acc_std": std_target,
                    "target_night_acc": perf_night_target,
                    "target_night_acc_std": std_night_target,
                    "target_f1": f1_target,
                    "target_f1_std": std_f1_target,
                    "target_f1_night": f1_night_target,
                    "target_f1_night_std": f1_night_std_target,
                })

                print(
                    "Ep:",
                    epoch,
                    "Loss:",
                    round(np.mean(train_loss), 2),
                    "Acc:",
                    round(np.mean(perf), 2),
                    "AccNight:",
                    round(perf_night_train, 2),
                    "LossVal:",
                    round(np.mean(val_loss), 2),
                    "AccVal:",
                    round(np.mean(perf_val), 2),
                    "AccValNight:",
                    round(perf_night_val, 2),
                    "AccTarNight:",
                    round(perf_night_target, 2),
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
            history_path = f"results/history/history_{module_name}_{dataset_target}_{tmatype}.pkl"
            df_history = pd.DataFrame(history)
            df_history.to_pickle(history_path)

            torch.save(best_model, f"results/models/{module_name}_{dataset_target}_{tmatype}.pt")
            # %%
            n_target = len(X_target)
            results = []
            for n_subj in range(n_target):
                X_t = X_target[n_subj]
                y_t = y_target[n_subj]
                subject = subject_ids_target[n_subj]

                # create dataloader
                dataset_t = SequenceDataset(
                    X_t, y_t, [subject] * len(X_t), [dataset_target] * len(X_t)
                )

                target_sampler = SequenceSampler(
                    dataset_t.metadata, n_windows, n_windows_stride, randomize=False
                )

                dataloader_target = DataLoader(
                    dataset_t,
                    batch_size=16,
                    sampler=target_sampler,
                    num_workers=0,
                    pin_memory=True,
                    drop_last=False,
                )

                y_pred_all, y_true_all = list(), list()
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
                        "module": module_name,
                        "subject": n_subj,
                        "seed": seed,
                        "dataset_t": dataset_target,
                        "y_target": y_t,
                        "y_pred": y_pred,
                        "tmanorm": tmanorm,
                        "tmalayer": tmalayer,
                        "tmatype": tmatype,
                        "filter_size": filter_size,
                        "weight": weight,
                        "n_windows": n_windows,
                        "n_windows_stride": n_windows_stride,
                        "lr": lr,
                    }
                )

            try:
                df_results = pd.read_pickle(results_path)
            except FileNotFoundError:
                df_results = pd.DataFrame()
            df_results = pd.concat((df_results, pd.DataFrame(results)))
            df_results.to_pickle(results_path)
