# %%
from collections import defaultdict, Counter
import numpy as np

from braindecode import EEGClassifier
from braindecode.samplers import SequenceSampler, BalancedSequenceSampler
from braindecode.models import USleep

from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from torch.optim.lr_scheduler import CosineAnnealingLR, ChainedScheduler, LinearLR
from skorch.callbacks import EarlyStopping, EpochScoring, LRScheduler, Callback
from skorch.helper import predefined_split
from skorch.dataset import Dataset
from numbers import Integral
from braindecode.samplers import RecordingSampler

from monge_alignment.utils import MongeAlignment

from typing import Iterable

import torch
from torch import nn

# import DAtaloader
from tqdm import tqdm

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


def accuracy_multi(model, X, y):
    y_pred = model.predict(X)
    acc = []
    for i in range(len(y)):
        acc.append(accuracy_score(y[i], y_pred[i]))
    return np.mean(acc)


# %%
dataset_names = [
    "ABC",
    "CHAT",
    "CFS",
    "SHHS",
    # "HOMEPAP",
    # "CCSHS",
    # "MASS",
    # "Physionet",
    # "SOF",
    # "MROS",
]

data_dict = {}
max_subjects = 3
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
max_epochs = 200
batch_size = 64
patience = 30
n_jobs = 1
seed = 42
lr = 1e-3
weight = "unbalanced"
use_scheduler = False

scaling = "None"
results_path = (
    f"results/pickle/results_usleep_{scaling}_"
    f"{len(dataset_names)}_dataset_with_{max_subjects}"
    f"_subjects_scheduler_{use_scheduler}_lr_{lr}.pkl"
)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
rng = check_random_state(seed)
dataset_target = "CHAT"
# for dataset_target in dataset_names:
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

if scaling == "subject":
    ma = MongeAlignment(n_jobs=n_jobs)
    X_train = ma.fit_transform(X_train)
    X_val = ma.transform(X_val)
    X_target = ma.transform(X_target)
    del ma

# %%

domains = np.concatenate(
    [[domain_train[i]] * len(X_train[i]) for i in range(len(domain_train))]
)
subjects = np.concatenate(
    [[subjects_train[i]] * len(X_train[i]) for i in range(len(subjects_train))]
)
dataset = SequenceDataset(
    np.concatenate(X_train, axis=0), np.concatenate(y_train, axis=0), subjects, domains
)
# %%
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

n_chans, n_time = X_train[0][0].shape
n_classes = len(np.unique(y_train[0]))
# %%

train_sampler = SequenceSampler(
    dataset.metadata, n_windows, n_windows_stride, random_state=seed, randomize=False
)

# %%

valid_sampler = SequenceSampler(
    dataset_val.metadata, n_windows, n_windows_stride, randomize=False
)

in_chans, input_size_samples = dataset[0][0].shape

model = USleep(
    n_chans=in_chans,
    sfreq=100,
    depth=12,
    with_skip_connection=True,
    n_outputs=n_classes,
    n_times=input_size_samples,
)

# %%

train_acc = EpochScoring(
    scoring=accuracy_multi,
    on_train=True,
    name="train_acc",
    lower_is_better=False,
)
valid_acc = EpochScoring(
    scoring=accuracy_multi,
    on_train=False,
    name="valid_acc",
    lower_is_better=False,
)

warmup_epochs = 10


# Chained Scheduler
def scheduler(optimizer, last_epoch):
    return ChainedScheduler(
        [
            LinearLR(
                optimizer,
                start_factor=0.3333,
                end_factor=1.0,
                total_iters=warmup_epochs,
            ),
            CosineAnnealingLR(optimizer, T_max=max_epochs - 1, eta_min=lr / 10),
        ]
    )


callbacks = [
    ("train_acc_multi", train_acc),
    ("valid_acc_multi", valid_acc),
    # (
    #     "early_stopping",
    #     EarlyStopping(
    #         monitor="valid_loss",
    #         patience=patience,
    #         load_best=True,
    #         lower_is_better=True,
    #     ),
    # ),
]
if use_scheduler:
    callbacks = [
        ("train_acc_multi", train_acc),
        ("valid_acc_multi", valid_acc),
        (
            "early_stopping",
            EarlyStopping(
                monitor="valid_loss",
                patience=patience,
                load_best=True,
                lower_is_better=True,
            )
        ),
        (
            "lr_scheduler", LRScheduler(scheduler)
        )
    ]

print(callbacks)

clf = EEGClassifier(
    module=model,
    max_epochs=max_epochs,
    batch_size=batch_size,
    criterion=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam,
    optimizer__lr=lr,
    iterator_train__shuffle=False,
    iterator_train__sampler=train_sampler,
    iterator_valid__sampler=valid_sampler,
    device=device,
    train_split=predefined_split(dataset_val),
    callbacks=callbacks,
    classes=np.unique(y_train[0]),
)
clf.set_params(callbacks__valid_acc=None)

clf.fit(dataset, y=None)


# %%
module = clf.module_
score_train = []
for i in range(len(X_train)):
    X = X_train[i]
    y = y_train[i]

    y_pred = (
        module(torch.tensor(X, dtype=torch.float32).to(device))
        .detach()
        .cpu()
        .argmax(axis=1)
    )

    score_train.append(accuracy_score(y, y_pred))

print(f"Train accuracy: {np.mean(score_train)}")

score_val = []
for i in range(len(X_val)):
    X = X_val[i]
    y = y_val[i]

    y_pred = (
        module(torch.tensor(X, dtype=torch.float32).to(device))
        .detach()
        .cpu()
        .argmax(axis=1)
    )

    score_val.append(accuracy_score(y, y_pred))

print(f"Validation accuracy: {np.mean(score_val)}")

score_target = []
for i in range(len(X_target)):
    X = X_target[i]
    y = y_target[i]

    y_pred = (
        module(torch.tensor(X, dtype=torch.float32).to(device))
        .detach()
        .cpu()
        .argmax(axis=1)
    )

    score_target.append(accuracy_score(y, y_pred))

print(f"Target accuracy: {np.mean(score_target)}")

# %%
# create dataloader
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=64,
    sampler=train_sampler,
    num_workers=0,
    pin_memory=True,
    drop_last=True,
)

# %%
# get the first batch
loss = []
accuracy = []
criterion = nn.CrossEntropyLoss()
for i, (X, y) in enumerate(dataloader):
    X = X.to(device)
    y = y.to(device)
    output = module(X)
    y_pred = output.argmax(axis=1)
    loss.append(criterion(output, y).item())
    accuracy.append((y == y_pred).float().mean().item())
print(np.mean(loss))
print(np.mean(accuracy))

# %%
# get the output of the model

output = module(X)

# %%
criterion = nn.CrossEntropyLoss()
loss = criterion(output, y)
import ipdb; ipdb.set_trace()
# %% save mdule

# torch.save(module, f"results/models/{module_name}_{dataset_target}.pt")
# # %%
# n_target = len(X_target)
# results = []
# for n_subj in range(n_target):
#     X_t = X_target[n_subj]
#     y_t = y_target[n_subj]
#     subject = subject_ids_target[n_subj]

#     y_pred = clf.module_(
#         torch.tensor(X_t, dtype=torch.float32).to(device)
#     ).cpu().detach().numpy().argmax(axis=1)
#     # %%
#     results.append(
#         {
#             "module": module_name,
#             "subject": n_subj,
#             "seed": seed,
#             "dataset_t": dataset_target,
#             "y_target": y_t,
#             "y_pred": y_pred,
#             "scaling": scaling,
#             "weight": weight,
#             "n_windows": n_windows,
#             "n_windows_stride": n_windows_stride,
#             "lr": lr,
#         }
#     )
# # %%
# try:
#     df_results = pd.read_pickle(results_path)
# %%

# except FileNotFoundError:
#     df_results = pd.DataFrame()
# df_results = pd.concat((df_results, pd.DataFrame(results)))
# df_results.to_pickle(results_path)
