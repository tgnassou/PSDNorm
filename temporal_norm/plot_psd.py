# %%
import numpy as np

from braindecode.samplers import SequenceSampler

from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from skorch.dataset import Dataset

from monge_alignment.utils import MongeAlignment
import matplotlib.pyplot as plt
from typing import Iterable

import torch
from torch import nn
import torch.fft

from ms3.utils._PSDNorm import PSDNorm, welch_psd
# import DAtaloader
from tqdm import tqdm

import pandas as pd

device = "cuda:1" if torch.cuda.is_available() else "cpu"
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
    # y_pred = y_pred[:, 12:24]
    # y = y[:, 12:24]
    return accuracy_score(y.flatten(), y_pred.flatten())


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
max_epochs = 300
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
dataset_target = "ABC"
# for dataset_target in dataset_names:
# X_target, y_target, subject_ids_target = data_dict[dataset_target]
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

# load state dict
model = torch.load("results/models/usleep_MASS.pt")
# %%

# %%

train_sampler = SequenceSampler(
    dataset.metadata, n_windows, n_windows_stride, random_state=seed, randomize=False
)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    sampler=train_sampler,
)

# %%
encoder = model.encoder[0].block_prepool[:2].to("cpu")
PSDNorm = PSDNorm(filter_size=128) 
PSDNorm_2 = PSDNorm(filter_size=128)

X_flatten_all = []
psd_input = []
psd_input_norm = []
psd_output_norm = []
psd_output = []
psd_output_2 = []
y_all = []
# encoder.eval()
# PSDNorm.eval()
with torch.no_grad():
    for batch in dataloader:
        X, y = batch
        y_all.append(y)
        X_flatten = X.permute(0, 2, 1, 3)  # (B, C, S, T)
        X_flatten = X_flatten.flatten(start_dim=2)
        X_flatten_all.append(X_flatten)
        psd_input.append(welch_psd(X_flatten, window="hann", nperseg=128)[1])
        output = encoder(X_flatten)

        psd_output.append(welch_psd(output, window="hann", nperseg=128)[1])

        X_flatten_filtered = PSDNorm(X_flatten)
        psd_input_norm.append(welch_psd(X_flatten_filtered, window="hann", nperseg=128)[1])
        output_filtered = encoder(X_flatten_filtered)
        psd_output_norm.append(welch_psd(output_filtered, window="hann", nperseg=128)[1])

        output_filtered_2 = PSDNorm_2(output_filtered)

        psd_output_2.append(welch_psd(output_filtered_2, window="hann", nperseg=128)[1])

# %%

freqs = welch_psd(X_flatten, window="hann", nperseg=128)[0]
psd_input = torch.cat(psd_input, axis=0)
psd_input_norm = torch.cat(psd_input_norm, axis=0)
psd_output = torch.cat(psd_output, axis=0)
psd_output_norm = torch.cat(psd_output_norm, axis=0)
psd_output_2 = torch.cat(psd_output_2, axis=0)
X_flatten = torch.cat(X_flatten_all, axis=0)

y_all = torch.cat(y_all, axis=0)
# %%
fig, axes = plt.subplots(3, 2, figsize=(10, 10), sharex=True, )

axes[0, 0].plot(freqs, psd_input[:, 0].T.cpu().numpy(), alpha=0.3, color="cornflowerblue")
axes[0, 0].set_xlim(0, 0.2)
axes[0, 0].set_title("PSD of one batch in the input space")
axes[0, 0].set_ylabel("Power")


axes[0, 1].plot(freqs, psd_output[:, 0].T.cpu().numpy(), alpha=0.3, color="cornflowerblue")
axes[0, 1].set_xlim(0, 0.2)
axes[0, 1].set_title("PSD of one batch after one encoder")

axes[1, 0].plot(
    freqs, psd_input_norm[:, 0].T.cpu().numpy(), alpha=0.3, color="cornflowerblue"
)
axes[1, 0].set_xlim(0, 0.2)
axes[1, 0].set_title("PSD of one batch in the input space after TMA")
axes[1, 0].set_xlabel("Frequency (Hz)")
axes[1, 0].set_ylabel("Power")

axes[1, 1].plot(
    freqs, psd_output_norm[:, 0].T.cpu().numpy(), alpha=0.3, color="cornflowerblue"
)
axes[1, 1].set_xlim(0, 0.2)
axes[1, 1].set_title("PSD of one batch after TMA and one encoder")

axes[2, 1].plot(
    freqs, psd_output_2[:, 0].T.cpu().numpy(), alpha=0.3, color="cornflowerblue"
)
axes[2, 1].set_xlim(0, 0.2)
axes[2, 1].set_title("PSD of one batch after TMA, one encoder and one other TMA")
axes[2, 1].set_xlabel("Frequency (Hz)")
axes[2, 1].set_ylabel("Power")
plt.tight_layout()

axes[2, 0].axis("off")

# %%
# %%
plt.plot(psd_input[200:211, 0].T.cpu().numpy(), alpha=0.7, color="cornflowerblue")
plt.xlim(0, 20)
plt.title("PSD of one batch in the input space")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")

# %%
plt.plot(X_flatten[100, 0].cpu().numpy())
plt.xlim(0, 3000)

# %%
plt.plot(freqs, psd_input_norm[:, 0].T.cpu().numpy(), alpha=0.3, color="cornflowerblue")
plt.xlim(0, 0.2)
plt.title("PSD of one batch in the input space after TMA")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")

# %%
plt.plot(freqs, psd_output[:, 0].T.cpu().numpy(), alpha=0.3, color="cornflowerblue")
plt.xlim(0, 0.2)
plt.title("PSD of one batch after one encoder")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")

# %%
plt.plot(freqs, psd_output_norm[:, 0].T.cpu().numpy(), alpha=0.3, color="cornflowerblue")
plt.xlim(0, 0.2)
plt.title("PSD of one batch after TMA and one encoder")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")

# %%
plt.plot(freqs, psd_output_2[:, 0].T.cpu().numpy(), alpha=0.3, color="cornflowerblue")
plt.xlim(0, 0.2)
plt.title("PSD of one batch after TMA, one encoder and one other TMA")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")