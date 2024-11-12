# %%
import numpy as np

from braindecode.models import SleepStagerChambon2018, TimeDistributed
from braindecode import EEGClassifier
from braindecode.samplers import RecordingSampler

from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split

from skorch.callbacks import EarlyStopping
from skorch.helper import predefined_split
from skorch.dataset import Dataset

from monge_alignment.utils import MongeAlignment

from typing import Iterable
from numbers import Integral

import torch
from torch import nn

from tqdm import tqdm

import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"


class dataset_sequence(Dataset):
    def __init__(self, X, y, target_transform=None):
        super().__init__(X=X, y=y)
        self.target_transform = target_transform

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
        """
        Parameters
        ----------
        idx : int | list
            Index of window and target to return. If provided as a list of
            ints, multiple windows and targets will be extracted and
            concatenated. The target output can be modified on the
            fly by the ``traget_transform`` parameter.
        """
        if isinstance(idx, Iterable):  # Sample multiple windows
            item = self._get_sequence(idx)
        else:
            item = super().__getitem__(idx)
        if self.target_transform is not None:
            item = item[:1] + (self.target_transform(item[1]),) + item[2:]

        return item


class DomainAwareSequenceSampler(RecordingSampler):
    """Sample sequences of consecutive windows.

    Parameters
    ----------
    metadata : pd.DataFrame
        See RecordingSampler.
    n_windows : int
        Number of consecutive windows in a sequence.
    n_windows_stride : int
        Number of windows between two consecutive sequences.
    random_state : np.random.RandomState | int | None
        Random state.

    Attributes
    ----------
    info : pd.DataFrame
        See RecordingSampler.
    file_ids : np.ndarray of ints
        Array of shape (n_sequences,) that indicates from which file each
        sequence comes from. Useful e.g. to do self-ensembling.
    """
    def __init__(self, metadata, n_windows, n_windows_stride, random=False,
                 random_state=None):
        super().__init__(metadata, random_state=random_state)
        self.random = random
        self.n_windows = n_windows
        self.n_windows_stride = n_windows_stride
        self.start_inds, self.file_ids = self._compute_seq_start_inds()

    def _compute_seq_start_inds(self):
        """Compute sequence start indices.

        Returns
        -------
        np.ndarray :
            Array of shape (n_sequences,) containing the indices of the first
            windows of possible sequences.
        np.ndarray :
            Array of shape (n_sequences,) containing the unique file number of
            each sequence. Useful e.g. to do self-ensembling.
        """
        end_offset = 1 - self.n_windows if self.n_windows > 1 else None
        start_inds = self.info['index'].apply(
            lambda x: x[:end_offset:self.n_windows_stride]).values
        file_ids = [[i] * len(inds) for i, inds in enumerate(start_inds)]
        return np.concatenate(start_inds), np.concatenate(file_ids)

    def __len__(self):
        return len(self.start_inds)

    def __iter__(self):
        if self.random:
            start_inds = self.start_inds.copy()
            self.rng.shuffle(start_inds)
            for start_ind in start_inds:
                yield tuple(range(start_ind, start_ind + self.n_windows))
        else:
            for start_ind in self.start_inds:
                yield tuple(range(start_ind, start_ind + self.n_windows))



# %%
dataset_names = [
    "ABC",
    "CHAT",
    # "CFS",
    # "SHHS",
    # "HOMEPAP",
    # "CCSHS",
    # "MASS",
    # "Physionet",
    # "SOF",
    # "MROS",
]

data_dict = {}
max_subjects = 100
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
module_name = "chambon"
max_epochs = 150
batch_size = 128
patience = 15
n_jobs = 10
seed = 42
weight = "unbalanced"

# %%

for scaling in ["subject", "None"]:
    results_path = (
    f"results/pickle/results_LODO_sequence_{module_name}_{scaling}_"
    f"{len(dataset_names)}_dataset_with_{max_subjects}_subjects_{weight}.pkl"
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
            ) = train_test_split(
                X_, y_, subjects_, test_size=valid_size
            )

            X_train += X_train_
            X_val += X_val_
            y_train += y_train_
            y_val += y_val_
            subjects_train += subjects_train_
            subjects_val += subjects_val_
            domain_train += [dataset_source] * len(X_train_)
            domain_val += [dataset_source] * len(X_val_)
            print(
                f"Dataset {dataset_source}: {len(X_train_)}"
                f" train, {len(X_val_)} val"
            )

    if scaling == "subject":
        ma = MongeAlignment(n_jobs=n_jobs)
        X_train = ma.fit_transform(X_train)
        X_val = ma.transform(X_val)
        X_target = ma.transform(X_target)
        del ma
    # %%
    dataset = dataset_sequence(
        np.concatenate(X_train, axis=0), np.concatenate(y_train, axis=0)
    )
    dataset_val = dataset_sequence(
        np.concatenate(X_val, axis=0), np.concatenate(y_val, axis=0)
    )

    n_chans, n_time = X_train[0][0].shape
    n_classes = len(np.unique(y_train[0]))
    # %%

    n_windows = 11
    n_windows_stride = 1
    n_sequences = int(len(dataset) / n_windows)

    md = pd.DataFrame(
        {
            "target": np.concatenate(y_train, axis=0),
            "subject": np.concatenate(
                [
                    [subjects_train[i]] * len(X_train[i])
                    for i in range(len(subjects_train))
                ],
                axis=0,
            ),
            "run": np.concatenate(
                [
                    [domain_train[i]] * len(X_train[i])
                    for i in range(len(domain_train))
                ],
                axis=0,
            ),
            "i_window_in_trial": np.zeros(len(dataset)),
            "i_start_in_trial": np.zeros(len(dataset)),
            "i_stop_in_trial": 3000 * np.ones(len(dataset)),
        }
    )
    train_sampler = SequenceSampler(
        md, n_windows, n_windows_stride, random=True, random_state=seed
    )

    n_sequences_val = int(len(dataset_val) / n_windows)

    md_val = pd.DataFrame(
        {
            "target": np.concatenate(y_val, axis=0),
            "subject": np.concatenate(
                [
                    [subjects_val[i]] * len(X_val[i])
                    for i in range(len(subjects_val))
                ],
                axis=0,
            ),
            "run": np.concatenate(
                [
                    [domain_val[i]] * len(X_val[i])
                    for i in range(len(domain_val))
                ],
                axis=0,
            ),
            "i_window_in_trial": np.zeros(len(dataset_val)),
            "i_start_in_trial": np.zeros(len(dataset_val)),
            "i_stop_in_trial": 3000 * np.ones(len(dataset_val)),
        }
    )

    valid_sampler = SequenceSampler(md_val, n_windows, n_windows_stride)
    # %%    

    # Use label of center window in the sequence
    def get_center_label(x):
        if isinstance(x, Integral):
            return x
        return x[np.ceil(len(x) / 2).astype(int)] if len(x) > 1 else x


    dataset.target_transform = get_center_label
    dataset_val.target_transform = get_center_label

    feat_extractor = SleepStagerChambon2018(
        n_chans, sfreq=100, n_outputs=n_classes, return_feats=True
    )

    module = nn.Sequential(
        TimeDistributed(feat_extractor),  # apply model on each 30-s window
        nn.Sequential(  # apply linear layer on concatenated feature vectors
            nn.Flatten(start_dim=1),
            nn.Dropout(0.5),
            nn.Linear(feat_extractor.len_last_layer * n_windows, n_classes),
        ),
    )
    # %%

    callbacks = [
        (
            "early_stopping",
            EarlyStopping(
                monitor="valid_loss",
                patience=patience,
                load_best=True,
                lower_is_better=True,
            ),
        ),
    ]
    # %%

    clf = EEGClassifier(
        module=module,
        max_epochs=max_epochs,
        batch_size=batch_size,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam,
        iterator_train__shuffle=False,
        iterator_train__sampler=train_sampler,
        iterator_valid__sampler=valid_sampler,
        optimizer__lr=1e-3,
        device=device,
        train_split=predefined_split(dataset_val),
        callbacks=callbacks,
    )

    clf.fit(dataset, y=None)

    n_target = len(X_target)
    results = []
    for n_subj in range(n_target):
        X_t = X_target[n_subj]
        y_t = y_target[n_subj]
        subject = subject_ids_target[n_subj]
        md_target = pd.DataFrame(
            {
                "target": y_t,
                "subject": [subject] * len(y_t),
                "run": [0] * len(y_t),
                "i_window_in_trial": np.zeros(len(y_t)),
                "i_start_in_trial": np.zeros(len(y_t)),
                "i_stop_in_trial": 3000 * np.ones(len(y_t)),
            }
        )
        target_sampler = SequenceSampler(md_target, n_windows, n_windows_stride=1)
        dataset_test = dataset_sequence(X_t, y_t)
        dataset_test.target_transform = get_center_label

        # define clf test

        clf_test = EEGClassifier(
            module=clf.module_,
            max_epochs=2,
            batch_size=batch_size,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam,
            iterator_train__shuffle=False,
            iterator_train__sampler=target_sampler,
            iterator_valid__sampler=target_sampler,
            optimizer__lr=1e-3,
            device=device,
            train_split=predefined_split(dataset_test),
            callbacks=callbacks,
        )

        # initialize clf with the best model
        clf_test.initialize()

        y_pred = clf_test.predict(dataset_test)
        results.append(
            {
                "module": module_name,
                "subject": n_subj,
                "seed": seed,
                "dataset_t": dataset_target,
                "y_target": y_t[n_windows // 2:-(n_windows // 2)],
                "y_pred": y_pred,
                "scaling": scaling,
                "weight": weight,
            }
        )
    try:
        df_results = pd.read_pickle(results_path)
    except FileNotFoundError:
        df_results = pd.DataFrame()
    df_results = pd.concat((df_results, pd.DataFrame(results)))
    df_results.to_pickle(results_path)

    # %%
