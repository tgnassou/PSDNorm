import numpy as np
from pathlib import Path
import h5py

import torch
from torch.utils.data import DataLoader
from braindecode.samplers import SequenceSampler
from scipy.signal import convolve

from typing import Iterable

import pandas as pd

from temporal_norm.config import DATA_H5_PATH
from temporal_norm.utils._sampler import BalancedSequenceSampler


class MultiDomainDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        metadata,
        dict_filters=False,
        target_transform=None,
    ):
        self.metadata = metadata.copy()
        self._rename_columns(self.metadata)
        self.dict_filters = dict_filters
        self.target_transform = target_transform

    def _epoching(self, X, size):
        """Create a epoch of size `size` on the data `X`.

        Parameters
        ----------
        X : array, shape=(C, T)
            Data.
        size : int
            Size of the window.

        Returns
        -------
        array, shape=(n_epochs, C, size)
        """
        data = []
        start = 0
        end = size
        step = size
        length = X.shape[-1]
        while end <= length:
            data.append(X[:, start:end])
            start += step
            end += step
        return np.array(data)

    def _convolve(self, X, H):
        window_size = X.shape[-1]
        X = np.concatenate(X, axis=-1)

        C = len(X)
        X_norm = [convolve(X[chan], H[chan]) for chan in range(C)]
        X_norm = np.array(X_norm)

        X_norm = self._epoching(X_norm, window_size)
        return X_norm.astype(np.float32)

    def _rename_columns(self, df):
        df.rename(
            columns={
                "dataset_name": "run",
                "subject_id": "subject",
                "y": "target",
                "sample": "i_window_in_trial",
            },
            inplace=True,
        )
        df["i_start_in_trial"] = np.zeros(len(df), dtype=int)
        df["i_stop_in_trial"] = 3000 * np.ones(len(df), dtype=int)

    def _get_sequence(self, indices):
        X, y = list(), list()
        for idx in indices:
            # path = self.metadata.iloc[idx]["path"]
            dataset = self.metadata.iloc[idx]["run"]
            subject = self.metadata.iloc[idx]["subject"]
            session = self.metadata.iloc[idx]["session"]
            if session == "1.0":
                session = "1"
            elif session == "2.0":
                session = "2"
            elif session == "3.0":
                session = "3"
            elif session == "nan":
                session = "None"
            sample = self.metadata.iloc[idx]["i_window_in_trial"]

            path = Path(DATA_H5_PATH) / f"{dataset}.h5"
            # read h5 part
            with h5py.File(path, "r") as f:
                X.append(f[f"subject_{subject}/session_{session}/{sample}"][:])
            y.append(self.metadata.iloc[idx]["target"])

        X = np.stack(X, axis=0)
        if self.dict_filters:
            dataset_name = self.metadata.iloc[indices[0]]["run"]
            subject_id = self.metadata.iloc[indices[0]]["subject"]
            X = self._convolve(
                X, self.dict_filters[(dataset_name, subject_id)]
            )

        y = np.array(y)
        if self.target_transform:
            y = self.target_transform(y)

        return X, y

    def __getitem__(self, idx):
        if not isinstance(idx, Iterable):
            raise ValueError("idx must be an iterable.")
        return self._get_sequence(idx)

    def __len__(self):
        """Return the total number of samples in the flattened index."""
        return len(self.metadata)


def filter_metadata(metadata, dataset_names, subject_ids=None):
    metadata = metadata.copy()
    metadata_filtered = pd.DataFrame()
    for dataset_name in dataset_names:
        metadata_per_dataset = metadata[metadata.dataset_name == dataset_name]
        if dataset_name == "MASS":
            metadata_per_dataset = metadata_per_dataset.fillna(0)
        if subject_ids is not None:
            metadata_per_dataset = metadata_per_dataset[
                metadata_per_dataset.subject_id.isin(subject_ids[dataset_name])
            ]
        metadata_filtered = pd.concat(
            [metadata_filtered, metadata_per_dataset],
            axis=0
        )
    metadata_filtered.reset_index(drop=True, inplace=True)
    return metadata_filtered


def get_subject_ids(metadata, dataset_names):
    subject_ids = dict()
    for dataset_name in dataset_names:
        subject_ids[dataset_name] = metadata[
            metadata.dataset_name == dataset_name
        ].subject_id.unique()
    return subject_ids


def get_dataloader(
    metadata,
    dataset_names,
    subject_ids,
    n_windows,
    n_windows_stride,
    batch_size,
    num_workers,
    dict_filters=None,
    randomize=True,
    balanced=None,
    target_transform=None,
):
    metadata = filter_metadata(metadata, dataset_names, subject_ids)
    dataset = MultiDomainDataset(metadata, dict_filters=dict_filters, target_transform=target_transform)
    if balanced:
        probs = get_probs(metadata, dataset_names)
        n_sequences = int(len(metadata) / 10)
        sampler = BalancedSequenceSampler(
            dataset.metadata,
            n_windows=n_windows,
            n_windows_stride=n_windows_stride,
            random_state=42,
            probs=probs,
            n_sequences=n_sequences,
        )
    else:
        sampler = SequenceSampler(
            dataset.metadata,
            n_windows=n_windows,
            n_windows_stride=n_windows_stride,
            random_state=42,
            randomize=randomize,
        )
    dataloader = DataLoader(
        dataset, batch_size=batch_size,
        sampler=sampler, num_workers=num_workers
    )
    return dataloader


def get_probs(metadata, dataset_names, alpha=0.5):
    metadata["sub+session"] = metadata.apply(lambda x: f"{x['subject_id']}_{x['session']}", axis=1)
    length = {}
    for dataset in dataset_names:
        length[dataset] = metadata[metadata["dataset_name"] == dataset]["sub+session"].nunique()

    probs = {}
    for dataset in dataset_names:
        probs[dataset] = alpha / len(dataset_names) + (1 - alpha) * (1 / length[dataset]) / sum([1 / length[dataset] for dataset in dataset_names])
    return probs