import numpy as np
from pathlib import Path
import h5py
from functools import lru_cache

import torch
from torch.utils.data import DataLoader
from braindecode.samplers import SequenceSampler
from scipy.signal import convolve

from typing import Iterable

import pandas as pd

from temporal_norm.config import DATA_H5_PATH
from temporal_norm.utils._sampler import BalancedSequenceSampler
from temporal_norm.utils._functions import get_probs


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

        # Convert metadata columns to NumPy arrays for fast indexing
        self.runs = self.metadata["run"].values
        self.subjects = self.metadata["subject"].values
        self.sessions = self.metadata["session"].astype(str).values
        self.samples = self.metadata["i_window_in_trial"].values
        self.targets = self.metadata["target"].values

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

    @lru_cache(maxsize=32)
    def _get_h5_file(self, dataset):
        return h5py.File(Path(DATA_H5_PATH) / f"{dataset}.h5", "r")

    def _get_sequence(self, indices):
        indices = np.asarray(indices)

        # Fetch values for all indices at once
        datasets = self.runs[indices]
        subjects = self.subjects[indices]
        sessions = self.sessions[indices]

        # Check that all entries refer to the same dataset / subject / session
        if not (np.all(datasets == datasets[0]) and
                np.all(subjects == subjects[0]) and
                np.all(sessions == sessions[0])):
            import warnings
            warnings.warn(
                f"Be careful, indices {indices} do not correspond to the same subject/session."
                "This may lead to unexpected behavior."
            )

        dataset = datasets[0]
        subject = subjects[0]
        session = sessions[0]

        # Normalize session name
        session_map = {"1.0": "1", "2.0": "2", "3.0": "3", "nan": "None"}
        session = session_map.get(session, session)

        # Get sample indices
        sample_indices = self.samples[indices]
        
        # Check if the samples are contiguous
        # THIS SHOULD BE REMOVED IN THE FUTURE
        if np.all(np.diff(sample_indices) == 1):
            first_sample = sample_indices[0]
            last_sample = sample_indices[-1] + 1
        else:
            # print("Non-contiguous samples detected. This should not happen.")
            first_sample = np.min(sample_indices)
            last_sample = first_sample + len(sample_indices)

        # Read data from HDF5
        f = self._get_h5_file(dataset)
        X = f[f"subject_{subject}/session_{session}"][first_sample:last_sample]

        y = self.targets[indices]

        if self.target_transform:
            y = self.target_transform(y)
        return X, y, subject, session

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
    pin_memory,
    persistent_workers,
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
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return dataloader
