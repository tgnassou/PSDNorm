# %%
import numpy as np
import pandas as pd
from temporal_norm.utils._dataset import MultiDomainDataset
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
metadata["sub+session"] = metadata.apply(lambda x: f"{x['subject_id']}_{x['session']}", axis=1)
# %%
# get lenght of dataset_names
length = {}
for dataset in dataset_names:
    length[dataset] = metadata[metadata["dataset_name"] == dataset]["sub+session"].nunique()

# %%
# create probability of draw a dataset 
probs = {}
alpha = 0.5
for dataset in dataset_names:
    probs[dataset] = alpha / len(dataset_names) + (1 - alpha) * (1 / length[dataset]) / sum([1 / length[dataset] for dataset in dataset_names])
# %%

# pick a dataset
dataset_n = np.random.choice(list(prob.keys()), p=list(prob.values()))

# %%
dataset = MultiDomainDataset(metadata, )
# %%
dataset.metadata.query("run == @dataset_n")
# %%
from braindecode.samplers import RecordingSampler
class SequenceSampler(RecordingSampler):
    """Sample sequences of consecutive windows.

    Parameters
    ----------
    metadata : pd.DataFrame
        See RecordingSampler.
    n_windows : int
        Number of consecutive windows in a sequence.
    n_windows_stride : int
        Number of windows between two consecutive sequences.
    random : bool
        If True, sample sequences randomly. If False, sample sequences in
        order.
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

    def __init__(
        self, metadata, n_windows, n_windows_stride, probs, randomize=False, random_state=None, n_sequences=int(1e6),
    ):
        super().__init__(metadata, random_state=random_state)
        self.randomize = randomize
        self.n_windows = n_windows
        self.n_sequences = n_sequences
        self.n_windows_stride = n_windows_stride
        self.start_inds, self.ind_dataset = self._compute_seq_start_inds()
        self.probs = probs

    def sample_dataset(self, ):
        """Return a random dataset.

        Returns
        -------
        int
            Sampled class.
        int
            Index to the recording the class was sampled from.
        """
        return self.rng.choice(list(self.probs.keys()), p=list(self.probs.values()))

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
        start_inds = (
            self.info["index"]
            .apply(lambda x: x[: end_offset : self.n_windows_stride])
            .values
        )
        # get the run of each start multiindex
        ind_dataset = self.info.index.get_level_values(2)
        ind_dataset = [[ind_dataset[i]]*len(inds) for i, inds in enumerate(start_inds)]
        
        start_inds = np.concatenate(start_inds)
        ind_dataset = np.concatenate(ind_dataset)

        return start_inds, ind_dataset

    def __len__(self):
        return len(self.start_inds)

    def __iter__(self):
        for _ in range(self.n_sequences):
            dataset = self.sample_dataset()
            start_inds = self.start_inds.copy()
            ind_dataset = self.ind_dataset.copy()
            idx_selected = np.where(ind_dataset == dataset)[0]
            id_selected = np.random.choice(idx_selected)
            start_ind = start_inds[id_selected]
            yield tuple(range(start_ind, start_ind + self.n_windows)), dataset


# %%
sampler = SequenceSampler(
    dataset.metadata,
    n_windows=35,
    n_windows_stride=1,
    random_state=42,
    randomize=False,
    probs=probs,
)

# %%
for inds in sampler:
    print(inds)
    if inds[1] == "SHHS":
        break
# %%
[len(inds) for inds in sampler.start_inds]
# %%
sampler.rng.choice(list(sampler.probs.keys()), p=list(sampler.probs.values()))
# %%
list(sampler.probs.keys())
# %%
probs.keys()
# %%
