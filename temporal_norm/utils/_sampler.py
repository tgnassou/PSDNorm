import numpy as np
from torch.utils.data.sampler import Sampler
from torch.utils.data.distributed import DistributedSampler
from sklearn.utils import check_random_state
from typing import Optional
from braindecode.samplers import RecordingSampler


class BalancedSequenceSampler(RecordingSampler):
    """Sample sequences of consecutive windows with balanced classes.

    Parameters
    ----------
    metadata : pd.DataFrame
        See RecordingSampler.
    n_windows : int
        Number of consecutive windows in a sequence.
    n_windows_stride : int
        Number of windows between two consecutive sequences.
    probs : dict
        Dictionary with the probability of sampling each dataset.
    random_state : np.random.RandomState | int | None
        Random state.
    n_sequences : int
        Number of sequences to sample.


    Attributes
    ----------
    info : pd.DataFrame
        See RecordingSampler.
    """

    def __init__(
        self, metadata, n_windows, n_windows_stride, probs, random_state=None, n_sequences=int(1e6),
    ):
        super().__init__(metadata, random_state=random_state)
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
        start_inds = self.start_inds.copy()
        ind_dataset = self.ind_dataset.copy()
        for _ in range(self.n_sequences):
            dataset = self.sample_dataset()
            idx_selected = np.where(ind_dataset == dataset)[0]
            id_selected = np.random.choice(idx_selected)
            start_ind = start_inds[id_selected]
            yield tuple(range(start_ind, start_ind + self.n_windows)), dataset
