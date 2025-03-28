from numbers import Integral
import numpy as np


def get_center_label(x):
    if isinstance(x, Integral):
        return x
    return x[np.ceil(len(x) / 2).astype(int)] if len(x) > 1 else x


def get_probs(metadata, dataset_names, alpha=0.5):
    metadata["sub+session"] = metadata.apply(lambda x: f"{x['subject_id']}_{x['session']}", axis=1)
    length = {}
    for dataset in dataset_names:
        length[dataset] = metadata[metadata["dataset_name"] == dataset]["sub+session"].nunique()

    probs = {}
    for dataset in dataset_names:
        probs[dataset] = alpha / len(dataset_names) + (1 - alpha) * (1 / length[dataset]) / sum([1 / length[dataset] for dataset in dataset_names])
    return probs
