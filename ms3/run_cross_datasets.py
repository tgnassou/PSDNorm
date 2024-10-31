# %%
import numpy as np

from braindecode.models import SleepStagerChambon2018
from braindecode import EEGClassifier

from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split

from skorch.callbacks import EarlyStopping
from skorch.helper import predefined_split
from skorch.dataset import Dataset

from monge_alignment.utils import MongeAlignment

import torch
from torch import nn

from tqdm import tqdm

import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"


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
    for subject in tqdm(subject_ids_):
        X_.append(np.load(f"data/{dataset_name}/X_{subject}.npy"))
        y_.append(np.load(f"data/{dataset_name}/y_{subject}.npy"))
        if len(X_) == max_subjects:
            break
    data_dict[dataset_name] = [X_, y_, subject_ids_]

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
        f"results/pickle/results_cross_dataset_{module_name}_{scaling}_"
        f"{len(dataset_names)}_dataset_with_{max_subjects}"
        "_subjects_{weight}.pkl"
    )
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    rng = check_random_state(seed)
    for dataset_source in dataset_names:
        print(f"Source: {dataset_source}")

        X_, y_, _ = data_dict[dataset_source]
        valid_size = 0.2
        (
            X_train,
            X_val,
            y_train,
            y_val,
        ) = train_test_split(
            X_, y_, test_size=valid_size
        )

        if scaling == "subject":
            ma = MongeAlignment(
                method="temp",
                filter_size=256,
                reg=1e-3,
                concatenate_epochs=True,
                n_jobs=10,
            )
            X_train = ma.fit_transform(X_train)
            X_val = ma.transform(X_val)

        n_chans, n_time = X_train[0][0].shape
        n_classes = len(np.unique(y_train[0]))

        valid_dataset = Dataset(
            np.concatenate(X_val, axis=0), np.concatenate(y_val, axis=0)
        )
        class_weights = compute_class_weight(
            "balanced",
            classes=np.unique(np.concatenate(y_train)),
            y=np.concatenate(y_train)
        )
        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)

        module = SleepStagerChambon2018(
            n_chans=n_chans, n_outputs=n_classes, sfreq=100
        )
        clf = EEGClassifier(
            module=module,
            max_epochs=max_epochs,
            batch_size=batch_size,
            criterion=nn.CrossEntropyLoss(
                weight=torch.Tensor(class_weights).to(device)
                if weight == "balanced" else None
            ),
            optimizer=torch.optim.Adam,
            iterator_train__shuffle=True,
            iterator_train__num_workers=20,
            optimizer__lr=1e-3,
            device=device,
            train_split=predefined_split(valid_dataset),
            callbacks=[
                (
                    "early_stopping",
                    EarlyStopping(
                        monitor="valid_loss",
                        patience=patience,
                        load_best=True
                    ),
                )
            ],
            classes=np.arange(n_classes),
        )
        clf.fit(X_train, y_train)
        results = []
        for dataset_target in dataset_names:
            if dataset_source != dataset_target:
                X_target, y_target, _ = data_dict[dataset_target]
                if scaling == "subject":
                    X_target = ma.transform(X_target)

                n_target = len(X_target)
                results = []
                for n_subj in range(n_target):
                    X_t = np.array(X_target[n_subj])
                    y_t = y_target[n_subj]
                    y_pred = clf.predict(X_t)
                    results.append(
                        {
                            "module": module_name,
                            "subject": n_subj,
                            "seed": seed,
                            "dataset_s": dataset_source,
                            "dataset_t": dataset_target,
                            "y_target": y_t,
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
