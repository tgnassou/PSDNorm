# %%
import numpy as np

from ms3.utils import load_dataset
from ms3.config import DATA_PATH, DATA_LOCAL_PATH

from pathlib import Path
import pandas as pd

from tqdm import tqdm

# %%
dataset_names = [
    "SHHS",
]

data_dict = {}
# %%
n_subject = -1
metadata = []
metadata_path = "metadata_shhs.parquet"
for dataset_name in dataset_names:
    print(f"Processing {dataset_name}")
    X, y, subject_ids, sessions = load_dataset(
        n_subjects=n_subject,
        dataset_name=dataset_name,
        data_path=DATA_PATH,
        scaler="sample",
    )
    # save data to npy
    if dataset_name == "Physionet":
        dataset_name = "PhysioNet"
    Path(f"{DATA_LOCAL_PATH}/{dataset_name}").mkdir(parents=True, exist_ok=True)
    Path(f"{DATA_LOCAL_PATH}/{dataset_name}/npy").mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(len(X))):
        X_ = X[i]
        y_ = y[i]
        subject_id = subject_ids[i]
        session = sessions[i]
        Path(f"{DATA_LOCAL_PATH}/{dataset_name}/npy/subject_{subject_id}").mkdir(
            parents=True, exist_ok=True
        )
        Path(f"{DATA_LOCAL_PATH}/{dataset_name}/npy/subject_{subject_id}/session_{session}").mkdir(
            parents=True, exist_ok=True
        )
        for sample in range(len(X_)):
            path = (
                f"{DATA_LOCAL_PATH}/{dataset_name}/npy/subject_{subject_id}/session_{session}/X_{sample}.npy"
            )
            np.save(path, X_[sample])
            metadata = [
                {
                    "dataset_name": dataset_name,
                    "subject_id": subject_id,
                    "session": session,
                    "y": y_[sample],
                    "sample": sample,
                    "path": path,
                }
            ]
            try:
                df_metadata = pd.read_parquet(metadata_path)
            except FileNotFoundError:
                df_metadata = pd.DataFrame()
            df_metadata = pd.concat((df_metadata, pd.DataFrame(metadata)))
            df_metadata.to_parquet(metadata_path)
