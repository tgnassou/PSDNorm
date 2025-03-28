# %%
import pandas as pd
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
for dataset_name in dataset_names:
    print(dataset_name)
    print("n subs:", metadata[metadata["dataset_name"] == dataset_name]["subject_id"].nunique())
    print("n rec:", metadata[metadata["dataset_name"] == dataset_name]["sub+session"].nunique())

# %%
