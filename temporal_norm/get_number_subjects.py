# %%

import numpy as np
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
metadata = pd.read_csv("metadata/metadata_sleep.csv").drop(columns=["Unnamed: 0"])

# %%
# convert session to str
metadata["session"] = metadata["session"].astype(str)
# %%
# save in parquet
metadata.to_parquet("metadata/metadata_sleep.parquet")
# %%
metadata["sub+session"] = metadata.apply(lambda x: f"{x['subject_id']}_{x['session']}", axis=1)
# %%
for dataset_name in dataset_names:
    print(dataset_name)
    print("n subs:", metadata[metadata["dataset_name"] == dataset_name]["subject_id"].nunique())
    print("n rec:", metadata[metadata["dataset_name"] == dataset_name]["sub+session"].nunique())
