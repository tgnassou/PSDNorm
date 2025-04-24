# %%
import pandas as pd
import torch
from temporal_norm.utils.architecture import CareSleepNet
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
model = CareSleepNet(
    n_classes=5,
    n_channels=2,
    n_sequences=35,
).to(device="cpu")
# %%

X = torch.randn(10, 35, 2, 3000).to(device="cpu")

# %%
output = model(X)

print(output.shape)
# %%
x_eeg = X[:, :, :1, :]

# %%
x_eeg.view(10 * 35, 1, -1).shape
