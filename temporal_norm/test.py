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
    n_outputs=5,
    n_chans=2,
    n_windows=35,
    n
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

# %%
import torch
import torch.nn as nn

class MergeWindows(nn.Module):
    def __init__(self, n_windows):
        super().__init__()
        self.n_windows = n_windows

    def forward(self, x):
        # x: (n_batch * n_windows, n_chans, n_times)
        n_batch_times_n_windows, n_chans, n_times = x.shape
        n_batch = n_batch_times_n_windows // self.n_windows
        x = x.view(n_batch, self.n_windows, n_chans, n_times)
        x = x.permute(0, 2, 3, 1)  # (n_batch, n_chans, n_times, n_windows)
        x = x.reshape(n_batch, n_chans, n_times * self.n_windows)
        return x

class UnmergeWindows(nn.Module):
    def __init__(self, n_windows):
        super().__init__()
        self.n_windows = n_windows

    def forward(self, x):
        # x: (n_batch, n_chans, n_times * n_windows)
        n_batch, n_chans, total_time = x.shape
        n_times = total_time // self.n_windows
        x = x.view(n_batch, n_chans, n_times, self.n_windows)
        x = x.permute(0, 3, 1, 2)  # (n_batch, n_windows, n_chans, n_times)
        x = x.reshape(n_batch * self.n_windows, n_chans, n_times)
        return x
# %%
n_batch = 32
n_windows = 35
n_chans = 64
n_times = 3000

x_init = torch.randn(n_batch, n_windows, n_chans, n_times)

x = x_init.view(n_batch * n_windows, n_chans, -1)

merge = MergeWindows(n_windows)
unmerge = UnmergeWindows(n_windows)

merged = merge(x)
print(merged.shape)  # Should be (n_batch, n_chans, n_times * n_windows)

restored = unmerge(merged)
print(restored.shape) 
# %%
restored
# %%
x_final = restored.view(n_batch, n_windows, n_chans, n_times)
# %%
x_final == x_init

# %%
