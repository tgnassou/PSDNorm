# %%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

# %%
fnames = list(Path("results_all/pickle").glob("results*.pkl"))
df = pd.concat([pd.read_pickle(fname) for fname in fnames], axis=0)
# merge y_true and y_target when nan
# replace nan in column tma by "none"
# %%
df["f1"] = df.apply(lambda x: f1_score(x.y_true, x.y_pred, average="weighted"), axis=1)
df["acc"] = df.apply(lambda x: accuracy_score(x.y_true, x.y_pred), axis=1)
# %%

# replace None to "none"
# %%
import seaborn as sns
import matplotlib.pyplot as plt
df_plot = df.query("percentage != 0.1")
fig, ax = plt.subplots(1, 2, figsize=(7, 4), sharey=True)
df_source = df_plot.query("dataset_type == 'source'")
sns.boxplot(x="tma", y="f1", data=df_source, palette="tab10", ax=ax[0])

# change xticklabels
# ax[0].set_xticklabels(["Baseline", "TMA bary", "TMA tuning", "TMA preprocess"])

# for tick in ax.get_xticklabels():
#     tick.set_rotation(45)


ax[0].set_xlabel("")
ax[0].set_ylabel("F1")
ax[0].set_title("Source dataset (Test)")

df_target = df_plot.query("dataset_type == 'target'")
sns.boxplot(x="tma", y="f1", data=df_target, palette="tab10", ax=ax[1])
ax[1].set_xticklabels(["Baseline", "TMA bary", "TMA tuning", "TMA preprocess"])
ax[1].set_xlabel("")
ax[1].set_ylabel("F1")
ax[1].set_title("Target dataset")

for ax_ in ax:
    ax_.grid(True)
    for tick in ax_.get_xticklabels():
        tick.set_rotation(45)

plt.tight_layout()

# %%
# table
# df_target_ = df_target.groupby(["tma", "dataset"],).f1.mean().reset_index()

df_target.pivot_table(index=["tma", "dataset"], values="f1", aggfunc="mean")

# %%
import torch

# %%
model = torch.load("results_all/models/models_tma_bary_0.2.pt")
barycenter = model.encoder[0].block_prepool[2].barycenter
model_2 = torch.load("results_all/models/models_tma_learn_0.2.pt")
barycenter_learn = model_2.encoder[0].block_prepool[2].barycenter
model_3 = torch.load("results_all/models/models_tma_learn_log_exp_0.2.pt")
barycenter_log_exp = model_3.encoder[0].block_prepool[2].barycenter
model_4 = torch.load("results_all/models/models_tma_learn_ReLu_0.2.pt")
barycenter_Relu = model_4.encoder[0].block_prepool[2].barycenter

# %%
barycenter_learn.shape
# %%
plt.plot(barycenter.T.cpu().detach().numpy(), alpha=0.5)
# log
plt.yscale("log")
plt.title("Barycenter for first TMA with 6 channels")
# %%
plt.title("PSD learned for first TMA with 6 channels")
plt.plot(torch.exp(barycenter_learn).T.cpu().detach().numpy())
# plt.yscale("log")

# %%
plt.title("PSD learned for first TMA with log exp")
plt.plot(torch.log(1 + torch.exp(barycenter_log_exp)).cpu().detach().numpy())
plt.yscale("log")

# %%
plt.title("Bary learned for first TMA with ReLu")
plt.plot(torch.relu(barycenter_Relu).cpu().detach().numpy())
plt.yscale("log")
# %%
