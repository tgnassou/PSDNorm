# %%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score

# %%
fnames = list(Path("results/pickle_percentage").glob("results_*.pkl"))
df = pd.concat([pd.read_pickle(fname) for fname in fnames], axis=0)
df["f1"] = df.apply(lambda x: f1_score(x.y_true, x.y_pred, average="weighted"), axis=1)

# %%
df["tma"] = df["tma"].replace("no_tma", "BatchNorm")
df["tma"] = df["tma"].replace("tma_bary", "CMLN")
df["tma"] = df["tma"].replace("tma_bary_64", "CMLN")
df["tma"] = df["tma"].replace("tma_bary_16", "CMLN")
df["tma"] = df["tma"].replace("tma_geo_16", "CMLN")
df["tma"] = df["tma"].replace("tma_geo_32", "CMLN")
df["tma"] = df["tma"].replace("tma_geo_64", "CMLN")

# %%
df["f_size"] = df["filter_size"].apply(lambda x: 0 if x is None else x)

# %%
df_test = df.reset_index().query("dataset_type == 'target'")
# %%
# normalize f1 score by the maximum f1 score per dataset
# for dataset in ["CHAT", "SOF", "MASS"]:
#     df_dataset = df_test.query(f"dataset == '{dataset}'")
#     max_f1 = df_dataset.f1.max()
#     min_f1 = df_dataset.f1.min()
#     df_test.loc[df_dataset.index, "f1_normalized"] = (df_dataset.f1 - min_f1) / (max_f1 - min_f1)
# %%
fig, ax = plt.subplots(1, 1, figsize=(4.5, 3))
df_plot = df.groupby(["dataset_type", "tma", "percentage", "f_size",]).f1.mean().reset_index()
df_target = df_plot.query("dataset_type == 'target'")
sns.lineplot(
    data=df_target.query("tma == 'BatchNorm'"),
    x="percentage",
    y="f1",
    ax=ax,
    palette=sns.color_palette("colorblind")[1:],
    hue="f_size",
    linewidth=3,
    linestyle="--",
    alpha=0.7,
)
handles_batch, _ = ax.get_legend_handles_labels()
# reorder the lines
palette = [sns.color_palette("Blues_d", n_colors=5)[4], sns.color_palette("Blues_d", n_colors=5)[2], sns.color_palette("Blues_d", n_colors=5)[0]]
sns.lineplot(
    data=df_target.query("tma != 'BatchNorm'"),
    x="percentage",
    y="f1",
    hue="f_size",
    palette=palette,
    ax=ax,
    linewidth=3,
    alpha=0.7,
)
plt.tight_layout()
ax.set_xlabel("Percentage of training subjects")
ax.set_ylabel("F1")
handles, labels = ax.get_legend_handles_labels()
new_labels = ["BatchNorm", "PSDNorm(16)", "PSDNorm(32)", "PSDNorm(64)"]
ax.legend(handles=handles, labels=new_labels,)
ax.grid(True)
sns.despine()
ax.set_yticks(np.arange(0.65, 0.85, 0.05))
ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%",])
fig.savefig("results_percentage/figures/number_subjects.pdf", bbox_inches="tight")
# %%
df_target.pivot_table(index="tma", columns=["n_subject_train", "dataset"], values="f1", aggfunc="mean")

# %%
