# %%

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

# add statisical test

from statannotations.Annotator import Annotator

# %%
fnames = list(Path("results_LODO/pickles").glob("results*0.15*.pkl"))
df = pd.concat([pd.read_pickle(fname) for fname in fnames], axis=0)
# df["y_pred"] = df.apply(lambda x: np.concatenate(x.y_pred)[:, 10:25].flatten() if x.dataset_type == "target" else x.y_pred, axis=1)
# check if subj is available for both PSDNorm and BatchNorm otherwise discard it
df["f1"] = df.apply(lambda x: f1_score(x.y_true, x.y_pred, average="weighted"), axis=1)

# %%
df_psd = df.query("norm == 'PSDNorm'").copy()
df_bn = df.query("norm == 'BatchNorm'").copy()

# for each dataset check is the subject is available for both norm
df_psd = df_psd.groupby(["dataset", "subject"]).size().reset_index()
df_bn = df_bn.groupby(["dataset", "subject"]).size().reset_index()

df_filter = df_psd.merge(df_bn, on=["dataset", "subject"])

# keep only dataset and subject in df_filter
df_filter = df_filter[["dataset", "subject"]]
df = df.merge(df_filter, on=["dataset", "subject"])



# %% BOXPLOT
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
fig, ax = plt.subplots(figsize=(9, 3))
axis = sns.boxplot(
    data=df.query("dataset_type == 'target'"),
    x="dataset",
    y="f1",
    hue="norm",
    boxprops={"edgecolor": "none"},
    linewidth=0.8,
    flierprops=dict(marker=".", markersize=2),
    palette={"BatchNorm": "cornflowerblue", "PSDNorm": "lightcoral"}
)
# for patch in axis.patches:
#     r, g, b, a = patch.get_facecolor()
#     patch.set_facecolor((r, g, b, 0.9))

sns.despine()

# horizontal grid
plt.grid(axis="y", alpha=0.6)

pairs = []
for dataset in dataset_names:
    pairs.append(((dataset, "BatchNorm"), (dataset, "PSDNorm")))
annotator = Annotator(ax, pairs, data=df, x="dataset", y="f1", hue="norm")
annotator.configure(test="Wilcoxon", text_format="star", loc="inside", line_width=1)
annotator.apply_and_annotate()
plt.ylabel("F1 Score")
plt.xlabel("")
plt.show()
# rotate axis label

fig.savefig("figures/LODO_F1.pdf", bbox_inches="tight")
# %% Boxplot of the delta between the two norms

fig, ax = plt.subplots()
df_bn = df.query("norm == 'BatchNorm'").copy()
df_psd = df.query("norm == 'PSDNorm'").copy()
df_merge = df_bn.merge(
    df_psd,
    on=[
        "dataset",
        "subject",
    ],
)
df_merge["delta"] = df_merge.f1_y - df_merge.f1_x
# draw line at 0

axis = sns.barplot(data=df_merge.query("delta  >= 0"), x="dataset", y="delta")
for patch in axis.patches:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, 0.6))

axis = sns.barplot(
    data=df_merge.query("delta  < 0"), x="dataset", y="delta", color="red"
)
for patch in axis.patches:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, 0.6))

# add % above and below 0
for i, dataset in enumerate(dataset_names):
    df_ = df_merge.query(f"dataset == '{dataset}'")
    n_pos = (df_.delta >= 0).sum()
    n_neg = (df_.delta < 0).sum()
    n_tot = n_pos + n_neg
    ax.text(
        i,
        0.02,
        f"{n_pos/n_tot:.0%}",
        ha="center",
        va="bottom",
        color="black",
    )
    ax.text(
        i,
        -0.02,
        f"{n_neg/n_tot:.0%}",
        ha="center",
        va="top",
        color="black",
    )
sns.stripplot(
    data=df_merge.query("delta  >= 0"),
    x="dataset",
    y="delta",
    color="cornflowerblue",
    ax=ax,
    size=3,
    alpha=0.1,
    linewidth=0,
    jitter=0.3,
)


sns.stripplot(
    data=df_merge.query("delta  < 0"),
    x="dataset",
    y="delta",
    color="red",
    ax=ax,
    size=3,
    alpha=0.05,
    # linewidth=0,
    jitter=0.3,
)

sns.despine()


plt.axhline(0, color="black", linestyle="--")
plt.ylim(-0.15, 0.15)
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("$\Delta$ F1 Score")
plt.show()


# %% TABLE
df_tab = df.query("dataset_type == 'target'").copy()

df_tab = (
    df_tab.groupby(
        [
            "dataset",
            "norm",
            "subject",
        ]
    )
    .f1.mean()
    .reset_index()
)
df_tab = df_tab.groupby(["dataset", "norm"]).agg({"f1": ["mean", "std"]})

# df_mean = df_tab.groupby(["tma"]).mean().reset_index()
# df_mean["dataset"] = df_mean["dataset_type"]
# df_tot = pd.concat([df_tab, df_mean], axis=0)
df_tab = df_tab.reset_index()
df_tab["mean_std"] = df_tab.apply(
    lambda x: f"{x.f1['mean']:.2f} $\pm$ {x.f1['std']:.2f}", axis=1  # noqa
)

# %%
# bold the best result per dataset
idx_to_bold = df_tab.groupby(["dataset"]).f1.idxmax().f1["mean"].to_list()
for idx in idx_to_bold:
    value = df_tab.loc[idx, "mean_std"].values[0]
    df_tab.loc[idx, "mean_std"] = f"\\textbf{{{value}}}"
# %%
df_tab["Score"] = "F1 Score"
df_tab = df_tab.pivot_table(
    index=["dataset"], columns=["Score", "norm"], values="mean_std", aggfunc="first"
)
# df_tab = df_tab.iloc[:, [1, 0, 3, 2]]

# %%
df_tab_20 = df.copy()

df_tab_20 = df_tab_20.groupby(["dataset", "norm", "subject"]).f1.mean().reset_index()
df_tab_20_base = df_tab_20.query("norm == 'BatchNorm'").reset_index()
df_tab_20_subject = (
    df_tab_20_base.groupby(["dataset"])
    .f1.apply(lambda x: x.nsmallest(int(0.2 * len(x))))
    .reset_index()
)
# df_tab_20_base = df_tab_20_base.iloc[df_tab_20_subject..to_list()]

df_tab_20 = df_tab_20.merge(df_tab_20_base, on=["dataset", "subject"])
df_tab_20 = df_tab_20[
    [
        "dataset",
        "subject",
        "norm_x",
        "f1_x",
    ]
]
df_tab_20.columns = ["dataset", "subject", "norm", "f1"]
df_tab_20 = df_tab_20.groupby(["dataset", "norm"]).agg({"f1": ["mean", "std"]})

# df_mean_20 = df_tab_20.groupby(["tma"]).mean().reset_index()
# df_mean_20["dataset"] = df_mean_20["dataset_type"]
# df_tab_20 = df_tab_20.reset_index()
# df_tot_20 = pd.concat([df_tab_20, df_mean_20], axis=0)
df_tab_20 = df_tab_20.reset_index()
df_tab_20.fillna(0, inplace=True)
df_tab_20["mean_std"] = df_tab_20.apply(
    lambda x: f"{x.f1['mean']:.2f} $\pm$ {x.f1['std']:.2f}", axis=1  # noqa
)

# %%
idx_to_bold = df_tab_20.groupby(["dataset"]).f1.idxmax().f1["mean"].to_list()
for idx in idx_to_bold:
    value = df_tab_20.loc[idx, "mean_std"].values[0]
    df_tab_20.loc[idx, "mean_std"] = f"\\textbf{{{value}}}"

# %%
df_tab_20["Score"] = "$\Delta$ F1@20\% Score"
df_tab_20 = df_tab_20.pivot_table(
    index=["dataset"], columns=["Score", "norm"], values="mean_std", aggfunc="first"
)
# df_tab_20 = df_tab_20.iloc[:, [1, 0, 3, 2]]
# %%
# concatene and add index to the table
df_final = pd.concat([df_tab, df_tab_20], axis=1)

# %%
lat_tab = df_final.to_latex(
    escape=False,
    multicolumn_format="c",
    multirow=True,
    column_format="|l|l|cccc|cccc|",
)

# %%
lat_tab = lat_tab.replace("InstantNorm", "InstanceNorm")
lat_tab = lat_tab.replace("toprule", "hline")
lat_tab = lat_tab.replace("midrule", "hline")
lat_tab = lat_tab.replace(
    "multirow[t]{7}{*}{source}",
    "multirow{7}{*}{\\rotatebox[origin=c]{90}{Internal-Test}}",
)
lat_tab = lat_tab.replace(
    "multirow[t]{3}{*}{target}", "multirow{3}{*}{\\rotatebox[origin=c]{90}{External}}"
)
lat_tab = lat_tab.replace("source", "Mean")
lat_tab = lat_tab.replace("target", "Mean")
# %%
print(lat_tab)
# %%
