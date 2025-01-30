# %%

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
import torch
# add statisical test
from statannot import add_stat_annotation

# %%
fnames = list(Path("results/pickle_table").glob("results*.pkl"))
df = pd.concat([pd.read_pickle(fname) for fname in fnames], axis=0)
df["f1"] = df.apply(lambda x: f1_score(x.y_true, x.y_pred, average="weighted"), axis=1)

# %%
df_tab = df.copy()
df_tab = df_tab.groupby(["dataset", "dataset_type", "tma", "subject",]).f1.mean().reset_index()
df_tab = df_tab.groupby(['dataset', 'dataset_type', 'tma']).agg({"f1": ["mean", "std"]})

# df_mean = df_tab.groupby(["dataset_type", "tma"]).mean().reset_index()
# df_mean["dataset"] = df_mean["dataset_type"]
# df_tot = pd.concat([df_tab, df_mean], axis=0)
df_tab = df_tab.reset_index()
df_tab["mean_std"] = df_tab.apply(
    lambda x: f"{x.f1['mean']:.2f} $\pm$ {x.f1['std']:.2f}", axis=1 # noqa
)

# %%
# bold the best result per dataset
idx_to_bold = df_tab.groupby(["dataset_type", "dataset"]).f1.idxmax().f1["mean"].to_list()
for idx in idx_to_bold:
    value = df_tab.loc[idx, "mean_std"].values[0]
    df_tab.loc[idx, "mean_std"] = f"\\textbf{{{value}}}"
# %%
df_tab["Score"] = "F1 Score"
df_tab = df_tab.pivot_table(index=["dataset_type", "dataset"], columns=["Score", "tma"], values="mean_std", aggfunc="first")
df_tab = df_tab.iloc[:, [1, 0, 3, 2]]

# %%
df_tab_20 = df.copy()

df_tab_20 = df_tab_20.groupby(["dataset", "dataset_type", "tma", "subject"]).f1.mean().reset_index()
df_tab_20_base = df_tab_20.query("tma == 'no_tma'").reset_index()
df_tab_20_subject = df_tab_20_base.groupby(["dataset_type", "dataset"]).f1.apply(lambda x: x.nsmallest(int(0.2 * len(x)))).reset_index()
df_tab_20_base = df_tab_20_base.iloc[df_tab_20_subject.level_2.to_list()]

df_tab_20 = df_tab_20.merge(df_tab_20_base, on=["dataset_type", "dataset", "subject"])
df_tab_20 = df_tab_20[["dataset_type", "dataset", "subject", "tma_x", "f1_x",]]
df_tab_20.columns = ["dataset_type", "dataset", "subject", "tma", "f1"]
df_tab_20 = df_tab_20.groupby(['dataset', 'dataset_type', 'tma']).agg({"f1": ["mean", "std"]})

# df_mean_20 = df_tab_20.groupby(["dataset_type", "tma"]).mean().reset_index()
# df_mean_20["dataset"] = df_mean_20["dataset_type"]
# df_tab_20 = df_tab_20.reset_index()
# df_tot_20 = pd.concat([df_tab_20, df_mean_20], axis=0)
df_tab_20 = df_tab_20.reset_index()
df_tab_20.fillna(0, inplace=True)
df_tab_20["mean_std"] = df_tab_20.apply(
    lambda x: f"{x.f1['mean']:.2f} $\pm$ {x.f1['std']:.2f}", axis=1 # noqa
)

# %%
idx_to_bold = df_tab_20.groupby(["dataset_type", "dataset"]).f1.idxmax().f1["mean"].to_list()
for idx in idx_to_bold:
    value = df_tab_20.loc[idx, "mean_std"].values[0]
    df_tab_20.loc[idx, "mean_std"] = f"\\textbf{{{value}}}"

# %%
df_tab_20["Score"] = "$\Delta$ F1@20\% Score"
df_tab_20 = df_tab_20.pivot_table(index=["dataset_type", "dataset"], columns=["Score", "tma"], values="mean_std", aggfunc="first")
df_tab_20 = df_tab_20.iloc[:, [1, 0, 3, 2]]
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
lat_tab = lat_tab.replace("no_tma", "BatchNorm")
lat_tab = lat_tab.replace("tma_bary_16", "CMLN")
lat_tab = lat_tab.replace("InstantNorm", "InstanceNorm")
lat_tab = lat_tab.replace("toprule", "hline")
lat_tab = lat_tab.replace("midrule", "hline")
lat_tab = lat_tab.replace("multirow[t]{7}{*}{source}", "multirow{7}{*}{\\rotatebox[origin=c]{90}{Internal-Test}}")
lat_tab = lat_tab.replace("multirow[t]{3}{*}{target}", "multirow{3}{*}{\\rotatebox[origin=c]{90}{External}}")
lat_tab = lat_tab.replace("source", "Mean")
lat_tab = lat_tab.replace("target", "Mean")
# %%
print(lat_tab)
# %%
