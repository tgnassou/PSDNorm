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
fnames = list(Path("results_all/pickle").glob("results*.pkl"))
df = pd.concat([pd.read_pickle(fname) for fname in fnames], axis=0)
df = df.query("percentage == 1")
# df["acc"] = df.apply(lambda x: accuracy_score(x.y_true, x.y_pred), axis=1)
df["f1"] = df.apply(lambda x: f1_score(x.y_true, x.y_pred, average="weighted"), axis=1)


# %%
# rename tma_bary as tma_32
# df["tma"] = df_concat["tma"].replace("tma_bary", "tma_bary_32")
# %%

fig, ax = plt.subplots(1, 3, figsize=(9, 3.6), sharey=True)
df_plot = df.query("dataset_type == 'target' & model != 'chambon'")
for i, dataset in enumerate(df_plot.dataset.unique()):
    df_ = df_plot.query(f"dataset == '{dataset}'")
    df_ = df_.groupby(["tma", "n_subject_train", "subject",]).f1.mean().reset_index()
    order = ["no_tma", "tma_bary", "tma_neurips"]
    sns.boxplot(
        x="tma",
        y="f1",
        data=df_,
        ax=ax[i],
        # hue="model",
        # order=order,
        # legend=False if i != 0 else True,
    )
    # test_results = add_stat_annotation(
    #     ax[i], data=df_, x="tma", y="f1", order=order,
    #     box_pairs=[("no_tma", "tma_bary"), ("no_tma", "tma_neurips")],
    #     test='Mann-Whitney', text_format='star',
    #     verbose=2
    # )

    # ax[i].set_xticklabels(["Baseline", r"CMLN$^{bary}$", r"CMLN$^{param}$", "CMMN"])

    ax[i].set_xlabel("")
    ax[i].set_ylabel("F1")
    ax[i].set_title("Dataset: " + dataset)


for ax_ in ax:
    ax_.grid(True)
    for tick in ax_.get_xticklabels():
        tick.set_rotation(45)
fig.suptitle("With more data")
plt.tight_layout()
fig.savefig("results_all/figures/norm_comparison.pdf", bbox_inches="tight")

# %%
df_plot.pivot_table(index="tma", columns="dataset", values="f1", aggfunc="mean", margins=True).round(3)
# %%
df_plot.pivot_table(index="tma", columns="dataset", values="f1", aggfunc="mean", margins=True).round(3)

# %%
df_target.pivot_table(index="tma", values="f1", aggfunc="mean")

