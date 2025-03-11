# %%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
import torch
# add statisical test
from statannotations.Annotator import Annotator
from statannotations.stats.StatTest import StatTest
from scipy.stats import wilcoxon
# import partial
from functools import partial
# import text
from matplotlib.pyplot import text

# %%
fnames = list(Path("results/pickle_boxplot").glob("results*.pkl"))
df = pd.concat([pd.read_pickle(fname) for fname in fnames], axis=0)
df["f1"] = df.apply(lambda x: f1_score(x.y_true, x.y_pred, average="weighted"), axis=1)

# %%
df_filtered = df.query("tma in ['no_tma', 'tma_geo_16', 'tma_bary_16']")

df_filtered["tma"] = df_filtered["tma"].replace("tma_geo_16", "CMNL")
df_filtered["tma"] = df_filtered["tma"].replace("tma_bary_16", "CMNL")
df_filtered["tma"] = df_filtered["tma"].replace("no_tma", "BatchNorm")

# %%

fig, ax = plt.subplots(1, 3, figsize=(9, 3.1), sharey=True)
df_plot = df_filtered.query("dataset_type == 'target'")
df_plot = df_plot.groupby(["tma", "subject", "n_subject_train", "dataset",]).f1.mean().reset_index()
datasets = ["SOF", "MASS", "CHAT"]
for i, dataset in enumerate(datasets):
    df_ = df_plot.query(f"dataset == '{dataset}'")
    df_ = df_.groupby(["tma", "subject", "n_subject_train"]).f1.mean().reset_index()
    axis = sns.boxplot(
        x="tma",
        y="f1",
        data=df_,
        ax=ax[i],
        hue="n_subject_train",
        palette="tab10",
        # order=order,
        legend=False if i != 0 else True,
    )

    for patch in axis.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .6))

    pairs = [
        (("BatchNorm", 2849), ("CMLN", 424)),
        (("BatchNorm", 2849), ("CMLN", 2849)),
    ]

    # annotator = Annotator(ax[i], pairs, data=df_, x="tma", y="f1", hue="n_subject_train")
    # annotator.configure(test="Wilcoxon", text_format='star', loc='inside')
    # annotator.apply_test(alternative='greater')
    # annotator.annotate()
    ax[i].set_xticklabels(["BatchNorm", r"CMNL"])

    ax[i].set_xlabel("")
    ax[i].set_ylabel("F1")
    ax[i].set_title("Dataset: " + dataset)
    # remove legend
    # ax[i].get_legend().remove()
    # modify legend name
handles, labels = ax[0].get_legend_handles_labels()
ax[0].legend(handles=handles, labels=["15%", "100%"], title="Nb. train subjects")

for ax_ in ax:
    ax_.grid(True)
    # for tick in ax_.get_xticklabels():
    #     tick.set_rotation(45)

plt.tight_layout()
fig.savefig("results_all/figures/boxplot.pdf", bbox_inches="tight")

# %%
# create diverging cmap  centered in 0
df_plot_scatter = df_plot.query("tma == 'BatchNorm'")[["subject", "dataset", "f1","n_subject_train"]].merge(
    df_plot.query("tma == 'CMNL'")[["subject", "dataset", "f1", "n_subject_train"]],
    on=["subject", "dataset","n_subject_train"],
    suffixes=("", "_adapted"),
)
df_plot_scatter["delta"] = df_plot_scatter.f1_adapted - df_plot_scatter.f1
fig, axes = plt.subplots(
    2, 3, figsize=(5.8, 3.1), sharex=True, sharey=True, layout="constrained"
)

# cmap = sns.dark_palette("#69d", reverse=True, as_cmap=True)
cmap = sns.diverging_palette(20, 230, center="light", sep=1, as_cmap=True)

fig.tight_layout(rect=[0, 0, .9, 1])
# vmax = df_plot_scatter['delta'].max()
# vmin = df_plot_scatter['delta'].min()
# vmin = -vmax
for i, n_subject_train in enumerate([2849, 424]):
    for j, dataset in enumerate(datasets):
        df_plot_ = df_plot_scatter.query(f"dataset == '{dataset}' & n_subject_train == {n_subject_train}")
        ax = axes[i, j]
        sns.scatterplot(
            data=df_plot_.query("delta > 0"),
            x="f1",
            y="f1_adapted",
            linewidth=0,
            marker=".",
            ax=ax,
            alpha=0.5,
            palette="colorblind",
            legend=False,
        )
        sns.scatterplot(
            data=df_plot_.query("delta < 0"),
            x="f1",
            y="f1_adapted",
            linewidth=0,
            marker=".",
            ax=ax,
            alpha=0.5,
            palette="colorblind",
            legend=False,
        )
        n = np.sum(df_plot_["delta"] > 0)
        text(
            0.11, 0.25, f"{int(np.round(n/len(df_plot_)*100))}%",
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes,
            size=8
        )
        text(
            0.23, 0.07, f"{int(np.round((1 - n/len(df_plot_))*100))}%", 
            horizontalalignment='center', 
            verticalalignment='center',
            transform=ax.transAxes, 
            size=8)

        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]

        # now plot both limits against eachother
        ax.plot(lims, lims, "k-", alpha=0.75, zorder=0)
        ax.set_aspect("equal")
        ax.set_xlim(lims)
        if i == 0:
            # make the tile a bit higher
            ax.set_title(f"Target: {dataset} \n({int(len(df_plot_))} subj.)", fontsize=11, pad=20)

        ax.set_xlabel("F1 BatchNorm", fontsize=11)
        if j == 0:
            ax.set_ylabel("F1 PSDNorm", fontsize=11)
        else:
            ax.set_ylabel("")
        ax.set_ylim(lims)
        # change ticks size
        ax.tick_params(axis='both', which='major', labelsize=10)
        # put same  ticks for x and y
        ax.set_xticks(np.arange(0.5, 0.91, 0.2))
        ax.set_yticks(np.arange(0.5, 0.91, 0.2))
axes[1, 1].set_title("15% of train subjects", fontsize=11,)

plt.suptitle("100% of train subjects", fontsize=11, y=1.001, x=0.46)
# plt.tight_layout()
fig.subplots_adjust(wspace=0.1,)

fig.savefig("results_all/figures/scatter.pdf", bbox_inches="tight")

# %%
