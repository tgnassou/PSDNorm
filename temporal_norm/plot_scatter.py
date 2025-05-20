# %%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score

from matplotlib.pyplot import text
from joblib import Parallel, delayed


def get_name(x):
    if x["filter_size"] == 0:
        return x["norm"]
    else:
        return f"PSDNorm(F={x['filter_size']})"


def compute_bacc(row):
    return balanced_accuracy_score(row.y_true, row.y_pred)


# %%
fnames = list(Path("balanced").glob("results*.pkl"))
df = pd.concat([pd.read_pickle(fname) for fname in fnames], axis=0)
df = df[
    [
        "dataset",
        "seed",
        "subject",
        "n_subjects",
        "y_true",
        "y_pred",
        "filter_size",
        "norm",
        "model_name",
    ]
]

# %%

df["norm"] = df.apply(get_name, axis=1)

df["bacc"] = Parallel(n_jobs=-1)(
    delayed(compute_bacc)(row) for row in df.itertuples(index=False)
)
# %%

df_plot = df.query("n_subjects == 400").copy()
df_plot = (
    df_plot.groupby(["dataset", "norm", "subject", "n_subjects"])
    .agg(bacc=("bacc", "mean"))
    .reset_index()
)
df_plot_scatter_BN = df_plot.query("norm == 'BatchNorm'")[
    ["subject", "dataset", "bacc", "n_subjects"]
].merge(
    df_plot.query("norm == 'PSDNorm(F=5)'")[
        ["subject", "dataset", "bacc", "n_subjects"]
    ],
    on=["subject", "dataset", "n_subjects"],
    suffixes=("", "_adapted"),
)
df_plot_scatter_BN["delta"] = df_plot_scatter_BN.bacc_adapted - df_plot_scatter_BN.bacc

df_plot_scatter_IN = df_plot.query("norm == 'InstanceNorm'")[
    ["subject", "dataset", "bacc", "n_subjects"]
].merge(
    df_plot.query("norm == 'PSDNorm(F=5)'")[
        ["subject", "dataset", "bacc", "n_subjects"]
    ],
    on=["subject", "dataset", "n_subjects"],
    suffixes=("", "_adapted"),
)
df_plot_scatter_IN["delta"] = df_plot_scatter_IN.bacc_adapted - df_plot_scatter_IN.bacc

fig, axes = plt.subplots(
    2,
    2,
    figsize=(4.3, 3.3),
    sharex=True,
    sharey=True,
    layout="constrained",
)

fig.tight_layout(rect=[0, 0, 0.9, 1])
palette = sns.color_palette("colorblind")
datasets = ["MASS", "CHAT"]
for i in range(2):
    if i == 0:
        df_plot_scatter = df_plot_scatter_BN
        norm = "BatchNorm"
    else:
        df_plot_scatter = df_plot_scatter_IN
        norm = "InstanceNorm"
    for j, dataset in enumerate(datasets):
        df_plot_ = df_plot_scatter.query(f"dataset == '{dataset}' & n_subjects == 400")
        ax = axes[i, j]
        sns.scatterplot(
            data=df_plot_.query("delta > 0"),
            x="bacc_adapted",
            y="bacc",
            linewidth=0,
            marker=".",
            ax=ax,
            alpha=0.5,
            palette="colorblind",
            legend=False,
        )
        sns.scatterplot(
            data=df_plot_.query("delta < 0"),
            x="bacc_adapted",
            y="bacc",
            linewidth=0,
            marker=".",
            ax=ax,
            alpha=0.5,
            palette="colorblind",
            legend=False,
        )
        n = np.sum(df_plot_["delta"] > 0)
        text(
            0.94,
            0.10,
            f"{int(np.round(n/len(df_plot_)*100))}%",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            color=palette[0],
            size=10,
        )
        text(
            0.17,
            0.9,
            f"{int(np.round((1 - n/len(df_plot_))*100))}%",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            color=palette[1],
            size=10,
        )

        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]

        # now plot both limits against eachother
        ax.plot(lims, lims, "k-", alpha=0.75, zorder=0)
        ax.set_xlim(lims)
        ax.set_aspect("equal")
        if i == 0:
            ax.set_title(
                f"Target: {dataset} \n({int(len(df_plot_))} subj.)", fontsize=12, pad=10
            )
        ax.set_xlabel("BACC PSDNorm", fontsize=12)
        if i == 0:
            ax.set_ylabel("BACC \n BatchNorm", fontsize=12)
        else:
            ax.set_ylabel("BACC \n InstanceNorm", fontsize=12)
        ax.set_ylim(lims)
        # change ticks size
        ax.tick_params(axis="both", which="major", labelsize=11)
        # put same  ticks for x and y
        ax.set_xticks(np.arange(0.5, 0.91, 0.2))
        ax.set_yticks(np.arange(0.5, 0.91, 0.2))
# axes[1, 1].set_title(r"$\sim$ 4 000 subjects", fontsize=11,)

sns.despine()
fig.subplots_adjust(
    wspace=0.05,
)

fig.savefig("figures/scatter.pdf", bbox_inches="tight")
