# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score

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
df_plot = df.query("norm not in ['PSDNorm(F=1)', 'LayerNorm']").copy()
df_plot = (
    df_plot.groupby(
        [
            "norm",
            "seed",
            "n_subjects",
            "dataset",
        ]
    )
    .agg(bacc=("bacc", "mean"))
    .reset_index()
)
df_plot = (
    df_plot.groupby(
        [
            "norm",
            "seed",
            "n_subjects",
        ]
    )
    .agg(bacc=("bacc", "mean"))
    .reset_index()
)

fig, (ax1, ax2) = plt.subplots(
    1, 2, sharey=True, figsize=(5, 2.5), gridspec_kw={"width_ratios": [3, 1]}
)
sns.lineplot(
    data=df_plot,
    x="n_subjects",
    y="bacc",
    hue="norm",
    palette="tab10",
    linewidth=2,
    err_style=None,
    alpha=0.8,
    ax=ax1,
)
ax1.set_xticks([40, 100, 200, 400])
ax1.set_xticklabels([r"$\sim$400", r"$\sim$1000", r"$\sim$2000", r"$\sim$4000"])
ax1.set_ylim(0.735, 0.795)
ax1.set_xlim(0, 405)
ax1.set_xlabel("Number of subjects")
ax1.grid(axis="y", alpha=0.6)
ax1.set_ylabel("Balanced Accuracy Score")
sns.lineplot(
    data=df_plot,
    x="n_subjects",
    y="bacc",
    hue="norm",
    palette="tab10",
    linewidth=2,
    err_style=None,
    alpha=0.8,
    ax=ax2,
    legend=False,
)
ax2.grid(axis="y", alpha=0.6)
ax2.set_xticks([5730])
ax2.set_xlim(4000, 6000)
ax2.set_xticklabels(["All subjects"])
ax2.set_ylim(0.735, 0.795)
ax2.set_xlabel("")
d = 0.015  # size of diagonal break marker
kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False)
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the right axes
ax2.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal

sns.despine()
plt.tight_layout()
fig.savefig("figures/LODO_lineplot.pdf", bbox_inches="tight")
