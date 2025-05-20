# %%

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed
from pathlib import Path
from sklearn.metrics import (
    balanced_accuracy_score,
)

import scikit_posthocs as sp


def get_name(x):
    if x["filter_size"] == 0:
        return x["norm"]
    else:
        return f"PSDNorm(F={x['filter_size']})"


def get_ranks(df):
    df = (
        df.groupby(["norm", "dataset", "seed", "subject"])
        .agg(bacc=("bacc", "mean"))
        .reset_index()
    )
    df = (
        df.groupby(["norm", "dataset", "subject"])
        .agg(bacc=("bacc", "mean"))
        .reset_index()
    )
    df["dataset_subject"] = df.apply(lambda x: f"{x['dataset']}_{x['subject']}", axis=1)
    df["rank"] = df.groupby("dataset_subject")["bacc"].rank(ascending=False)

    avg_rank = (
        df.groupby("dataset_subject", group_keys=True)  # marker
        .bacc.rank(pct=False, ascending=False)
        .groupby(df.norm)
        .mean()
    )
    test_results = sp.posthoc_conover_friedman(
        df,
        melted=True,
        block_col="dataset_subject",
        block_id_col="dataset_subject",
        group_col="norm",
        y_col="bacc",
    )
    return avg_rank, test_results


def compute_bacc(row):
    return balanced_accuracy_score(row.y_true, row.y_pred)


# %%
fnames = list(Path("balanced_CNN").glob("results*_400_*.pkl"))
df_1 = pd.concat([pd.read_pickle(fname) for fname in fnames], axis=0)

fnames = list(Path("balanced").glob("results*_400_*.pkl"))
df_2 = pd.concat([pd.read_pickle(fname) for fname in fnames], axis=0)
df = pd.concat([df_1, df_2], axis=0)
# %%
df['bacc'] = Parallel(n_jobs=-1)(
    delayed(compute_bacc)(row) for row in df.itertuples(index=False)
)

df["norm"] = df.apply(get_name, axis=1)


# %%
fig, axes = plt.subplots(
    2,
    1,
    figsize=(4.5, 2.8),
    gridspec_kw={
        "hspace": 0.8,
        "height_ratios": [0.8, 1],
    },
    sharex=True,
)
colors = sns.color_palette("tab10")

df_ = df.copy()
df_ = df_.query("norm != 'PSDNorm(F=1)' and norm != 'LayerNorm'")

ax = axes[0]
df_usleep = df_.query("model_name == 'USleep'")
avg_rank, test_results = get_ranks(df_usleep)
# change the order of the ranks

avg_rank = avg_rank.reindex(
    ["BatchNorm", "InstanceNorm", "PSDNorm(F=5)", "PSDNorm(F=9)", "PSDNorm(F=17)"]
)

sp.critical_difference_diagram(
    avg_rank,
    test_results,
    ax=ax,
    label_fmt_left="{label} ",
    label_fmt_right=" {label}",
    label_props={
        "fontsize": 14,
    },
    crossbar_props={"color": "black", "linewidth": 2},
    marker_props={"marker": ""},
    elbow_props={"linewidth": 2},
    color_palette=[colors[3], colors[2], colors[1], colors[4], colors[0]],
    text_h_margin=0.1,
)
# increase ticks fontsize
ax.tick_params(axis="both", which="major", labelsize=12)

ax.set_title("USleep", fontsize=14)

ax = axes[1]
df_cnn = df_.query("model_name == 'CNNTransformer'")
avg_rank, test_results = get_ranks(df_cnn)
sp.critical_difference_diagram(
    avg_rank,
    test_results,
    label_fmt_left="{label} ",
    label_fmt_right=" {label}",
    ax=ax,
    label_props={
        "fontsize": 14,
    },
    text_h_margin=0.1,
    crossbar_props={"color": "black", "linewidth": 2},
    marker_props={"marker": ""},
    elbow_props={"linewidth": 2},
    color_palette=[colors[3], colors[4], colors[2], colors[1], colors[0]],
)
ax.tick_params(axis="both", which="major", labelsize=12)
ax.set_title("CNNTransformer", fontsize=14)
plt.savefig("figures/cd_diagram.pdf", bbox_inches="tight")
