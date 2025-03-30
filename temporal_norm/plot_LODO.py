import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score
from statannotations.Annotator import Annotator
import re

# %% Load all result files
fnames = list(Path("results_LODO/pickles").glob("results_*_LODO_*.pkl"))

# Extract metadata from filenames
pattern = re.compile(r"results_(?P<model>[^_]+)_(?P<norm>[^_]+)_(?P<percent>[^_]+)_LODO_(?P<dataset>[^.]+).pkl")

data = []
for fname in fnames:
    match = pattern.match(fname.name)
    if match:
        meta = match.groupdict()
        df_part = pd.read_pickle(fname)
        df_part["model"] = meta["model"]
        df_part["norm"] = meta["norm"]
        df_part["percent"] = meta["percent"]
        df_part["dataset"] = meta["dataset"]
        data.append(df_part)

# Concatenate all
if not data:
    raise RuntimeError("No result files matched the expected pattern.")
df = pd.concat(data, axis=0)
df["f1"] = df.apply(lambda x: f1_score(x.y_true, x.y_pred, average="weighted"), axis=1)

# %% Plot + Tables per (model, percent)
for (model_name, percent), df_group in df.groupby(["model", "percent"]):
    # === BOXPLOT ===
    fig, ax = plt.subplots(figsize=(9, 3))
    sns.boxplot(
        data=df_group.query("dataset_type == 'target'"),
        x="dataset",
        y="f1",
        hue="norm",
        boxprops={"edgecolor": "none"},
        linewidth=0.8,
        flierprops=dict(marker=".", markersize=2),
        palette={"BatchNorm": "cornflowerblue", "PSDNorm": "lightcoral"},
        ax=ax
    )
    sns.despine()
    plt.grid(axis="y", alpha=0.6)

    # Annotations
    available_datasets = sorted(df_group["dataset"].unique())
    available_norms = df_group["norm"].unique()
    pairs = []
    if "BatchNorm" in available_norms and "PSDNorm" in available_norms:
        for d in available_datasets:
            sub_df = df_group.query("dataset == @d")
            if set(["BatchNorm", "PSDNorm"]).issubset(sub_df["norm"].unique()):
                pairs.append(((d, "BatchNorm"), (d, "PSDNorm")))

    if pairs:
        annotator = Annotator(ax, pairs, data=df_group, x="dataset", y="f1", hue="norm")
        annotator.configure(test="Wilcoxon", text_format="star", loc="inside", line_width=1)
        annotator.apply_and_annotate()

    plt.ylabel("F1 Score")
    plt.xlabel("")
    plt.xticks(rotation=45)
    plt.title(f"Model: {model_name}, Percent: {percent}")
    plt.tight_layout()

    fig_path = Path("figures") / f"LODO_F1_{model_name}_{percent}.pdf"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    # === DELTA BARPLOT ===
    df_bn = df_group[df_group.norm == "BatchNorm"]
    df_psd = df_group[df_group.norm == "PSDNorm"]
    df_merge = df_bn.merge(df_psd, on=["dataset", "subject"], suffixes=("_bn", "_psd"))
    df_merge["delta"] = df_merge["f1_psd"] - df_merge["f1_bn"]

    fig, ax = plt.subplots()
    sns.barplot(data=df_merge.query("delta >= 0"), x="dataset", y="delta", color="cornflowerblue", ax=ax)
    sns.barplot(data=df_merge.query("delta < 0"), x="dataset", y="delta", color="red", ax=ax)

    for i, dataset in enumerate(df_merge["dataset"].unique()):
        df_ = df_merge[df_merge.dataset == dataset]
        if not df_.empty:
            n_pos = (df_.delta >= 0).sum()
            n_neg = (df_.delta < 0).sum()
            n_tot = n_pos + n_neg
            if n_tot > 0:
                ax.text(i, 0.02, f"{n_pos/n_tot:.0%}", ha="center", va="bottom")
                ax.text(i, -0.02, f"{n_neg/n_tot:.0%}", ha="center", va="top")

    sns.stripplot(data=df_merge, x="dataset", y="delta", hue=df_merge["delta"] >= 0,
                  palette={True: "cornflowerblue", False: "red"},
                  size=3, alpha=0.1, linewidth=0, jitter=0.3, ax=ax, legend=False)

    plt.axhline(0, color="black", linestyle="--")
    plt.ylim(-0.15, 0.15)
    plt.xticks(rotation=45)
    plt.xlabel("")
    plt.ylabel(r"$\Delta$ F1 Score")
    plt.title(f"Delta F1: {model_name}, Percent: {percent}")
    sns.despine()
    plt.tight_layout()

    delta_path = Path("figures") / f"LODO_F1_delta_{model_name}_{percent}.pdf"
    delta_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(delta_path, bbox_inches="tight")
    plt.close(fig)

    # === LATEX TABLE ===
    df_tab = df_group.query("dataset_type == 'target'").copy()
    df_tab = df_tab.groupby(["dataset", "norm", "subject"]).f1.mean().reset_index()
    df_tab = df_tab.groupby(["dataset", "norm"])['f1'].agg(['mean', 'std']).reset_index()

    df_tab["formatted"] = df_tab.apply(
        lambda x: rf"{x['mean']:.2f} $\pm$ {x['std']:.2f}", axis=1
    )

    df_tab["bold"] = False
    for dataset in df_tab["dataset"].unique():
        df_sub = df_tab[df_tab["dataset"] == dataset]
        if len(df_sub) >= 2:
            best_idx = df_sub["mean"].idxmax()
            df_tab.loc[best_idx, "bold"] = True

    df_tab["mean_std"] = df_tab.apply(
        lambda x: rf"\textbf{{{x['formatted']}}}" if bool(x["bold"]) else x["formatted"], axis=1
    )

    df_tab["Score"] = "F1 Score"
    df_tab = df_tab.pivot_table(index="dataset", columns=["Score", "norm"], values="mean_std", aggfunc="first").fillna("")

    table_tex = df_tab.to_latex(escape=False, multicolumn_format="c", multirow=True)
    table_path = Path("tables") / f"LODO_table_{model_name}_{percent}.tex"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with open(table_path, "w") as f:
        f.write(table_tex)
