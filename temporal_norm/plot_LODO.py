import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score

sns.set_theme(style="whitegrid", context="talk")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

root = Path(".")

parts = []
for f in (root / "results_LODO" / "pickles").glob("results_*_LODO_*.pkl"):
    dfp = pd.read_pickle(f)
    parts.append(dfp)

if not parts:
    raise RuntimeError("No result files matched the expected pattern.")

df = pd.concat(parts, ignore_index=True)

df["f1"] = df.apply(lambda r: f1_score(r.y_true, r.y_pred, average="weighted"), axis=1)
df["norm_layer"] = df.apply(
    lambda r: "BatchNorm" if r["norm"] == "BatchNorm" else f"PSDNorm_{r['filter_size']}",
    axis=1,
)

# dataset label for titles
DATASET_LABEL = ", ".join(sorted(df["dataset"].unique()))

# ── ordering: BatchNorm first in legend, USleep first on x-axis ────
bn_first = ["BatchNorm"]
psd_rest = sorted([n for n in df["norm_layer"].unique() if n != "BatchNorm"])
HUE_ORDER = bn_first + psd_rest

all_models = sorted(df["model_name"].unique())
MODEL_ORDER = (
    ["USleep"] + [m for m in all_models if m != "USleep"]
    if "USleep" in all_models else all_models
)

# ─────────────────────────  PLOT 1  ────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(
    data=df,
    x="model_name",
    y="f1",
    hue="norm_layer",
    hue_order=HUE_ORDER,
    order=MODEL_ORDER,
    ax=ax,
    showmeans=True,
    flierprops=dict(marker=".", markersize=2),
    linewidth=0.8,
)
ax.set_xlabel("Model")
ax.set_ylabel("F1 score")
ax.set_title(f"F1 score by model and normalisation layer — {DATASET_LABEL}")
ax.grid(axis="y", alpha=0.4)
sns.despine()
ax.legend(title="Normalisation", bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout(rect=[0, 0, 0.85, 1])
fig.savefig(FIG_DIR / "LODO_F1_model_vs_norm.png", bbox_inches="tight")
plt.close(fig)

# ─────────────────────────  PLOT 2  ────────────────────────────────
#  ΔF1 = PSDNorm − BatchNorm
df_bn = df[df["norm_layer"] == "BatchNorm"]

deltas = []
for (model, percent), sub in df.groupby(["model_name", "percentage"]):
    for psd_var in psd_rest:                # only PSDNorm_* entries
        psd = sub[sub["norm_layer"] == psd_var]
        if psd.empty:
            continue
        merged = df_bn.merge(psd, on=["dataset", "subject"], suffixes=("_bn", "_psd"))
        merged["delta"]   = merged["f1_psd"] - merged["f1_bn"]
        merged["variant"] = psd_var
        merged["model_name"]   = model
        deltas.append(merged)

if deltas:
    ddf = pd.concat(deltas, ignore_index=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=ddf,
        x="model_name",
        y="delta",
        hue="variant",
        order=MODEL_ORDER,
        hue_order=psd_rest,
        ax=ax,
        showmeans=True,
        flierprops=dict(marker=".", markersize=2),
        linewidth=0.8,
    )
    ax.axhline(0, color="black", ls="--", lw=1)
    ax.set_ylim(-0.2, 0.2)
    ax.set_xlabel("Model")
    ax.set_ylabel(r"$\Delta$ F1 (PSDNorm − BatchNorm)")
    ax.set_title(f"PSDNorm gain over BatchNorm — {DATASET_LABEL}")
    ax.grid(axis="y", alpha=0.4)
    sns.despine()
    # legend intentionally removed
    ax.get_legend().remove()
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    fig.savefig(FIG_DIR / "LODO_F1_delta_model_vs_norm.png", bbox_inches="tight")
    plt.close(fig)
