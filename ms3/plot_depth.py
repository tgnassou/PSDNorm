# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score
# add statisical test
from matplotlib.pyplot import text

# %%
fnames = list(Path("results/pickle_depth").glob("results*.pkl"))
df = pd.concat([pd.read_pickle(fname) for fname in fnames], axis=0)
df["f1"] = df.apply(lambda x: f1_score(x.y_true, x.y_pred, average="weighted"), axis=1)

# %%
df_base = df.query("tma == 'no_tma'")
# %%

fig, ax = plt.subplots(1, 3, figsize=(9, 3.1), sharey=True)
df_plot = df.query("dataset_type == 'target'")
df_plot = df_plot.groupby(["depth_tma", "subject", "dataset",]).f1.mean().reset_index()
datasets = ["SOF", "MASS", "CHAT"]
for i, dataset in enumerate(datasets):
    df_ = df_plot.query(f"dataset == '{dataset}'")
    df_base_ = df_base.query(f"dataset == '{dataset}'")
    df_ = df_.groupby(["depth_tma", "subject",]).f1.mean().reset_index()
    axis = sns.boxplot(
        x="depth_tma",
        y="f1",
        data=df_,
        ax=ax[i],
        # hue="depth",
        palette="tab10",
        # order=order,
        legend=False
    )

    for patch in axis.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .8))

    # pairs = [
    #     (("BatchNorm", 2849), ("CMLN", 424)),
    #     (("BatchNorm", 2849), ("CMLN", 2849)),
    # ]

    # annotator = Annotator(ax[i], pairs, data=df_, x="tma", y="f1", hue="depth")
    # annotator.configure(test="Wilcoxon", text_format='star', loc='inside')
    # annotator.apply_test(alternative='greater')
    # annotator.annotate()
    # ax[i].set_xticklabels(["BatchNorm", r"CMNL"])

    ax[i].set_xlabel("Depth")
    ax[i].set_ylabel("F1")
    ax[i].set_title("Dataset: " + dataset)
    # draw a black line at the base
    ax[i].axhline(df_base_.f1.median(), color="black", linestyle="--", label="BatchNorm")
    if i == 0:
        ax[i].legend()
    # remove legend
    # ax[i].get_legend().remove()
    # modify legend name

for ax_ in ax:
    ax_.grid(True)
    # for tick in ax_.get_xticklabels():
    #     tick.set_rotation(45)


plt.tight_layout()
fig.savefig("results_all/figures/boxplot.pdf", bbox_inches="tight")


# %%
sns.lineplot(
    x="depth_tma",
    y="f1",
    data=df_plot,
    hue="dataset",
    palette="tab10",
    # order=order,
    # legend=False if i != 0 else True,
)
# %%
