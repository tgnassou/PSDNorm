# %%
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
df_tab = df.query("n_subjects in [5730, 400]").copy()

df_tab = (
    df_tab.groupby(["dataset", "norm", "seed", "n_subjects"]).bacc.mean().reset_index()
)
mean_tab = df_tab.groupby(["norm", "n_subjects", "seed"]).bacc.mean().reset_index()
df_tab = df_tab.groupby(
    [
        "dataset",
        "n_subjects",
        "norm",
    ]
).agg({"bacc": ["mean", "std"]})

# add mean tab as a dataset
mean_tab = mean_tab.groupby(["norm", "n_subjects"]).agg({"bacc": ["mean", "std"]})
mean_tab = mean_tab.reset_index()
mean_tab["dataset"] = "Mean"
mean_tab["mean_std"] = mean_tab.apply(
    lambda x: f"{x.bacc['mean']*100:.2f} $\pm$ {x.bacc['std']*100:.2f}", axis=1  # noqa
)
df_tab = df_tab.reset_index()
df_tab["mean_std"] = df_tab.apply(
    lambda x: f"{x.bacc['mean']*100:.2f} $\pm$ {x.bacc['std']*100:.2f}",
    axis=1,  # noqa
)

df_tab = pd.concat([df_tab, mean_tab], axis=0, ignore_index=True)

idx_to_bold = df_tab.groupby(
    ["n_subjects", "dataset"]
).bacc.idxmax().bacc["mean"].to_list()
for idx in idx_to_bold:
    value = df_tab.loc[idx, "mean_std"].values[0]
    df_tab.loc[idx, "mean_std"] = f"\\textbf{{{value}}}"

df_tab["Score"] = "bacc Score"
df_tab = df_tab.pivot_table(
    index=["n_subjects", "dataset"],
    columns=["norm"],
    values="mean_std",
    aggfunc="first",
)

new_order = [
    "BatchNorm",
    "LayerNorm",
    "InstanceNorm",
    "PSDNorm(F=5)",
    "PSDNorm(F=9)",
    "PSDNorm(F=17)",
]

df_tab = df_tab.loc[:, pd.IndexSlice[:, new_order]]


lat_tab = df_tab.to_latex(
    escape=False,
    multicolumn_format="c",
    multirow=True,
)

print(lat_tab)
