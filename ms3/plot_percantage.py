# %%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
# %%
fnames = list(Path("results_percentage/pickle").glob("results_*.pkl"))
df = pd.concat([pd.read_pickle(fname) for fname in fnames], axis=0)

df["f1"] = df.apply(lambda x: f1_score(x.y_true, x.y_pred, average="weighted"), axis=1)
df["acc"] = df.apply(lambda x: accuracy_score(x.y_true, x.y_pred), axis=1)

# %%
fig, ax = plt.subplots(1, 2,, sharey=True)
df_source = df.query("dataset_type == 'source'")
sns.lineplot(x="percentage", y="f1", hue="dataset", data=df_source,)
plt.tight_layout()
ax.set_title("Effect of number of subjects")
ax.set_xlabel("Percentage")
ax.set_ylabel("F1")

df_target = df.query("dataset_type == 'target'")
sns.lineplot(x="percentage", y="f1", hue="dataset", data=df_target,)
plt.tight_layout()
ax.set_title("Effect of number of subjects")
ax.set_xlabel("Percentage")
ax.set_ylabel("F1")

# %%
df_mean = df.groupby(["percentage", "dataset_target"]).acc.mean().reset_index()
fig, ax = plt.subplots(1, 1,)
sns.lineplot(x="percentage", y="acc", data=df, )
plt.tight_layout()
ax.set_title("Effect of number of subjects")
ax.set_xlabel("Percentage")
ax.set_ylabel("Accuracy")
# %%
