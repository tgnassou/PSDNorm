# %%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
# %%
history_loss = np.load("results/history_loss.npy")
history_acc = np.load("results/history_acc.npy")
history_val_loss = np.load("results/history_val_loss.npy")
history_val_acc = np.load("results/history_val_acc.npy")
history_acc_night_train = np.load("results/history_acc_night_train.npy")
history_acc_night_val = np.load("results/history_acc_night_val.npy")
history_acc_night_target = np.load("results/history_acc_night_target.npy")

# %%
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
axes[0].plot(history_loss, label="Loss")
axes[0].plot(history_val_loss, label="Loss Val")
axes[0].legend()
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")

axes[1].plot(history_acc, label="Acc")
axes[1].plot(history_val_acc, label="Acc Val")
axes[1].legend()
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")

axes[2].plot(history_acc_night_train, label="Acc Night Train")
axes[2].plot(history_acc_night_val, label="Acc Night Val")
axes[2].plot(history_acc_night_target, label="Acc Night Target")
axes[2].legend()
axes[2].set_title("Accuracy Night")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Accuracy")
plt.tight_layout()
# %%
fnames = list(Path("results/pickle").glob("results_LODO_sequence*d.pkl"))
df = pd.concat([pd.read_pickle(fname) for fname in fnames], axis=0)

df["acc"] = df.apply(lambda x: f1_score(x.y_target, x.y_pred, average="weighted"), axis=1)

# %%
# do the mean of the 3 scores for each dataset and scaling
df = df[["dataset_t", "module", "scaling", "acc", "subject"]]
df.loc[df.scaling == "subject", "scaling"] = "TMA"
df_mean = df.groupby(["dataset_t", "scaling", "module"]).mean().reset_index()
# df_lodo_mean = df_lodo.groupby(["dataset_t", "scaling"]).bal.mean().reset_index()

# change scaling subject to scaling TMA
# %%
df_tab = df_mean.pivot(index="dataset_t", columns=["module", "scaling"], values="acc")
# %%
# round the values
print(df_tab.to_latex(float_format="%.2f"))
# %%
import seaborn as sns

sns.boxplot(data=df, x="module", y="acc", hue="scaling")
# %%
