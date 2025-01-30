# %%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
# %%

history = pd.read_pickle("results_all/history/history_no_tma_1.pkl")
history_loss = history["train_loss"]
history_val_loss = history["val_loss"]

# history_acc_night_train = history["train_acc_night"]
# history_acc_night_std_train = history["train_acc_night_std"]
# history_acc_night_val = history["val_acc_night"]
# history_acc_night_std_val = history["val_acc_night_std"]
# history_acc_night_target = history["target_acc"]
# history_acc_night_std_target = history["target_acc_std"]

# history_f1_night_train = history["train_f1_night"]
# history_f1_night_std_train = history["train_f1_night_std"]
# history_f1_night_val = history["val_f1_night"]
# history_f1_night_std_val = history["val_f1_night_std"]
# history_f1_night_target = history["target_f1"]
# history_f1_night_std_target = history["target_f1_std"]

history_acc_train = history["train_acc"]
history_acc_val = history["val_acc"]
history_acc_std_val = history["val_std"]
# history_acc_target = history["target_acc"]
# history_acc_std_target = history["target_acc_std"]

# %%
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
axes[0].plot(history_loss, label="Train")
axes[0].plot(history_val_loss, label="Val")
axes[0].legend()
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")

# axes[1].plot(history_acc_night_train, label="Train")
# plot std
# axes[1].fill_between(
#     np.arange(len(history_acc_night_train)),
#     np.array(history_acc_night_train) - np.array(history_acc_night_std_train),
#     np.array(history_acc_night_train) + np.array(history_acc_night_std_train),
#     alpha=0.3,
# )
# axes[1].plot(history_acc_night_val, label="Val")
# axes[1].fill_between(
#     np.arange(len(history_acc_night_val)),
#     np.array(history_acc_night_val) - np.array(history_acc_night_std_val),
#     np.array(history_acc_night_val) + np.array(history_acc_night_std_val),
#     alpha=0.3,
# )
# axes[1].plot(history_acc_night_target, label="Target")
# axes[1].fill_between(
#     np.arange(len(history_acc_night_target)),
#     np.array(history_acc_night_target) - np.array(history_acc_night_std_target),
#     np.array(history_acc_night_target) + np.array(history_acc_night_std_target),
#     alpha=0.3,
# )
# axes[1].legend()
# axes[1].set_title("Accuracy per Night")
# axes[1].set_xlabel("Epoch")
# axes[1].set_ylabel("Accuracy")
# axes[1].set_ylim(0.5, 0.95)

# axes[2].plot(history_f1_night_train, label="Train")
# axes[2].fill_between(
#     np.arange(len(history_f1_night_train)),
#     np.array(history_f1_night_train) - np.array(history_f1_night_std_train),
#     np.array(history_f1_night_train) + np.array(history_f1_night_std_train),
#     alpha=0.3,
# )
# axes[2].plot(history_f1_night_val, label="Val")
# axes[2].fill_between(
#     np.arange(len(history_f1_night_val)),
#     np.array(history_f1_night_val) - np.array(history_f1_night_std_val),
#     np.array(history_f1_night_val) + np.array(history_f1_night_std_val),
#     alpha=0.3,
# )
# axes[2].plot(history_f1_night_target, label="Target")
# axes[2].fill_between(
#     np.arange(len(history_f1_night_target)),
#     np.array(history_f1_night_target) - np.array(history_f1_night_std_target),
#     np.array(history_f1_night_target) + np.array(history_f1_night_std_target),
#     alpha=0.3,
# )
# axes[2].legend()
# axes[2].set_title("F1 score per Night")
# axes[2].set_xlabel("Epoch")
# axes[2].set_ylabel("Accuracy")
# axes[2].set_ylim(0.5, 0.95)

axes[2].plot(history_acc_train, label="Train")
axes[2].plot(history_acc_val, label="Val")
axes[2].fill_between(
    np.arange(len(history_acc_val)),
    np.array(history_acc_val) - np.array(history_acc_std_val),
    np.array(history_acc_val) + np.array(history_acc_std_val),
    alpha=0.3,
)
axes[2].plot(history_acc_target, label="Target")
axes[2].fill_between(
    np.arange(len(history_acc_target)),
    np.array(history_acc_target) - np.array(history_acc_std_target),
    np.array(history_acc_target) + np.array(history_acc_std_target),
    alpha=0.3,
)
axes[2].legend()
axes[2].set_title("Accuracy")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Accuracy")
axes[2].set_ylim(0.5, 0.95)

plt.tight_layout()
# %%
fnames = list(Path("results/pickle").glob("results_*.pkl"))
df = pd.concat([pd.read_pickle(fname) for fname in fnames], axis=0)

df["f1"] = df.apply(lambda x: f1_score(x.y_target, x.y_pred, average="weighted"), axis=1)
df["acc"] = df.apply(lambda x: accuracy_score(x.y_target, x.y_pred), axis=1)
# replace None to "none"
df.replace({None: "none"}, inplace=True)
df["norm"] = df.apply(lambda x: x.tmanorm + str(x.tmalayer), axis=1)
# %%
import seaborn as sns
import matplotlib.pyplot as plt

df_baseline = df.query("tmatype in ['None', 'offline']")
df_plot = df.query("mean == 'geometric'")
df_plot = pd.concat([df_baseline, df_plot], axis=0)

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
df_1 = df_plot[df_plot.dataset_t == "CHAT"]
sns.boxplot(x="tmatype", y="f1", hue="filter_size", data=df_1, ax=axes[0, 0], palette="tab10")
axes[0, 0].set_title("F1")
axes[0, 0].set_ylim(0.5, 1)
axes[0, 0].grid(True)

sns.boxplot(x="tmatype", y="acc", hue="filter_size", data=df_1, ax=axes[0, 1], palette="tab10")
axes[0, 1].set_title("Accuracy")
axes[0, 1].set_ylim(0.5, 1)
# add grid
axes[0, 1].grid(True)

df_2 = df_plot[df_plot.dataset_t == "MASS"]
sns.boxplot(x="tmatype", y="f1", hue="filter_size", data=df_2, ax=axes[1, 0], palette="tab10")
axes[1, 0].set_ylim(0.5, 1)
sns.boxplot(x="tmatype", y="acc", hue="filter_size", data=df_2, ax=axes[1, 1], palette="tab10")
axes[1, 1].set_ylim(0.5, 1)
axes[1, 0].set_title("F1")
axes[1, 0].grid(True)
axes[1, 1].grid(True)
axes[1, 1].set_title("Accuracy")

for ax in axes.flatten():
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

plt.tight_layout()


# %%
# %%
import seaborn as sns
import matplotlib.pyplot as plt

df_plot = df.query("mean != 'geometric'")

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
df_1 = df_plot[df_plot.dataset_t == "CHAT"]
sns.boxplot(x="tmatype", y="f1", hue="filter_size", data=df_1, ax=axes[0, 0], palette="tab10")
axes[0, 0].set_title("F1")
axes[0, 0].set_ylim(0.5, 1)
axes[0, 0].grid(True)

sns.boxplot(x="tmatype", y="acc", hue="filter_size", data=df_1, ax=axes[0, 1], palette="tab10")
axes[0, 1].set_title("Accuracy")
axes[0, 1].set_ylim(0.5, 1)
# add grid
axes[0, 1].grid(True)

df_2 = df_plot[df_plot.dataset_t == "MASS"]
sns.boxplot(x="tmatype", y="f1", hue="filter_size", data=df_2, ax=axes[1, 0], palette="tab10")
axes[1, 0].set_ylim(0.5, 1)
sns.boxplot(x="tmatype", y="acc", hue="filter_size", data=df_2, ax=axes[1, 1], palette="tab10")
axes[1, 1].set_ylim(0.5, 1)
axes[1, 0].set_title("F1")
axes[1, 0].grid(True)
axes[1, 1].grid(True)
axes[1, 1].set_title("Accuracy")

for ax in axes.flatten():
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

plt.tight_layout()
# %%
# table
df.groupby(["norm", "dataset_t", "filter_size"]).f1.mean()
df.pivot_table(index="norm", columns=["dataset_t", "filter_size"], values="f1", aggfunc="mean")

# %%
df.groupby(["norm", "dataset_t"]).f1.mean()

# %%
barys = np.load("results/barycenter_usleep_CHAT_test.npy")
# %%
barys.shape
# %%
# make gif of barycenter
import matplotlib.pyplot as plt
import numpy as np
import imageio
from pathlib import Path
import os
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

barys = np.load("results/barycenter_usleep_CHAT_test.npy")
barys = barys[1:]
# normalize
# make gif
images = []
for i in range(barys.shape[0]):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(barys[i, 0])
    ax.set_title(f"Iter {i}")
    fig.savefig(f"results/barycenter/barycenter_{i}.png")
    plt.close(fig)
    images.append(f"results/barycenter/barycenter_{i}.png")
# %%

with imageio.get_writer("results/barycenter/barycenter.gif", mode="I") as writer:
    for filename in images:
        image = imageio.imread(filename)
        writer.append_data(image)
# %%
