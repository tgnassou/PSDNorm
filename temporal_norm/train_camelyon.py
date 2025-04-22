# %%
import copy
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from config import LMDB_PATH
from temporal_norm.utils import LMDBImageDataset
from temporal_norm.utils.architecture import DenseNet


config = {
    "split_scheme": "official",
    "model": "densenet121",
    "model_kwargs": {"pretrained": False},
    "transform": "image_base",
    "target_resolution": (96, 96),
    "loss_function": "cross_entropy",
    "groupby_fields": ["hospital"],
    "val_metric": "acc_avg",
    "val_metric_decreasing": False,
    "optimizer": "SGD",
    "momentum": 0.9,
    "scheduler": None,
    "batch_size": 256,
    "lr": 0.001,
    "weight_decay": 0.01,
    "n_epochs": 50,
    "patience": 5,
    "n_groups_per_batch": 2,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "n_classes": 2,
    "norm": "PSDNorm",
}

print("norm:", config["norm"])
print_tqdm = False
# %%
# Load the dataset
print("Loading dataset")
transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize(config["target_resolution"]),
        transforms.ToTensor(),
    ]
)
train_data = LMDBImageDataset(LMDB_PATH, transform=transform, domain_filter=[0, 1, 2])
val_data = LMDBImageDataset(LMDB_PATH, transform=transform, domain_filter=[3])
test_data = LMDBImageDataset(LMDB_PATH, transform=transform, domain_filter=[4])


# %%
dataloader_train = torch.utils.data.DataLoader(
    train_data,
    batch_size=config["batch_size"],
    drop_last=True,
    num_workers=6,
    shuffle=True,
)
dataloader_val = torch.utils.data.DataLoader(
    val_data,
    batch_size=config["batch_size"],
    drop_last=True,
    num_workers=6,
    shuffle=False,
)
dataloader_target = torch.utils.data.DataLoader(
    test_data,
    batch_size=config["batch_size"],
    drop_last=False,
    num_workers=6,
    shuffle=False,
)

print(f"Number of training batches: {len(dataloader_train)}")
print(f"Number of validation batches: {len(dataloader_val)}")
print(f"Number of target batches: {len(dataloader_target)}")
print()
# %%

model = DenseNet(num_classes=1, norm=config["norm"], filter_size=5)

model = model.to(config["device"])

num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {num_trainable_params:,}")

# model = torch.compile(model)
# %%
loss = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(
    model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
)

# %%
print()
print("Start training")
min_val_loss = np.inf
for epoch in range(config["n_epochs"]):
    model.train()
    time_start = time.time()
    y_true_all = []
    y_pred_all = []
    train_loss = np.zeros(len(dataloader_train))
    running_loss = 0.0
    running_window = len(dataloader_train) // 20  # Number of batches for averaging loss
    for i, batch in enumerate(
        tqdm(dataloader_train, desc="Training", unit="batch", disable=not print_tqdm)
    ):
        optimizer.zero_grad()
        batch_X, batch_y, domain = batch
        batch_X = batch_X.to(config["device"], non_blocking=True)
        batch_y = batch_y.to(config["device"], non_blocking=True)
        output = model(batch_X).squeeze(1)
        # Convert batch_y to the same shape as output
        loss_batch = loss(output, batch_y.float())
        loss_batch.backward()
        optimizer.step()

        y_true_all.append(batch_y)
        y_pred = torch.sigmoid(output)
        y_pred = (y_pred > 0.5).float()
        y_pred_all.append(y_pred)
        train_loss[i] = loss_batch.item()

        running_loss += loss_batch.item()
        if (i + 1) % running_window == 0 and print_tqdm:
            avg_loss = running_loss / running_window
            tqdm.write(f"Batch {i+1}/{len(dataloader_train)}, Avg Loss: {avg_loss:.3f}")
            running_loss = 0.0

    y_pred_all = [y.cpu().numpy() for y in y_pred_all]
    y_true_all = [y.cpu().numpy() for y in y_true_all]

    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)

    perf = accuracy_score(y_true, y_pred)

    model.eval()
    y_true_all = []
    y_pred_all = []
    with torch.no_grad():
        val_loss = np.zeros(len(dataloader_val))
        for i, batch in enumerate(dataloader_val):
            x, y, domain = batch
            x = x.to(config["device"])
            y = y.to(config["device"])
            output = model(x).squeeze(1)
            loss_batch = loss(output, y.float())

            y_true_all.append(y)
            y_pred = torch.sigmoid(output)
            y_pred = (y_pred > 0.5).float()
            y_pred_all.append(y_pred)
            val_loss[i] = loss_batch.item()

        y_pred_all = [y.cpu().numpy() for y in y_pred_all]
        y_true_all = [y.cpu().numpy() for y in y_true_all]

        y_pred = np.concatenate(y_pred_all)
        y_true = np.concatenate(y_true_all)

        perf_val = accuracy_score(y_true, y_pred)

    time_end = time.time()
    # Print the results
    print(
        "Ep:",
        epoch,
        "Loss:",
        round(np.mean(train_loss), 2),
        "Acc:",
        round(np.mean(perf), 2),
        "LossVal:",
        round(np.mean(val_loss), 2),
        "AccVal:",
        round(np.mean(perf_val), 2),
        "Time:",
        round(time_end - time_start, 2),
    )

    # do early stopping
    if min_val_loss > np.mean(val_loss):
        min_val_loss = np.mean(val_loss)
        patience_counter = 0
        best_model = copy.deepcopy(model)
    else:
        patience_counter += 1
        if patience_counter > config["patience"]:
            print("Early stopping")
            break

# %%
print("Testing")
model.eval()
y_true_all = []
y_pred_all = []
iteration = 0
with torch.no_grad():
    for batch in dataloader_target:
        x, y, domain = batch
        x = x.to(config["device"])
        y = y.to(config["device"])
        output = best_model(x).squeeze(1)

        y_true_all.append(y)
        y_pred = torch.sigmoid(output)
        y_pred = (y_pred > 0.5).float()
        y_pred_all.append(y_pred)

    y_pred_all = [y.cpu().numpy() for y in y_pred_all]
    y_true_all = [y.cpu().numpy() for y in y_true_all]

    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)

    perf_test = accuracy_score(y_true, y_pred)
print(f"Test Acc: {perf_test}")

results = {
    "y_true": y_true_all, "y_pred": y_pred_all, "domain": domain, "norm": "BatchNorm"
}

# Save the results
output_dir = Path("results")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "camelyon17_results.pkl"
with open(output_path, "wb") as f:
    pickle.dump(results, f)
    # Print the results
print(f"Results saved to {output_path}")
