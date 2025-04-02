# %%
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import torch
import time

config = {
    'split_scheme': 'official',
    'model': 'densenet121',
    'model_kwargs': {'pretrained': False},
    'transform': 'image_base',
    'target_resolution': (96, 96),
    'loss_function': 'cross_entropy',
    'groupby_fields': ['hospital'],
    'val_metric': 'acc_avg',
    'val_metric_decreasing': False,
    'optimizer': 'SGD',
    'momentum': 0.9,
    'scheduler': None,
    'batch_size': 32,
    'lr': 0.001,
    'weight_decay': 0.01,
    'n_epochs': 10,
    'n_groups_per_batch': 2,
    'device': 'cuda',
}
# %%

dataset = get_dataset(dataset="camelyon17", download=True)
# %%
train_data = dataset.get_subset(
    "train",
    transform=transforms.ToTensor(),
)
val_data = dataset.get_subset(
    "val",
    transform=transforms.ToTensor(),
)
test_data = dataset.get_subset(
    "test",
    transform=transforms.ToTensor(),
)
# %%
train_loader = get_train_loader("standard", train_data, batch_size=config['batch_size'])
val_loader = torch.utils.data.DataLoader(val_data, batch_size=config['batch_size'])
test_loader = torch.utils.data.DataLoader(test_data, batch_size=config['batch_size'])

# %%

last_layer_name = "classifier"
constructor = getattr(torchvision.models, config['model'])
model = constructor()
d_features = getattr(model, last_layer_name).in_features
setattr(model, last_layer_name, nn.Linear(d_features, dataset.n_classes))

model = model.to(config['device'])
# %%
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=config['lr'],
    momentum=config["momentum"],
    weight_decay=config['weight_decay']
)
# %%
for epoch in range(config['n_epochs']):
    model.train()
    time_init = time.time()
    y_all = []
    y_pred_all = []
    for batch in train_loader:
        x, y, metadata = batch
        x = x.to(config['device'])
        y = y.to(config['device'])
        optimizer.zero_grad()
        y_pred = model(x)
        loss_value = loss(y_pred, y)

        loss_value.backward()
        optimizer.step()

        y_all.append(y)
        y_pred_all.append(y_pred)
        break

    y_all = torch.cat(y_all)
    y_pred_all = torch.cat(y_pred_all)
    acc = (y_all == y_pred_all.argmax(dim=1)).float().mean()

    model.eval()
    y_all = []
    y_pred_all = []
    for batch in val_loader:
        x, y, metadata = batch
        x = x.to(config['device'])
        y = y.to(config['device'])
        y_pred = model(x)
        loss_value = loss(y_pred, y)

        y_all.append(y)
        y_pred_all.append(y_pred)

    y_all = torch.cat(y_all)
    y_pred_all = torch.cat(y_pred_all)
    acc_val = (y_all == y_pred_all.argmax(dim=1)).float().mean()

    print(f"Epoch {epoch}, Loss: {loss_value.item()}, Train Acc: {acc.item()}, Val Acc: {acc_val.item()}, Time: {time.time() - time_init} s")

# %%

print("Testing")
model.eval()
y_all = []
y_pred_all = []
for batch in test_loader:
    x, y, metadata = batch
    x = x.to(config['device'])
    y = y.to(config['device'])
    y_pred = model(x)
    y_all.append(y)
    y_pred_all.append(y_pred)

y_all = torch.cat(y_all)
y_pred_all = torch.cat(y_pred_all)
acc_test = (y_all == y_pred_all.argmax(dim=1)).float().mean()
print(f"Test Acc: {acc_test.item()}")