import torch, torchvision
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch_train import TorchTrain
from torchmetrics import Accuracy, Precision, Recall


to_tensor = transforms.ToTensor()
train_data = datasets.FashionMNIST(
    root="data", train=True, download=True, transform=to_tensor
)
test_data = datasets.FashionMNIST(
    root="data", train=False, download=True, transform=to_tensor
)

BATCH_SIZE = 32
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


class Model0(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        return x


model0 = Model0()

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model0.parameters(), lr=0.001)

acc = Accuracy(task="multiclass", num_classes=10)
precision = Precision(num_classes=10, task="multiclass")
recall = Recall(num_classes=10, task="multiclass")
custom_metric = lambda yhat, y: 10.000
metrics = {
    "accuracy": acc,
    "precision": precision,
    # "recall": recall,
    "custom": custom_metric,
}

tt = TorchTrain(model0, optimizer, loss, metrics=metrics)
history = tt.fit(
    train_loader,
    test_loader,
    epochs=3,
    train_steps_per_epoch=300,
    validation_steps_per_epoch=200,
)
print(history)
