import torch.nn as nn
import torch


class Model0(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        return x


class Conv1(nn.Module):
    def __init__(self, color_channel: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=color_channel,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 7 * 7, out_features=output_shape),
        )

    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return x
        
class TwoInputs(nn.Module):
    def __init__(self, neurons=10, in1=1, in2=1):
        super(TwoInputs, self).__init__()
        self.neurons = neurons
        self.in1 = in1
        self.in2 = in2
        self.b1 = self._branch_one()
        self.b2 = self._branch_two()

    def _branch_one(self):
        b1 = nn.Sequential(
            nn.Linear(self.in1, self.neurons),
            nn.ReLU(),
            nn.Linear(self.neurons, self.neurons),
        )
        return b1
    
    def _branch_two(self):
        b2 = nn.Sequential(
            nn.Linear(self.in2, self.neurons),
            nn.ReLU(),
            nn.Linear(self.neurons, self.neurons),
        )
        return b2
        
    def forward(self, x1, x2):
        x1 = self.b1(x1)
        x2 = self.b2(x2)
        x = x1*x2
        x = torch.sum(x, dim=1).unsqueeze(1)
        return x
