import torch.nn as nn
import torch.nn.functional as functional

class FC1(nn.Module):
    def __init__(self, features: int, outputs: int):
        super(FC1, self).__init__()
        self.fc1 = nn.Linear(features, 40)
        self.output = nn.Linear(40, outputs)
        self.activation = functional.elu
        return

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.output(x)
        return x


class FC2(nn.Module):
    def __init__(self, features: int, outputs: int):
        super(FC2, self).__init__()
        self.fc1 = nn.Linear(features, 40)
        self.fc2 = nn.Linear(40, 40)
        self.output = nn.Linear(40, outputs)
        self.activation = functional.elu
        return

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.output(x)
        return x

