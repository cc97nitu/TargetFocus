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


class FC2BN(nn.Module):
    def __init__(self, features: int, outputs: int):
        super(FC2BN, self).__init__()
        self.fc1 = nn.Linear(features, 40)
        self.fc2 = nn.Linear(40, 40)
        self.bn1 = nn.BatchNorm1d(40)
        self.bn2 = nn.BatchNorm1d(40)
        self.output = nn.Linear(40, outputs)
        self.activation = functional.elu
        return

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.bn1(x)
        x = self.activation(self.fc2(x))
        x = self.bn2(x)
        x = self.output(x)
        return x


class FC3(nn.Module):
    def __init__(self, features: int, outputs: int):
        super(FC3, self).__init__()
        self.fc1 = nn.Linear(features, 10)
        self.fc2 = nn.Linear(10, 10)
        self.output = nn.Linear(10, outputs)
        self.activation = functional.elu
        return

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.output(x)
        return x


class FC4(nn.Module):
    def __init__(self, features: int, outputs: int):
        super(FC4, self).__init__()
        self.fc1 = nn.Linear(features, 80)
        self.fc2 = nn.Linear(80, 80)
        self.output = nn.Linear(80, outputs)
        self.activation = functional.elu
        return

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.output(x)
        return x


class FC5(nn.Module):
    def __init__(self, features: int, outputs: int):
        super(FC5, self).__init__()
        self.fc1 = nn.Linear(features, 40)
        self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(40, 40)
        self.fc4 = nn.Linear(40, 40)
        self.output = nn.Linear(40, outputs)
        self.activation = functional.elu
        return

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.output(x)
        return x


class FC5BN(nn.Module):
    def __init__(self, features: int, outputs: int):
        super(FC5BN, self).__init__()
        self.fc1 = nn.Linear(features, 40)
        self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(40, 40)
        self.fc4 = nn.Linear(40, 40)
        self.bn1 = nn.BatchNorm1d(40)
        self.bn2 = nn.BatchNorm1d(40)
        self.bn3 = nn.BatchNorm1d(40)
        self.bn4 = nn.BatchNorm1d(40)
        self.output = nn.Linear(40, outputs)
        self.activation = functional.elu
        return

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.bn1(x)
        x = self.activation(self.fc2(x))
        x = self.bn2(x)
        x = self.activation(self.fc3(x))
        x = self.bn3(x)
        x = self.activation(self.fc4(x))
        x = self.bn4(x)
        x = self.output(x)
        return x
