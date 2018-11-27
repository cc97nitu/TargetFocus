import torch
import torch.nn as nn
import torch.nn.functional as functional


class FulCon1(nn.Module):
    """simple network for testing"""

    def __init__(self):
        super(FulCon1, self).__init__()

        self.inputLayer = nn.Linear(6, 40)
        self.layer1 = nn.Linear(40, 1)
        # self.outputLayer = nn.Linear(1, 1)

        return

    def forward(self, x):
        x = self.inputLayer(x)

        x = functional.elu(self.layer1(x))

        # x = self.outputLayer(x)

        return x

    def __repr__(self):
        return "FulCon1"


class FulCon2(nn.Module):
    """simple network for testing"""

    def __init__(self):
        super(FulCon2, self).__init__()

        self.inputLayer = nn.Linear(6, 60)
        self.layer1 = nn.Linear(60, 1)
        # self.outputLayer = nn.Linear(1, 1)

        return

    def forward(self, x):
        x = self.inputLayer(x)

        x = functional.elu(self.layer1(x))

        # x = self.outputLayer(x)

        return x

    def __repr__(self):
        return "FulCon2"


class FulCon3(nn.Module):
    """simple network for testing"""

    def __init__(self):
        super(FulCon3, self).__init__()

        self.inputLayer = nn.Linear(6, 80)
        self.layer1 = nn.Linear(80, 1)
        # self.outputLayer = nn.Linear(1, 1)

        return

    def forward(self, x):
        x = self.inputLayer(x)

        x = functional.elu(self.layer1(x))

        # x = self.outputLayer(x)

        return x

    def __repr__(self):
        return "FulCon3"


class FulCon4(nn.Module):
    """simple network for testing"""

    def __init__(self):
        super(FulCon4, self).__init__()

        self.inputLayer = nn.Linear(6, 120)
        self.layer1 = nn.Linear(120, 1)
        # self.outputLayer = nn.Linear(1, 1)

        return

    def forward(self, x):
        x = self.inputLayer(x)

        x = functional.elu(self.layer1(x))

        # x = self.outputLayer(x)

        return x

    def __repr__(self):
        return "FulCon4"


class FulCon5(nn.Module):
    """simple network for testing"""

    def __init__(self):
        super(FulCon5, self).__init__()

        self.inputLayer = nn.Linear(6, 20)
        self.layer1 = nn.Linear(20, 1)
        # self.outputLayer = nn.Linear(1, 1)

        return

    def forward(self, x):
        x = self.inputLayer(x)

        x = functional.elu(self.layer1(x))

        # x = self.outputLayer(x)

        return x

    def __repr__(self):
        return "FulCon5"


class FulCon6(nn.Module):
    """simple network for testing"""

    def __init__(self):
        super(FulCon6, self).__init__()

        self.inputLayer = nn.Linear(6, 40)
        self.layer1 = nn.Linear(40, 40)
        self.layer2 = nn.Linear(40, 1)
        # self.outputLayer = nn.Linear(1, 1)

        return

    def forward(self, x):
        x = self.inputLayer(x)

        x = functional.elu(self.layer1(x))
        x = functional.elu(self.layer2(x))

        # x = self.outputLayer(x)

        return x

    def __repr__(self):
        return "FulCon6"


class FulCon7(nn.Module):
    """simple network for testing"""

    def __init__(self):
        super(FulCon7, self).__init__()

        self.inputLayer = nn.Linear(6, 80)
        self.layer1 = nn.Linear(80, 80)
        self.layer2 = nn.Linear(80, 1)
        # self.outputLayer = nn.Linear(1, 1)

        return

    def forward(self, x):
        x = self.inputLayer(x)

        x = functional.elu(self.layer1(x))
        x = functional.elu(self.layer2(x))

        # x = self.outputLayer(x)

        return x

    def __repr__(self):
        return "FulCon7"


class FulCon8(nn.Module):
    """simple network for testing"""

    def __init__(self):
        super(FulCon8, self).__init__()

        self.inputLayer = nn.Linear(6, 40)
        self.layer1 = nn.Linear(40, 40)
        self.layer2 = nn.Linear(40, 40)
        self.layer3 = nn.Linear(40, 1)
        # self.outputLayer = nn.Linear(1, 1)

        return

    def forward(self, x):
        x = self.inputLayer(x)

        x = functional.elu(self.layer1(x))
        x = functional.elu(self.layer2(x))
        x = functional.elu(self.layer3(x))

        # x = self.outputLayer(x)

        return x

    def __repr__(self):
        return "FulCon8"


class FulCon9(nn.Module):
    """simple network for testing"""

    def __init__(self):
        super(FulCon9, self).__init__()

        self.inputLayer = nn.Linear(6, 80)
        self.layer1 = nn.Linear(80, 80)
        self.layer2 = nn.Linear(80, 80)
        self.layer3 = nn.Linear(80, 1)
        # self.outputLayer = nn.Linear(1, 1)

        return

    def forward(self, x):
        x = self.inputLayer(x)

        x = functional.elu(self.layer1(x))
        x = functional.elu(self.layer2(x))
        x = functional.elu(self.layer3(x))

        # x = self.outputLayer(x)

        return x

    def __repr__(self):
        return "FulCon9"
