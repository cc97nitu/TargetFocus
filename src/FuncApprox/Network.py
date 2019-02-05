import torch
import torch.nn as nn
import torch.nn.functional as functional


class FulConI1(nn.Module):
    """simple network for testing"""

    def __init__(self, resolution, numberOfActions):
        super(FulConI1, self).__init__()

        self.inputLayer = nn.Linear(resolution ** 2, 100)
        self.layer1 = nn.Linear(100, numberOfActions)

        return

    def forward(self, x):
        x = self.inputLayer(x)

        x = functional.elu(self.layer1(x))

        return x

    def __repr__(self):
        return "FulCon1"


if __name__ == '__main__':
    resolution = 30
    numberOfActions = 25

    net = FulConI1(resolution, numberOfActions)

    testImages = torch.randn(10, resolution ** 2)
