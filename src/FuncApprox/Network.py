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
