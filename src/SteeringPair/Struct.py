from __future__ import annotations
import random
from collections import namedtuple

import torch

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class CyclicBuffer(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, item):
        """Stores an item."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class ReplayMemory(CyclicBuffer):

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity


class RunningStat(object):
    """Track mean and standard deviation of the incoming samples."""
    def __init__(self, sampleShape):
        self.numberSamples = 0
        self.first = torch.zeros(sampleShape)
        self.second = torch.zeros(sampleShape)
        self.epsilon = 1e-9  # added to the denominator of std for numerical stability

    def append(self, sample):
        self.numberSamples += 1
        self.first += sample
        self.second += sample**2

    def stats(self):
        mean = self.first / self.numberSamples
        std = torch.sqrt(self.numberSamples * self.second - self.first**2) / self.numberSamples
        return mean, std

    def runningNorm(self, sample):
        self.append(sample)
        mean, std = self.stats()
        return (sample - mean) / (std + self.epsilon)

    def __repr__(self):
        return "mean: {}, std: {}".format(*self.stats())


if __name__ == "__main__":
    import torch.distributions

    # set up a distribution
    numberFeatures = 6

    mean = torch.randn(numberFeatures)
    std = torch.randn(numberFeatures) * 1e-3
    print("mean: {}, std: {}".format(mean, torch.abs(std)))

    dist = torch.distributions.multivariate_normal.MultivariateNormal(mean, torch.diag(std)**2)

    # a batch dimension gets added
    run = RunningStat(numberFeatures)

    for i in range(int(1e1)):
        s = dist.sample()
        run.append(s)

    print("observed")
    print(run)


