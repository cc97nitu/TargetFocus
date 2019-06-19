from __future__ import annotations
import random
from collections import namedtuple

import torch

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


if __name__ == '__main__':
    state = State(torch.randn(2), torch.randn(2))
    changes = torch.randn(2)

    firstAction = Action(state, changes)
    secondAction = Action(state, changes)

    if firstAction == secondAction:
        print(True)
