import random
from itertools import product

from Struct import Action


class Agent(object):
    """the agent"""

    def __init__(self, q):
        # action set
        possibleChangesPerMagnet = (1e-1, 1e-2, 1e-3, 0, -1e-1, -1e-2, -1e-3)
        self.actionSet = tuple((x, y) for x, y in product(possibleChangesPerMagnet, possibleChangesPerMagnet))

        # probability to act greedy
        self.epsilon = 0

        # Q-function
        self.q = q

        return

    def takeAction(self, state):
        """take an action according to current state"""
        # go greedy or not?
        if random.uniform(0, 1) < self.epsilon:
            # greedy
            return None
        else:
            # random selection
            changes = random.choice(self.actionSet)
            return Action(state, changes[0], changes[1])

