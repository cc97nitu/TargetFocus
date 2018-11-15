import torch
import random
from itertools import product

from Struct import Action


class Agent(object):
    """the agent"""

    def __init__(self, q):
        # action set
        possibleChangesPerMagnet = (1e-1, 1e-2, 1e-3, 0, -1e-1, -1e-2, -1e-3)
        self.actionSet = tuple(torch.tensor((x, y), dtype=torch.float) for x, y in product(possibleChangesPerMagnet, possibleChangesPerMagnet))

        # probability to act greedy
        self.epsilon = 1

        # Q-function
        self.q = q

        # memory
        self.__shortMemory = []
        self.memorySize = 50
        self.traceDecay = 0.3

        return

    def takeAction(self, state):
        """take an action according to current state"""
        # go greedy or not?
        if random.uniform(0, 1) < self.epsilon:
            # greedy selection
            # find best action
            allActions = torch.stack(tuple(torch.cat((state.strengths, state.focus, changes)) for changes in self.actionSet))
            evaluation = self.q.evaluateBunch(allActions)
            action = Action(state, self.actionSet[evaluation.argmax()])
            return action
        else:
            # random selection
            return Action(state, random.choice(self.actionSet))

    def remember(self, transition):
        """place a transition in the memory"""
        # reduce eligibility for old memories
        for memory in self.__shortMemory:
            memory *= self.traceDecay

        # add new memory
        if len(self.__shortMemory) < self.memorySize:
            self.__shortMemory.append(transition)
        else:
            del self.__shortMemory[0]
            self.__shortMemory.append(transition)

        return

    def getShortMemory(self):
        print(self.__shortMemory)
        return

    def getTDlambdaTargets(self, traceDecay):
        """generate TD lambda update targets from short memory"""
        pass


