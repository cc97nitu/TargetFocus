import torch
import random
from itertools import product

from Struct import Action


class Agent(object):
    """the agent"""

    def __init__(self, q):
        # action set
        possibleChangesPerMagnet = (1e-2, 1e-3, 0, -1e-2, -1e-3)
        self.actionSet = tuple(torch.tensor((x, y), dtype=torch.float) for x, y in product(possibleChangesPerMagnet, possibleChangesPerMagnet))

        # probability to act greedy
        self.epsilon = 0

        # Q-function
        self.q = q

        # memory
        self.__shortMemory = []
        self.memorySize = 50
        self.traceDecay = 0.3

        # learning
        self.discount = 0.9
        self.learningRate = 0.5

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
            memory *= self.traceDecay * self.discount

        # add new memory
        if len(self.__shortMemory) < self.memorySize:
            self.__shortMemory.append(transition)
        else:
            del self.__shortMemory[0]
            self.__shortMemory.append(transition)

        return

    def getShortMemory(self):
        return self.__shortMemory

    def wipeShortMemory(self):
        """wipe all recent experience"""
        self.__shortMemory = []
        return

    def learn(self, netInput, labels):
        """train Q-function"""
        self.q.trainer.applyUpdate(netInput, labels)
        return

    def getSarsaLambda(self):
        """generate TD lambda update targets from short memory"""
        # get temporal difference error
        delta = self.__shortMemory[-1].reward + self.discount * self.q.evaluate(self.takeAction(self.__shortMemory[-1].nextState)) - self.q.evaluate(self.__shortMemory[-1].action)

        # current Q-values
        netInput = []
        for memory in self.__shortMemory:
            netInput.append(torch.cat((memory.action.state.strengths, memory.action.state.focus, memory.action.changes)))

        netInput = torch.stack(netInput)

        # updates for every state in memory with respect to its eligibility
        labels = []
        for memory in self.__shortMemory:
            labels.append(self.learningRate * delta * memory.action.eligibility)

        labels = torch.tensor(labels)
        labels = torch.unsqueeze(labels, 1)

        return netInput, labels





