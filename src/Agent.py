import torch
import random
from itertools import product
from random import shuffle

from Struct import Action


class Agent(object):
    """
    Represents the agent.
    """

    def __init__(self, q, epsilon, discount, learningRate, memorySize, traceDecay, targetGenerator):
        # action set
        possibleChangesPerMagnet = (1e-2, 1e-3, 0, -1e-3, -1e-2)
        self.actionSet = tuple(torch.tensor((x, y), dtype=torch.float) for x, y in
                               product(possibleChangesPerMagnet, possibleChangesPerMagnet))

        # probability to act greedy
        self.epsilon = epsilon

        # Q-function
        self.q = q

        # memory
        self.shortMemory = []
        self.memorySize = memorySize
        self.traceDecay = traceDecay

        self.replayMemory = []
        self.replayMemorySize = int(1e5)

        # learning
        self.discount = discount
        self.learningRate = learningRate
        self.targetGenerator = targetGenerator

        return

    def takeAction(self, state):
        """take an action according to current state"""
        # go greedy or not?
        if random.uniform(0, 1) < self.epsilon:
            # greedy selection
            # find best action
            allActions = torch.stack(
                tuple(torch.cat((state.strengths, state.relCoord, changes)) for changes in self.actionSet))
            evaluation = self.q.evaluateBunch(allActions)
            action = Action(state, self.actionSet[evaluation.argmax()])
            return action
        else:
            # random selection
            return Action(state, random.choice(self.actionSet))

    def bestAction(self, state, isTensor=False):
        """returns best action and it's rating"""
        # get value for every possible action
        if not isTensor:
            allActions = torch.stack(
                tuple(torch.cat((state.strengths, state.relCoord, changes)) for changes in self.actionSet))
        else:
            allActions = torch.stack(
                tuple(torch.cat((state, changes)) for changes in self.actionSet))

        allValues = self.q.evaluateBunch(allActions)

        # determine index of highest value
        bestIndex = allValues.argmax()

        # get best action
        bestAction = allActions[bestIndex, -2:]

        return bestAction, allValues[bestIndex]

    def remember(self, transition):
        """place a transition in the memory"""
        # reduce eligibility for old memories
        for memory in self.shortMemory:
            memory *= self.traceDecay * self.discount

        # add new memory
        if len(self.shortMemory) < self.memorySize:
            self.shortMemory.append(transition)
        else:
            del self.shortMemory[0]
            self.shortMemory.append(transition)

        return

    def getShortMemory(self):
        return self.shortMemory

    def wipeShortMemory(self):
        """wipe all recent experience"""
        self.shortMemory = []
        return

    def updateReplayMemory(self, memories):
        """
        Store new memories in the agent's replay memory. The replay memory is then shuffled.
        :param memories: list of Transition objects to append to the replay memory.
        :return: None
        """
        # make space if there is no one for new memories
        while len(self.replayMemory) + len(memories) > self.replayMemorySize:
            del self.replayMemory[0]

        self.replayMemory += memories
        shuffle(self.replayMemory)

        return

    def learn(self, memory):
        """train Q-function"""
        self.q.trainer.applyUpdate(*self.targetGenerator(self, memory))
        return

    def __repr__(self):
        return "Agent: learningRate={}, discount={}, epsilon={}, network={}, targetGenerator={}, memorySize={}, traceDecay={}".format(
            self.learningRate, self.discount, self.epsilon, str(self.q.network), self.targetGenerator.__name__,
            self.memorySize, self.traceDecay)
