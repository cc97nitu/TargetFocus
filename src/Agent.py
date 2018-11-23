import torch
import random
from itertools import product

from Struct import Action


class Agent(object):
    """the agent"""

    def __init__(self, q, epsilon=0.8, discount=0.9, learningRate=0.5, traceDecay=0.3):
        # action set
        possibleChangesPerMagnet = (1e-2, 1e-3, 0, -1e-2, -1e-3)
        # possibleChangesPerMagnet = (0, -1e-2, -1e-3)
        self.actionSet = tuple(torch.tensor((x, y), dtype=torch.float) for x, y in
                               product(possibleChangesPerMagnet, possibleChangesPerMagnet))

        # probability to act greedy
        self.epsilon = epsilon

        # Q-function
        self.q = q

        # memory
        self.shortMemory = []
        self.memorySize = 50
        self.traceDecay = traceDecay

        # learning
        self.discount = discount
        self.learningRate = learningRate

        return

    def takeAction(self, state):
        """take an action according to current state"""
        # go greedy or not?
        if random.uniform(0, 1) < self.epsilon:
            # greedy selection
            # find best action
            allActions = torch.stack(
                tuple(torch.cat((state.strengths, state.focus, changes)) for changes in self.actionSet))
            evaluation = self.q.evaluateBunch(allActions)
            action = Action(state, self.actionSet[evaluation.argmax()])
            return action
        else:
            # random selection
            return Action(state, random.choice(self.actionSet))

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

    def learn(self, netInput, labels):
        """train Q-function"""
        self.q.trainer.applyUpdate(netInput, labels)
        return

    def getSarsaLambda(self, shortMemory):
        """generate TD lambda update targets from short memory"""
        # get temporal difference error
        delta = shortMemory[-1].reward + self.discount * self.q.evaluate(
            self.takeAction(shortMemory[-1].nextState)) - self.q.evaluate(shortMemory[-1].action)

        # states
        netInput = []
        for memory in shortMemory:
            netInput.append(
                torch.cat((memory.action.state.strengths, memory.action.state.focus, memory.action.changes)))

        netInput = torch.stack(netInput)

        # updates for every state in memory with respect to its eligibility
        labels = []
        for memory in shortMemory:
            labels.append(self.learningRate * delta * memory.action.eligibility)

        labels = torch.tensor(labels)
        labels = torch.unsqueeze(labels, 1)

        return netInput, labels

    def getDQN(self, shortMemory):
        """generates DQN update targets from short memory"""
        sampleSize = self.memorySize // 5

        if len(shortMemory) < sampleSize:
            sample = shortMemory
        else:
            sample = random.sample(shortMemory, sampleSize)

        # states
        netInput = []
        for memory in sample:
            netInput.append(
                torch.cat((memory.action.state.strengths, memory.action.state.focus, memory.action.changes)))

        netInput = torch.stack(netInput)

        # updates for Q-values
        labels = []
        for memory in sample:
            if memory.nextState:
                labels.append(memory.reward)
            else:
                currentQ = self.q.evaluate(memory.action)
                labels.append(currentQ + self.learningRate * (
                            self.discount * self.q.evaluateMax(memory.nextState, self.actionSet) - currentQ))

        labels = torch.tensor(labels)
        labels = torch.unsqueeze(labels, 1)

        return netInput, labels
