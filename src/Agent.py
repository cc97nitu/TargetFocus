import random

from Struct import Action

class Agent(object):

    def __init__(self, valueFunction, greediness, actionSet, discount, learningRate,):
        """
        Creates an agent who follows an epsilon-greedy policy
        :param valueFunction: the value function considered
        :param greediness: probability to act greedy
        :param actionSet: set containing all possible changes to the magnets' deflections
        :param discount: discount factor from the expected future reward
        :param learningRate: learning rate used to create update targets
        """
        # initialize properties
        self.valueFunction = valueFunction
        self.greediness = greediness
        self.actionSet = actionSet
        self.discount = discount
        self.learningRate = learningRate

        self.episodeMemory = []  # list to store transitions observed in current episode

        return

    def act(self, state):
        """
        Act upon a given state under an epsilon-greedy policy
        :param state: current state
        :return: chosen action
        """
        # act greedy?
        if random.uniform(0, 1) < self.greediness:
            # greedy choice
            return self.valueFunction.getBestAction(state)
        else:
            # explorative move
            return Action(state, random.choice(self.actionSet))

    def learn_Q(self, memory):
        self.valueFunction.train(*self.valueFunction.genQTargets(memory, self.learningRate, self.discount))

        return


if __name__ == '__main__':
    from itertools import product

    import torch

    from Struct import Transition
    from Environment import Environment
    from QValue import QValue
    from FuncApprox.Network import FulConI1
    from FuncApprox.Trainer import SGD

    # action set
    possibleChangesPerMagnet = (1e-2, 1e-3, 0, -1e-2, -1e-3)
    actionSet = tuple(torch.tensor((x, y), dtype=torch.float) for x, y in
                      product(possibleChangesPerMagnet, possibleChangesPerMagnet))

    # environment and dummy state
    resolutionTarget, resolutionOversize = 80, 20
    resolution = resolutionTarget + resolutionOversize

    env = Environment(0, 0, resolutionTarget, resolutionOversize)
    state = env.initialState
    action = Action(state, torch.tensor((0.01, -0.01)))
    state, reward = env.react(action)

    # initialize value function
    valueFunction = QValue(FulConI1, resolution, actionSet, SGD, 20)

    # initialize agent
    agent = Agent(valueFunction, 0.9, actionSet, 0.9, 0.5)

    # act once and store the observed transition in the agent's memory
    action = agent.act(state)
    nextState, reward = env.react(action)

    transition = Transition(action, reward, nextState)
    agent.episodeMemory.append(transition)
