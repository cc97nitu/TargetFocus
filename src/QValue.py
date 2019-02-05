from itertools import product

import torch

from Environment import Environment
from Struct import Action
from FuncApprox.Network import FulConI1


class QValue(object):
    """Implementation of the value function.
    Acts as a wrapper for the neural network. All data conversion from Struct objects to input/output tensors is handled here."""

    def __init__(self, network, resolution, numberOfActions, trainer=None, epochs=None):
        """
        Constructor
        :param network: network to use
        :param resolution: resolution of the image representing the state
        :param numberOfActions: number of actions to be evaluated, matches number of output neurons
        :param trainer: trainer to use for training
        :param epochs: number of epochs used for supervised learning on a single training call
        """
        # initialize network
        self.network = network(resolution, numberOfActions)

        # initialize trainer
        if trainer is not None:
            if epochs is None:
                raise ValueError("epochs cannot be empty for training")

            self.trainer = trainer(self.network, epochs=epochs)
        else:
            self.trainer = None

        return

    def getActionValues(self, state):
        """
        Get Q(s, a) for all actions a when being in state s
        :param state: current state
        :return: tensor consisting of Q(s, a)
        """
        netInput = state.image.view(-1)

        return self.network(netInput)

    def getBestAction(self, state, changesSet):
        """
        Get the networks suggestion when being in state
        :param state: current state
        :param changesSet: set of possible changes (= actions)
        :return: Action object
        """
        bestIndex = self.getActionValues(state).argmax()

        return Action(state, changesSet[bestIndex])

    def train(self, samples, labels):
        """
        Train the neural network
        :param samples: input
        :param labels: designated output
        :return: None
        """
        self.trainer.applyUpdate(samples, labels)

        return


if __name__ == '__main__':
    # target resolution
    resolutionTarget, resolutionOversize = 80, 20
    resolution = resolutionTarget + resolutionOversize

    # action set
    possibleChangesPerMagnet = (1e-2, 1e-3, 0, -1e-2, -1e-3)
    actionSet = tuple(torch.tensor((x, y), dtype=torch.float) for x, y in
                      product(possibleChangesPerMagnet, possibleChangesPerMagnet))

    # environment and dummy state
    env = Environment(0, 0, resolutionTarget, resolutionOversize)
    state = env.initialState
    action = Action(state, torch.tensor((0.01, -0.01)))
    state, reward = env.react(action)

    # initialize value function
    valueFunction = QValue(FulConI1, resolution, len(actionSet))


