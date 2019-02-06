from itertools import product

import torch

from Environment import Environment
from Struct import Action
from FuncApprox.Network import FulConI1


class QValue(object):
    """Implementation of the value function.
    Acts as a wrapper for the neural network. All data conversion from Struct objects to input/output tensors is handled here."""

    def __init__(self, network, resolution, actionSet, trainer=None, epochs=None):
        """
        Constructor
        :param network: network to use
        :param resolution: resolution of the image representing the state
        :param actionSet: number of actions to be evaluated, matches number of output neurons
        :param trainer: trainer to use for training
        :param epochs: number of epochs used for supervised learning on a single training call
        """
        # store action set
        self.actionSet = actionSet

        # initialize network
        self.network = network(resolution, len(actionSet))

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
        # netInput = state.image

        return self.network(netInput)

    def getBestAction(self, state):
        """
        Get the networks suggestion when being in state
        :param state: current state
        :return: Action object
        """
        bestIndex = self.getActionValues(state).argmax()

        return Action(state, self.actionSet[bestIndex])

    def train(self, samples, labels):
        """
        Train the neural network
        :param samples: input
        :param labels: designated output
        :return: None
        """
        self.trainer.applyUpdate(samples, labels)

        return

    def genQTargets(self, transitionList, learningRate, discount):
        """
        Generate update targets for the network according to the Q-learning rule
        :param transitionList: list containing observed transitions
        :param learningRate: learning rate to use
        :param discount: discount factor of the expected future reward
        :return: two torch tensors containing samples and corresponding labels
        """
        samples, labels = [], []

        for transition in transitionList:
            # append the state as sample
            samples.append(transition.action.state.image.view(-1))
            # samples.append(transition.action.state.image)

            # get current prediction
            prediction = self.getActionValues(transition.action.state)

            # get maximum q-value for the next state
            maxQNext = self.getActionValues(transition.nextState).max()

            # find index of the action subject to update
            for i in range(0, len(self.actionSet)):
                if torch.equal(transition.action.changes, self.actionSet[i]):
                    # update current prediction according to Q-learning

                    prediction[i] = prediction[i] + learningRate * (transition.reward + discount * maxQNext - prediction[i])
                    break

            labels.append(prediction)

        return torch.stack(samples).detach(), torch.stack(labels).detach()  # detach from their graphs so gradients can be calculated properly during training


if __name__ == '__main__':
    from Struct import Transition
    from FuncApprox.Trainer import SGD

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
    valueFunction = QValue(FulConI1, resolution, actionSet, SGD, 20)

    # observe a transition
    action = valueFunction.getBestAction(state)
    nextState, reward = env.react(action)
    transition = Transition(action, reward, nextState)

    # generate learning targets
    samples, labels = valueFunction.genQTargets([transition, transition, transition], 0.5, 0.9)


