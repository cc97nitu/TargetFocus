import abc
import torch

import FuncApprox.Trainer as Trainer
from FuncApprox.Network import FulCon1


class QFunction(object, metaclass=abc.ABCMeta):
    """abstract version of the action-value function"""

    @abc.abstractmethod
    def evaluate(self, action):
        """evaluate an action"""
        pass

    @abc.abstractmethod
    def update(self, action, update):
        """update value estimate for action"""
        pass


class QNeural(QFunction):
    """constitute the Q-Function with a neural network"""

    def __init__(self, network=None, trainer=None, epochs=20):
        # initialize network
        if network is not None:
            self.network = network()
        else:
            self.network = FulCon1()

        # initialize trainer
        if trainer is not None:
            self.trainer = trainer(self.network, epochs=epochs)
        else:
            self.trainer = Trainer.Adam(self.network, epochs=epochs)
        return

    def evaluate(self, action):
        """get Q-value for action"""
        netInput = torch.cat((action.state.strengths, action.state.relCoord, action.changes)).unsqueeze(0)

        return self.network(netInput).item()

    def evaluateMax(self, state, actionSet):
        netInput = []
        for action in actionSet:
            netInput.append(torch.cat((state.strengths, state.relCoord, action)))

        netInput = torch.stack(netInput)

        return self.network(netInput).max().item()

    def evaluateBunch(self, bunch):
        return self.network(bunch)

    def update(self, action, update):
        pass
