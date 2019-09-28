from abc import ABC, abstractmethod

from SteeringPair_Stochastic import Environment

class AbstractModel(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        # number features describing a state
        self.numberFeatures = Environment.features
        self.numberActions = len(Environment.actionSet)
        self.device = Environment.device

    @abstractmethod
    def to_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, dictionary: dict):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass


class AbstractTrainer(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def trainAgent(self, numEpisodes):
        pass

    @abstractmethod
    def benchAgent(self, numEpisodes):
        pass
