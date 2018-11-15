import abc
import torch


class Trainer(object, metaclass=abc.ABCMeta):
    """abstract base class for Trainer"""

    @abc.abstractmethod
    def __init__(self, network):
        self.network = network

    @abc.abstractmethod
    def applyUpdate(self, sample, label):
        """train the approximator"""
        pass


class SimpleTrainer(Trainer):
    def __init__(self, network):
        super().__init__(network)

        self.epochs = 10

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=0.001)

        return

    def applyUpdate(self, sample, label):
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            out = self.network(sample)
            loss = self.criterion(out, label)
            loss.backward()
            self.optimizer.step()

        return


class DecayTrainer(Trainer):
    def __init__(self, network, epochs=100, schedulerLRDecay=0.95):
        super().__init__(network)

        self.epochs = epochs
        self.schedulerLRDecay = schedulerLRDecay

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=0.001)

        return

    def applyUpdate(self, sample, label):
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=self.schedulerLRDecay)

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            out = self.network(sample)
            loss = self.criterion(out, label)
            loss.backward()
            self.optimizer.step()
            scheduler.step()

        return


class Adagrad(Trainer):
    def __init__(self, network, epochs=100, schedulerLRDecay=1):
        super().__init__(network)

        self.epochs = epochs
        self.schedulerLRDecay = schedulerLRDecay

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adagrad(self.network.parameters(), lr=0.001)

        return

    def applyUpdate(self, sample, label):
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=self.schedulerLRDecay)

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            out = self.network(sample)
            loss = self.criterion(out, label)
            loss.backward()
            self.optimizer.step()
            scheduler.step()

        return


class Adam(Trainer):
    def __init__(self, network, epochs=100, schedulerLRDecay=1):
        super().__init__(network)

        self.epochs = epochs
        self.schedulerLRDecay = schedulerLRDecay

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)

        return

    def applyUpdate(self, sample, label):
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=self.schedulerLRDecay)

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            out = self.network(sample)
            loss = self.criterion(out, label)
            loss.backward()
            self.optimizer.step()
            scheduler.step()

        return
