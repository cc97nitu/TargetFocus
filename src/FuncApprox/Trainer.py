import abc
import torch


# use CUDA
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Trainer(object, metaclass=abc.ABCMeta):
    """abstract base class for Trainer"""

    @abc.abstractmethod
    def __init__(self, network):
        self.network = network

    @abc.abstractmethod
    def applyUpdate(self, sample, label):
        """train the approximator"""
        pass


class SGD(Trainer):
    def __init__(self, network, epochs=20):
        super().__init__(network)

        self.epochs = epochs

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

    def __repr__(self):
        return "SGD"


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


class RMSprop(Trainer):
    def __init__(self, network, epochs=100, schedulerLRDecay=1):
        super().__init__(network)

        self.epochs = epochs
        self.schedulerLRDecay = schedulerLRDecay

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.RMSprop(self.network.parameters(), lr=0.001)

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

    def __repr__(self):
        return "RMSprop"


class ASGD(Trainer):
    def __init__(self, network, epochs=100, schedulerLRDecay=1):
        super().__init__(network)

        self.epochs = epochs
        self.schedulerLRDecay = schedulerLRDecay

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.ASGD(self.network.parameters(), lr=0.001)

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

    def __repr__(self):
        return "ASGD"


class Adadelta(Trainer):
    def __init__(self, network, epochs=100, schedulerLRDecay=1):
        super().__init__(network)

        self.epochs = epochs
        self.schedulerLRDecay = schedulerLRDecay

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adadelta(self.network.parameters(), lr=0.001)

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

    def __repr__(self):
        return "Adadelta"


class Rprop(Trainer):
    def __init__(self, network, epochs=100, schedulerLRDecay=1):
        super().__init__(network)

        self.epochs = epochs
        self.schedulerLRDecay = schedulerLRDecay

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Rprop(self.network.parameters(), lr=0.001)

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

    def __repr__(self):
        return "Rprop"


class Adamax(Trainer):
    def __init__(self, network, epochs=100, schedulerLRDecay=1):
        super().__init__(network)

        self.epochs = epochs
        self.schedulerLRDecay = schedulerLRDecay

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adamax(self.network.parameters(), lr=0.001)

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

    def __repr__(self):
        return "Adamax"


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

    def __repr__(self):
        return "Adagrad"


class Adam(Trainer):
    def __init__(self, network, epochs=20, schedulerLRDecay=1):
        super().__init__(network)

        self.epochs = epochs
        self.schedulerLRDecay = schedulerLRDecay

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)

        return

    def applyUpdate(self, sample, label):
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=self.schedulerLRDecay)

        # move to gpu
        #        self.network = self.network.to(device)
        #        sample = sample.to(device)
        #        label = label.to(device)

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            out = self.network(sample)
            loss = self.criterion(out, label)
            loss.backward()
            self.optimizer.step()
            scheduler.step()

        # move back to cpu
        #        self.network = self.network.to("cpu")

        return

    def __repr__(self):
        return "Adam"
