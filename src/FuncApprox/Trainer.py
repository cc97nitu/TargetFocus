import abc
import torch
import torch.utils.data

# use CUDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Trainer(object, metaclass=abc.ABCMeta):
    """abstract base class for Trainer"""

    def __init__(self, network, epochs=20, schedulerLRDecay=1):
        self.network = network
        self.epochs = epochs
        self.criterion = torch.nn.MSELoss()
        self.schedulerLRDecay = schedulerLRDecay

    def applyUpdate(self, sample, label):
        # create dataset and data loader
        trainSet = torch.utils.data.TensorDataset(sample, label)
        trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=32, shuffle=True, num_workers=2)

        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=self.schedulerLRDecay)

        # move to gpu
        self.network = self.network.to(device)

        for epoch in range(self.epochs):
            for sample, label in trainLoader:
                # move to gpu
                sample = sample.to(device)
                label = label.to(device)

                # optimize network
                self.optimizer.zero_grad()
                out = self.network(sample)
                loss = self.criterion(out, label)
                loss.backward()
                self.optimizer.step()
                scheduler.step()

        # move back to cpu
        self.network = self.network.to("cpu")

        return


class SGD(Trainer):
    def __init__(self, network, epochs=100, schedulerLRDecay=1):
        super().__init__(network, epochs, schedulerLRDecay)

        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=1e-3)

        return

    def __repr__(self):
        return "SGD"


class RMSprop(Trainer):
    def __init__(self, network, epochs=100, schedulerLRDecay=1):
        super().__init__(network, epochs, schedulerLRDecay)

        self.optimizer = torch.optim.RMSprop(self.network.parameters())

        return

    def __repr__(self):
        return "RMSprop"


class ASGD(Trainer):
    def __init__(self, network, epochs=100, schedulerLRDecay=1):
        super().__init__(network, epochs, schedulerLRDecay)

        self.optimizer = torch.optim.ASGD(self.network.parameters())

        return

    def __repr__(self):
        return "ASGD"


class Adadelta(Trainer):
    def __init__(self, network, epochs=100, schedulerLRDecay=1):
        super().__init__(network, epochs, schedulerLRDecay)

        self.optimizer = torch.optim.Adadelta(self.network.parameters())

        return

    def __repr__(self):
        return "Adadelta"


class Rprop(Trainer):
    def __init__(self, network, epochs=100, schedulerLRDecay=1):
        super().__init__(network, epochs, schedulerLRDecay)

        self.optimizer = torch.optim.Rprop(self.network.parameters())

        return

    def __repr__(self):
        return "Rprop"


class Adamax(Trainer):
    def __init__(self, network, epochs=100, schedulerLRDecay=1):
        super().__init__(network, epochs, schedulerLRDecay)

        self.optimizer = torch.optim.Adamax(self.network.parameters())

        return

    def __repr__(self):
        return "Adamax"


class Adagrad(Trainer):
    def __init__(self, network, epochs=100, schedulerLRDecay=1):
        super().__init__(network, epochs, schedulerLRDecay)

        self.optimizer = torch.optim.Adagrad(self.network.parameters())

        return

    def __repr__(self):
        return "Adagrad"


class Adam(Trainer):
    def __init__(self, network, epochs=20, schedulerLRDecay=1):
        super().__init__(network, epochs, schedulerLRDecay)

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)

        return

    def __repr__(self):
        return "Adam"


class LBFGS(Trainer):
    def __init__(self, network, epochs=20, schedulerLRDecay=1):
        super().__init__(network, epochs, schedulerLRDecay)

        self.optimizer = torch.optim.LBFGS(self.network.parameters(), lr=0.001)

        self.sample = None
        self.label = None

        return

    def closure(self):
        self.optimizer.zero_grad()
        output = self.network(self.sample)
        loss = self.criterion(output, self.label)
        loss.backward()
        return loss

    def applyUpdate(self, sample, label):
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=self.schedulerLRDecay)

        # move to gpu
        self.network = self.network.to(device)
        self.sample = sample.to(device)
        self.label = label.to(device)

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            out = self.network(self.sample)
            loss = self.criterion(out, self.label)
            loss.backward()
            self.optimizer.step(self.closure)
            scheduler.step()

        # move back to cpu
        self.network = self.network.to("cpu")

        return

    def __repr__(self):
        return "LBFGS"
