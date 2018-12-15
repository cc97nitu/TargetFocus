import torch
import pickle
import pandas as pd

from itertools import product

import Trainer
import Network

# fetch the data
with open("../../dump/supervisedLearning/12.14.18/supervisedTrainingSet", 'rb') as file:
    data = pickle.load(file)

trainInput = data[0]
trainLabels = data[1]
valInput = data[2]
valLabels = data[3]

# initialize
attempts = 40
epochs = 20
criterion = torch.nn.MSELoss()

# choose networks and trainers
networks = (Network.FulCon10,)
trainers = (Trainer.Adam, Trainer.Adamax, Trainer.Adagrad, Trainer.SGD, Trainer.ASGD, Trainer.RMSprop, Trainer.Rprop, Trainer.Adadelta, Trainer.LBFGS)

# try to train
loss = []

for network, trainer in product(networks, trainers):
    print("network={}, trainer={}".format(network, trainer))

    for attempt in range(attempts):
        # initialize
        net = network()
        train = trainer(net, epochs=1)

        # loss before
        loss.append(pd.DataFrame({"set": "training", "epoch": 0, "loss": criterion(net(trainInput), trainLabels).item(),
                                  "Optimizer": "{}".format(train), "network": "{}".format(net)}, index=[0]))
        loss.append(pd.DataFrame({"set": "validation", "epoch": 0, "loss": criterion(net(valInput), valLabels).item(),
                                  "Optimizer": "{}".format(train), "network": "{}".format(net)}, index=[0]))

        for epoch in range(1, epochs + 1):
            # train
            train.applyUpdate(trainInput, trainLabels)

            # loss after latest epoch
            loss.append(
                pd.DataFrame({"set": "training", "epoch": epoch, "loss": criterion(net(trainInput), trainLabels).item(),
                              "Optimizer": "{}".format(train), "network": "{}".format(net)}, index=[0]))
            loss.append(pd.DataFrame(
                {"set": "validation", "epoch": epoch, "loss": criterion(net(valInput), valLabels).item(),
                 "Optimizer": "{}".format(train),
                 "network": "{}".format(net)}, index=[0]))

loss = pd.concat(loss)

# dump results
with open("../../dump/supervisedLearning/12.14.18/trainResults", 'wb') as file:
    pickle.dump(loss, file)
