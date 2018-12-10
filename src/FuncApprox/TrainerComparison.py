import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np

from itertools import product

import Trainer
import Network

# fetch the data
with open("/home/dylan/RL/TempDiff/TargetFocus/dump/supervisedTrainingSet", 'rb') as file:
    data = pickle.load(file)

trainInput = data[0]
trainLabels = data[1]
valInput = data[2]
valLabels = data[3]

# initialize
epochs = 20
criterion = torch.nn.MSELoss()

# choose networks and trainers
networks = (Network.FulCon10,)
trainers = (Trainer.Adam,)

# try to train
trainLoss = np.empty(epochs + 1)
valLoss = np.empty(epochs + 1)

for network, trainer in product(networks, trainers):
    # initialize
    net = network()
    train = trainer(net, epochs=1)

    # loss before
    trainLoss[0] = criterion(net(trainInput), trainLabels)
    valLoss[0] = criterion(net(valInput), valLabels)

    for epoch in range(1, epochs + 1):
        # train
        train.applyUpdate(trainInput, trainLabels)

        # loss after
        trainLoss[epoch] = criterion(net(trainInput), trainLabels)
        valLoss[epoch] = criterion(net(valInput), valLabels)

# visualize
fig, ax = plt.subplots()
ax.plot(trainLoss, label='trainingData')
ax.plot(valLoss, label='validationData')

plt.legend()
plt.show()
plt.close()
