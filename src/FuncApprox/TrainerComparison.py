import sys
sys.path.append("../")

import torch
import pickle
import pandas as pd

from itertools import product

import Trainer
import Network
from Agent import Agent
from QValue import QNeural

# fetch the data
with open("../../dump/supervisedLearning/12.14.18/supervisedTrainingSet", 'rb') as file:
    data = pickle.load(file)

trainInput = data[0]
trainLabels = data[1]
valInput = data[2]
valLabels = data[3]
trainRecommendedActions = data[4]
valRecommendedActions = data[5]

# initialize
attempts = 4
epochs = 2
criterion = torch.nn.MSELoss()

# choose networks and trainers
networks = (Network.FulCon10,)
trainers = (Trainer.Adam,)
# trainers = (Trainer.Adam, Trainer.Adamax, Trainer.Adagrad, Trainer.SGD, Trainer.ASGD, Trainer.RMSprop, Trainer.Rprop, Trainer.Adadelta, Trainer.LBFGS)

# try to train
resultFrame = []
agent = Agent(QNeural())

for network, trainer in product(networks, trainers):
    print("network={}, trainer={}".format(network, trainer))

    for attempt in range(attempts):
        # initialize
        net = network()
        train = trainer(net, epochs=1)

        # loss before
        resultFrame.append(pd.DataFrame({"set": "training", "epoch": 0, "loss": criterion(net(trainInput), trainLabels).item(),
                                  "Optimizer": "{}".format(train), "network": "{}".format(net)}, index=[0]))
        resultFrame.append(pd.DataFrame({"set": "validation", "epoch": 0, "loss": criterion(net(valInput), valLabels).item(),
                                  "Optimizer": "{}".format(train), "network": "{}".format(net)}, index=[0]))

        # get recommended actions from current predictions and updates
        agent.q.network = net

        trainHit, valHit = 0, 0

        for i in range(len(trainInput)):
            recAction = agent.bestAction(trainInput[i], isTensor=True)
            if torch.allclose(recAction, trainRecommendedActions[i]):
                trainHit += 1

        for i in range(len(valInput)):
            recAction = agent.bestAction(valInput[i], isTensor=True)
            if torch.allclose(recAction, valRecommendedActions[i]):
                valHit += 1

        print("accuracy on training data: {0}, validation data: {1}".format(trainHit / len(trainInput), valHit / len(valInput)))

        # train for epochs
        for epoch in range(1, epochs + 1):
            # train
            train.applyUpdate(trainInput, trainLabels)

            # loss after latest epoch
            resultFrame.append(
                pd.DataFrame({"set": "training", "epoch": epoch, "loss": criterion(net(trainInput), trainLabels).item(),
                              "Optimizer": "{}".format(train), "network": "{}".format(net)}, index=[0]))
            resultFrame.append(pd.DataFrame(
                {"set": "validation", "epoch": epoch, "loss": criterion(net(valInput), valLabels).item(),
                 "Optimizer": "{}".format(train),
                 "network": "{}".format(net)}, index=[0]))

resultFrame = pd.concat(resultFrame)

# # dump results
# with open("../../dump/supervisedLearning/12.14.18/trainResults", 'wb') as file:
#     pickle.dump(resultFrame, file)
