import pickle
import torch
import numpy as np

from itertools import product

# fetch the agent with filled replay memory
with open("../dump/agent", 'rb') as file:
    agent = pickle.load(file)

# generate training set from replay memory
allInput, allLabels = [], []

for shortMemory in agent.replayMemory:
    netInput, labels = agent.getSarsaLambda(shortMemory)
    allInput.append(netInput)
    allLabels.append(labels)

allInput = torch.cat(allInput)
allLabels = torch.cat(allLabels)

# Creating data indices for training and validation split
validationSplit = .2
shuffleDataset = True
random_seed = 42

dataset_size = len(allInput)
indices = list(range(dataset_size))
split = int(np.floor(validationSplit * dataset_size))
if shuffleDataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
trainIndices, valIndices = indices[split:], indices[:split]

# create new input and labels
trainInput = torch.randn(len(trainIndices), len(allInput[0]))
trainLabels = torch.randn(len(trainIndices), len(allLabels[0]))

valInput = torch.randn(len(valIndices), len(allInput[0]))
valLabels = torch.randn(len(valIndices), len(allLabels[0]))

for i in range(len(trainIndices)):
    trainInput[i] = allInput[trainIndices[i]]
    trainLabels[i] = allLabels[trainIndices[i]]

for i in range(len(valIndices)):
    valInput[i] = allInput[valIndices[i]]
    valLabels[i] = allLabels[valIndices[i]]

# # get recommended actions from current predictions and updates
# trainRecommendedActions, valRecommendedActions = [], []
#
# for i in range(len(trainInput)):
#     recAction, value = agent.bestAction(trainInput[i, :4], isTensor=True)
#
#     if trainLabels[i].item() > value.item():
#         trainRecommendedActions.append(trainInput[i, -2:])
#     else:
#         trainRecommendedActions.append(recAction)
#
# for i in range(len(valInput)):
#     recAction, value = agent.bestAction(valInput[i, :4], isTensor=True)
#
#     if valLabels[i].item() > value.item():
#         valRecommendedActions.append(trainInput[i, -2:])
#     else:
#         valRecommendedActions.append(recAction)

# save to disk
with open("../dump/supervisedTrainingSet", 'wb') as file:
    pickle.dump((trainInput, trainLabels, valInput, valLabels,), file)
    # pickle.dump((trainInput, trainLabels, valInput, valLabels, trainRecommendedActions, valRecommendedActions,), file)
