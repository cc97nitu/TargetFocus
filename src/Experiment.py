import pickle
import pandas as pd
import numpy as np
import torch

import DQN

# define dummy hyper parameters in order to create trainer-objects for benching
hypPara_RandomBehavior = {"BATCH_SIZE": None, "GAMMA": None, "TARGET_UPDATE": None, "EPS_START": 1, "EPS_END": 1,
                          "EPS_DECAY": 1, "MEMORY_SIZE": None}

hypPara_GreedyBehavior = {"BATCH_SIZE": None, "GAMMA": None, "TARGET_UPDATE": None, "EPS_START": 0, "EPS_END": 0,
                          "EPS_DECAY": 1, "MEMORY_SIZE": None}

# fetch pre-trained agents
trainResults = torch.load("/home/dylan/RL/TempDiff/TargetFocus/src/dump/agents.tar")
agents = trainResults["agents"]

# save mean returns whereas each entry is the average over the last meanSamples returns
returns = list()
meanSamples = 10

# run simulation with greedy behavior
for agent in agents:
    print("greedy run {}".format(agent))
    model = DQN.Model()
    model.load_state_dict(agents[agent])
    trainer = DQN.Trainer(model, **hypPara_GreedyBehavior)
    episodeReturns = trainer.benchAgent(50)
    episodeReturns = [x[0].item() for x in episodeReturns]

    # mean over last meanSamples episodes
    mean = list()
    for j in range(meanSamples, len(episodeReturns)):
        mean.append(np.mean(episodeReturns[j - meanSamples:j + 1]))

    returns.append(pd.DataFrame({"episode": [i + 1 for i in range(meanSamples, len(episodeReturns))],
                                 "behavior": ["greedy" for i in range(meanSamples, len(episodeReturns))],
                                 "meanReturn": mean}))

# run simulation with random behavior
for i in range(len(agents)):
    print("random run {}".format(i))
    dummyModel = DQN.Model()
    trainer = DQN.Trainer(dummyModel, **hypPara_RandomBehavior)
    episodeReturns = trainer.benchAgent(50)
    episodeReturns = [x[0].item() for x in episodeReturns]

    # mean over last meanSamples episodes
    mean = list()
    for j in range(meanSamples, len(episodeReturns)):
        mean.append(np.mean(episodeReturns[j - meanSamples:j + 1]))

    returns.append(pd.DataFrame({"episode": [i + 1 for i in range(meanSamples, len(episodeReturns))],
                                 "behavior": ["random" for i in range(meanSamples, len(episodeReturns))],
                                 "meanReturn": mean}))

# concat to pandas data frame
returns = pd.concat(returns)

# dump
with open("/home/dylan/RL/TempDiff/TargetFocus/src/dump/benchmark", "wb") as file:
    pickle.dump(returns, file)
