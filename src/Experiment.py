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
trainResults = torch.load("/dev/shm/agents.tar")
agents = trainResults["agents"]

# save mean returns whereas each entry is the average over the last meanSamples returns
returns = list()
greedyTerminations, randomTerminations = {"successful": list(), "failed": list(), "aborted": list()}, {"successful": list(), "failed": list(), "aborted": list()}
meanSamples = 10

# run simulation with greedy behavior
for agent in agents:
    print("greedy run {}".format(agent))
    model = DQN.Model()
    model.load_state_dict(agents[agent])
    model.eval()
    trainer = DQN.Trainer(model, **hypPara_GreedyBehavior)
    episodeReturns, episodeTerminations = trainer.benchAgent(50)
    episodeReturns = [x[0].item() for x in episodeReturns]

    # mean over last meanSamples episodes
    mean = list()
    for j in range(meanSamples, len(episodeReturns)):
        mean.append(np.mean(episodeReturns[j - meanSamples:j + 1]))

    returns.append(pd.DataFrame({"episode": [i + 1 for i in range(len(episodeReturns))],
                                 "behavior": ["greedy" for i in range(len(episodeReturns))],
                                 "return": episodeReturns}))

    # log how episodes ended
    greedyTerminations["successful"].append(episodeTerminations["successful"])
    greedyTerminations["failed"].append(episodeTerminations["failed"])
    greedyTerminations["aborted"].append(episodeTerminations["aborted"])

# run simulation with random behavior
for i in range(len(agents)):
    print("random run {}".format(i))
    dummyModel = DQN.Model()
    dummyModel.eval()
    trainer = DQN.Trainer(dummyModel, **hypPara_RandomBehavior)
    episodeReturns, episodeTerminations = trainer.benchAgent(50)
    episodeReturns = [x[0].item() for x in episodeReturns]

    # mean over last meanSamples episodes
    mean = list()
    for j in range(meanSamples, len(episodeReturns)):
        mean.append(np.mean(episodeReturns[j - meanSamples:j + 1]))

    returns.append(pd.DataFrame({"episode": [i + 1 for i in range(len(episodeReturns))],
                                 "behavior": ["random" for i in range(len(episodeReturns))],
                                 "return": episodeReturns}))

    # log how episodes ended
    randomTerminations["successful"].append(episodeTerminations["successful"])
    randomTerminations["failed"].append(episodeTerminations["failed"])
    randomTerminations["aborted"].append(episodeTerminations["aborted"])


# concat to pandas data frame
returns = pd.concat(returns)

overallResults = {"returns": returns, "greedyTerminations": greedyTerminations, "randomTerminations": randomTerminations}

# dump
with open("/dev/shm/benchmark", "wb") as file:
    pickle.dump(overallResults, file)
