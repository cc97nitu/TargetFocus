"""
Create a benchmark for an agent who chooses every action with the same probability.
This is used as a baseline to assess a trained agent's performance.
"""

import pickle
import io
import pandas as pd
import numpy as np
import torch

from SteeringPair import Network
from SteeringPair import RANDOM
from SteeringPair.Environment import initEnvironment

import SQL

# fetch pre-trained agents
agents_id = 18
trainResults = SQL.retrieve(row_id=agents_id)
agents = trainResults["agents"]

# number of episodes during benchmark
benchEpisodes = 100

# arguments for SQL.insertBenchmark
data = {"agents_id": agents_id, "algorithm": trainResults["algorithm"], "bench_episodes": benchEpisodes,}

# choose algorithm
Algorithm = RANDOM
QNetwork = Network.FC7
PolicyNetwork = Network.Cat3

# environment config
envConfig = {"stateDefinition": "6d-norm", "actionSet": "A4", "rewardFunction": "propReward",
             "acceptance": 5e-3, "targetDiameter": 3e-2, "maxStepsPerEpisode": 50, "successBounty": 10,
             "failurePenalty": -10, "device": "cuda" if torch.cuda.is_available() else "cpu"}
initEnvironment(**envConfig)

# define dummy hyper parameters in order to create trainer-objects for benching
hypPara_RandomBehavior = {"BATCH_SIZE": None, "GAMMA": None, "TARGET_UPDATE": None, "EPS_START": 1, "EPS_END": 1,
                          "EPS_DECAY": 1, "MEMORY_SIZE": None}

dummyOptimizer = torch.optim.Adam
dummyStepSize = 1e-3

# save mean returns whereas each entry is the average over the last meanSamples returns
returns = list()
meanReturns = list()
terminations = {"successful": list(), "failed": list(), "aborted": list()}
meanSamples = 10

# run simulation with random behavior
for i in range(len(agents)):
    print("random run {}".format(i))
    dummyModel = Algorithm.Model(QNetwork=QNetwork, PolicyNetwork=PolicyNetwork)
    dummyModel.eval()
    trainer = Algorithm.Trainer(dummyModel, None, None, **hypPara_RandomBehavior)
    episodeReturns, episodeTerminations = trainer.benchAgent(50)
    episodeReturns = [x[0].item() for x in episodeReturns]

    # mean over last meanSamples episodes
    mean = list()
    for j in range(meanSamples, len(episodeReturns)):
        mean.append(np.mean(episodeReturns[j - meanSamples:j + 1]))

    meanReturns.append(pd.DataFrame({"episode": [i + 1 for i in range(meanSamples, len(episodeReturns))],
                                     "behavior": ["random" for i in range(meanSamples, len(episodeReturns))],
                                     "return": mean}))

    returns.append(pd.DataFrame({"episode": [i + 1 for i in range(len(episodeReturns))],
                                 "behavior": ["random" for i in range(len(episodeReturns))],
                                 "return": episodeReturns}))

    # log how episodes ended
    terminations["successful"].append(episodeTerminations["successful"])
    terminations["failed"].append(episodeTerminations["failed"])
    terminations["aborted"].append(episodeTerminations["aborted"])

# concat to pandas data frame
returns = pd.concat(returns)
meanReturns = pd.concat(meanReturns)

overallResults = {"return": returns, "meanReturn": meanReturns, "terminations": terminations}

# dump
buffer = io.BytesIO()
pickle.dump(overallResults, buffer)
data["benchBlob"]  = buffer.getvalue()

SQL.insertBenchmark(**data)
