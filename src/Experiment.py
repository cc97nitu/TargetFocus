import pickle
import io
import pandas as pd
import numpy as np
import torch

from SteeringPair import Network
from SteeringPair import DQN, REINFORCE, QActorCritic, RANDOM
from SteeringPair.Environment import initEnvironment

import SQL

# fetch pre-trained agents
agents_id = 10
trainResults = SQL.retrieve(row_id=agents_id)
agents = trainResults["agents"]

# number of episodes during benchmark
benchEpisodes = 100

# arguments for SQL.insertBenchmark
data = {"agents_id": agents_id, "algorithm": trainResults["algorithm"], "bench_episodes": benchEpisodes,}

# choose algorithm
Algorithm = REINFORCE
QNetwork = Network.FC7
PolicyNetwork = Network.Cat3

# environment config
envConfig = {"stateDefinition": "6d-norm", "actionSet": "A4", "rewardFunction": "propReward",
             "acceptance": 5e-3, "targetDiameter": 3e-2, "maxStepsPerEpisode": 50, "successBounty": 10,
             "failurePenalty": -10, "device": "cuda" if torch.cuda.is_available() else "cpu"}
initEnvironment(**envConfig)

# define dummy hyper parameters in order to create trainer-objects for benching
hypPara_GreedyBehavior = {"BATCH_SIZE": None, "GAMMA": None, "TARGET_UPDATE": None, "EPS_START": 0, "EPS_END": 0,
                          "EPS_DECAY": 1, "MEMORY_SIZE": None}

dummyOptimizer = torch.optim.Adam
dummyStepSize = 1e-3

# save mean returns whereas each entry is the average over the last meanSamples returns
returns = list()
meanReturns = list()
terminations = {"successful": list(), "failed": list(), "aborted": list()}
meanSamples = 10

# run simulation with greedy behavior
for agent in agents:
    print("greedy run {}".format(agent))
    model = Algorithm.Model(QNetwork=QNetwork, PolicyNetwork=PolicyNetwork)
    model.load_state_dict(agents[agent])
    model.eval()
    trainer = Algorithm.Trainer(model, dummyOptimizer, dummyStepSize, **hypPara_GreedyBehavior)
    episodeReturns, episodeTerminations = trainer.benchAgent(benchEpisodes)
    episodeReturns = [x[0].item() for x in episodeReturns]

    # mean over last meanSamples episodes
    mean = list()
    for j in range(meanSamples, len(episodeReturns)):
        mean.append(np.mean(episodeReturns[j - meanSamples:j + 1]))

    meanReturns.append(pd.DataFrame({"episode": [i + 1 for i in range(meanSamples, len(episodeReturns))],
                                     "behavior": ["greedy" for i in range(meanSamples, len(episodeReturns))],
                                     "return": mean}))

    returns.append(pd.DataFrame({"episode": [i + 1 for i in range(len(episodeReturns))],
                                 "behavior": ["greedy" for i in range(len(episodeReturns))],
                                 "return": episodeReturns}))

    # log how episodes ended
    terminations["successful"].append(episodeTerminations["successful"])
    terminations["failed"].append(episodeTerminations["failed"])
    terminations["aborted"].append(episodeTerminations["aborted"])


# concat to pandas data frame
returns = pd.concat(returns)
meanReturns = pd.concat(meanReturns)

overallResults = {"return": returns, "meanReturn": meanReturns, "terminations": terminations, }

# dump
buffer = io.BytesIO()
pickle.dump(overallResults, buffer)
data["benchBlob"]  = buffer.getvalue()

SQL.insertBenchmark(**data)
