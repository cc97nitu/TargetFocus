import io
import pandas as pd
import numpy as np
import torch

from SteeringPair import Network
from SteeringPair import DQN, REINFORCE, QActorCritic
from SteeringPair.Environment import initEnvironment

import SQL

# choose algorithm
Algorithm = REINFORCE
QNetwork = Network.FC8
PolicyNetwork = Network.Cat3

# choose optimizer
optimizer = torch.optim.Adam
stepSize = 3e-4

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# configure environment
envConfig = {"stateDefinition": "6d-norm", "actionSet": "A9", "rewardFunction": "propRewardStepPenalty",
             "acceptance": 5e-3, "targetDiameter": 3e-2, "maxStepsPerEpisode": 50, "successBounty": 10,
             "failurePenalty": -10, "device": device}
initEnvironment(**envConfig)

# define hyper parameters
hyperParams = {"BATCH_SIZE": 128, "GAMMA": 0.0, "TARGET_UPDATE": 10, "EPS_START": 0.5, "EPS_END": 0,
               "EPS_DECAY": 500, "MEMORY_SIZE": int(1e4)}

### train 20 agents and store the corresponding models in agents
agents = dict()
returns = list()
trainEpisodes = 200

meanSamples = 10

for i in range(20):
    print("training agent number {}".format(i))
    model = Algorithm.Model(QNetwork=QNetwork, PolicyNetwork=PolicyNetwork)

    trainer = Algorithm.Trainer(model, optimizer, stepSize, **hyperParams)
    episodeReturns, _ = trainer.trainAgent(trainEpisodes)
    episodeReturns = [x[0].item() for x in episodeReturns]

    # mean over last meanSamples episodes
    mean = list()
    for j in range(meanSamples, len(episodeReturns)):
        mean.append(np.mean(episodeReturns[j - meanSamples:j + 1]))

    returns.append(pd.DataFrame({"episode": [i + 1 for i in range(meanSamples, len(episodeReturns))],
                                 "behavior": ["random" for i in range(meanSamples, len(episodeReturns))],
                                 "return": mean}))

    agents["agent_{}".format(i)] = model.to_dict()

# merge data frames
returns = pd.concat(returns)

### save the trained agents to disk
envConfig["device"] = str(envConfig["device"].type)

# dump into file
torch.save({"environmentConfig": envConfig, "hyperParameters": hyperParams, "algorithm": Algorithm.__name__, "network": trainer.model,
            "optimizer": optimizer.__name__, "stepSize": stepSize,
            "trainEpisodes": trainEpisodes, "agents": agents, "returns": returns},
           "/dev/shm/agents.tar")

# dump into SQL
buffer = io.BytesIO()
torch.save({"environmentConfig": envConfig, "hyperParameters": hyperParams, "algorithm": Algorithm.__name__, "network": trainer.model,
            "optimizer": optimizer.__name__, "stepSize": stepSize,
            "trainEpisodes": trainEpisodes, "agents": agents, "returns": returns}, buffer)

columnData = {**envConfig, **hyperParams, "algorithm": Algorithm.__name__, "network": str(trainer.model), "optimizer": optimizer.__name__,
              "stepSize": stepSize, "trainEpisodes": trainEpisodes}

SQL.insert(columnData, buffer.getvalue())