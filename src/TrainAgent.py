import pandas as pd
import numpy as np
import torch

from SteeringPair import Network
from SteeringPair import DQN, REINFORCE, QActorCritic
from SteeringPair.Environment import initEnvironment

# choose algorithm
Algorithm = DQN
QNetwork = Network.FC7
PolicyNetwork = Network.Cat1

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# configure environment
envConfig = {"stateDefinition": "6d-norm", "actionSet": "A4", "rewardFunction": "propReward",
             "acceptance": 5e-3, "targetDiameter": 3e-2, "maxStepsPerEpisode": 50, "successBounty": 10,
             "failurePenalty": -10, "device": device}
initEnvironment(**envConfig)

# define hyper parameters
hyperParams = {"BATCH_SIZE": 128, "GAMMA": 0.0, "TARGET_UPDATE": 10, "EPS_START": 0.5, "EPS_END": 0,
               "EPS_DECAY": 500, "MEMORY_SIZE": int(1e4)}

### train 20 agents and store the corresponding models in agents
agents = dict()
returns = list()
trainEpisodes = 30

meanSamples = 10

for i in range(1):
    print("training agent number {}".format(i))
    model = Algorithm.Model(QNetwork=QNetwork, PolicyNetwork=PolicyNetwork)

    trainer = Algorithm.Trainer(model, **hyperParams)
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
torch.save({"environmentConfig": envConfig, "hyperParameters": hyperParams, "network": trainer.model, "trainEpisodes": trainEpisodes, "agents": agents, "returns": returns}, "/dev/shm/agents.tar")
