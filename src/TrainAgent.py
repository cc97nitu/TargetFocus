import pandas as pd
import numpy as np
import torch

import SteeringPair

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define hyper parameters
hyperParams = {"BATCH_SIZE": 128, "GAMMA": 0.0, "TARGET_UPDATE": 10, "EPS_START": 0.5, "EPS_END": 0,
               "EPS_DECAY": 500, "MEMORY_SIZE": int(1e4)}

### train 20 agents and store the corresponding models in agents
agents = dict()
returns = list()
trainEpisodes = 400

meanSamples = 10

for i in range(20):
    print("training agent number {}".format(i))
    model = SteeringPair.DQN.Model()

    trainer = SteeringPair.DQN.Trainer(model, **hyperParams)
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
torch.save({"hyperParameters": hyperParams, "trainEpisodes": trainEpisodes, "agents": agents, "returns": returns}, "/dev/shm/agents.tar")
