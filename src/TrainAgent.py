import os
import torch

import DQN

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define hyper parameters
hyperParams = {"BATCH_SIZE": 128, "GAMMA": 0.999, "TARGET_UPDATE": 30, "EPS_START": 0.5, "EPS_END": 0,
               "EPS_DECAY": 500, "MEMORY_SIZE": int(1e4)}

### train 20 agents and store the corresponding models in agents
agents = dict()
trainEpisodes = 200

for i in range(20):
    print("training agent number {}".format(i))
    model = DQN.Model()

    trainer = DQN.Trainer(model, **hyperParams)
    trainer.trainAgent(trainEpisodes)

    agents["agent_{}".format(i)] = model.to_dict()


### save the trained agents to disk
torch.save({"hyperParameters": hyperParams, "trainEpisodes": trainEpisodes, "agents": agents}, "/home/dylan/RL/TempDiff/TargetFocus/src/dump/agents.tar")
