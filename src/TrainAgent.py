import io
import pandas as pd
import numpy as np
import torch

# from SteeringPair import Network, DQN, DoubleDQN, REINFORCE, QActorCritic, RANDOM, A2C, A2C_noBoot, A2C_noBoot_v2
# from SteeringPair.Environment import initEnvironment

# from SteeringPair_Continuous import Network, REINFORCE
# from SteeringPair_Continuous.Environment import initEnvironment

# from SteeringPair_Stochastic import Network, REINFORCE, REINFORCE_runningNorm, DQN, DQN_runningNorm, A2C_noBoot_v2
# from SteeringPair_Stochastic.Environment import initEnvironment

from QuadLens import RANDOM, REINFORCE, A2C, A2C_noBoot, A2C_noBoot_v2, Network, initEnvironment

import SQL


# choose optimizer
optimizer = torch.optim.Adam
stepSize = 3e-4

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def trainAgent(envConfig, hyperParams, trainEpisodes, numberAgents, meanSamples):
    """train agents and store the corresponding models in agents"""
    agents = dict()
    returns = list()

    for i in range(numberAgents):
        print("training agent {}/{}".format(i + 1, numberAgents))
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
    envConfig["device"] = str(device.type)

    # dump into file
    torch.save({"environmentConfig": envConfig, "hyperParameters": hyperParams, "algorithm": Algorithm.__name__,
                "network": trainer.model,
                "optimizer": optimizer.__name__, "stepSize": stepSize,
                "trainEpisodes": trainEpisodes, "agents": agents, "returns": returns},
               "/dev/shm/agents.tar")

    # dump into SQL
    buffer = io.BytesIO()
    torch.save({"environmentConfig": envConfig, "hyperParameters": hyperParams, "algorithm": Algorithm.__name__,
                "network": trainer.model,
                "optimizer": optimizer.__name__, "stepSize": stepSize,
                "trainEpisodes": trainEpisodes, "agents": agents, "returns": returns}, buffer)

    columnData = {**envConfig, **hyperParams, "algorithm": Algorithm.__name__, "network": str(trainer.model),
                  "optimizer": optimizer.__name__,
                  "stepSize": stepSize, "trainEpisodes": trainEpisodes}

    SQL.insert(columnData, buffer.getvalue())


if __name__ == '__main__':
    # choose algorithm
    Algorithm = RANDOM
    QNetwork = Network.FC7
    PolicyNetwork = Network.Cat3

    # configure environment
    envConfig = {"stateDefinition": "RAW_16", "actionSet": "A9", "rewardFunction": "propRewardStepPenalty",
                 "acceptance": 1e-3, "targetDiameter": 3e-2, "maxIllegalStateCount": 0, "maxStepsPerEpisode": 50, "successBounty": 10,
                 "failurePenalty": -10, "device": "cuda" if torch.cuda.is_available() else "cpu"}
    initEnvironment(**envConfig)

    # define hyper parameters
    hyperParams = {"BATCH_SIZE": 128, "GAMMA": 0.9, "TARGET_UPDATE": 10, "EPS_START": 0.5, "EPS_END": 0,
                   "EPS_DECAY": 500, "MEMORY_SIZE": int(1e4)}

    # train agents
    trainKWargs = {"trainEpisodes": int(2.5e1), "numberAgents": 20, "meanSamples": 10}
    trainAgent(**{"envConfig": envConfig, "hyperParams": hyperParams, **trainKWargs})


