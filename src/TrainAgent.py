import io
import pandas as pd
import numpy as np
import torch

# from SteeringPair import Network, DQN, DoubleDQN, REINFORCE, QActorCritic, RANDOM, A2C, A2C_noBoot, A2C_noBoot_v2
# from SteeringPair.Environment import initEnvironment

# from SteeringPair_Continuous import Network, REINFORCE
# from SteeringPair_Continuous.Environment import initEnvironment

from SteeringPair_Stochastic import Network, REINFORCE, REINFORCE_runningNorm, DQN, A2C_noBoot_v2
from SteeringPair_Stochastic.Environment import initEnvironment

# from QuadLens import REINFORCE, A2C, A2C_noBoot, A2C_noBoot_v2, Network, initEnvironment

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
    Algorithm = None
    QNetwork = Network.FC7
    PolicyNetwork = Network.Cat3

    # configure environment
    envConfig = {"stateDefinition": "6d-norm", "actionSet": "A9", "rewardFunction": "stochasticPropRewardStepPenalty",
                 "acceptance": 5e-3, "targetDiameter": 3e-2, "maxIllegalStateCount": 0, "maxStepsPerEpisode": 50,
                 "stateNoiseAmplitude": 1e-13, "rewardNoiseAmplitude": 1, "successBounty": 10,
                 "failurePenalty": -10, "device": device}
    # initEnvironment(**envConfig)

    # define hyper parameters
    hyperParams = {"BATCH_SIZE": 128, "GAMMA": None, "TARGET_UPDATE": None, "EPS_START": 0.5, "EPS_END": 0,
                   "EPS_DECAY": 500, "MEMORY_SIZE": int(1e4)}

    # # train agents
    # trainKWargs = {"trainEpisodes": int(2.5e3), "numberAgents": 20, "meanSamples": 10}
    # trainAgent(**{"envConfig": envConfig, "hyperParams": hyperParams, **trainKWargs})

    # loop over training configurations
    trainKWargs = {"trainEpisodes": int(2.5e3), "numberAgents": 20, "meanSamples": 10}

    algorithms = [(0.999, REINFORCE, 10), (0.999, A2C_noBoot_v2, 0.1), (0.9, DQN, 10), ]

    for alg in algorithms:
        Algorithm = alg[1]
        hyperParams["GAMMA"] = alg[0]
        hyperParams["TARGET_UPDATE"] = alg[2]

        initEnvironment(**envConfig)
        trainAgent(**{"envConfig": envConfig, "hyperParams": hyperParams, **trainKWargs})







