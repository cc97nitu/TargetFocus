import pickle
import pandas as pd
import numpy as np
import torch

from SteeringPair import Network
from SteeringPair import DQN, REINFORCE, QActorCritic
from SteeringPair.Environment import initEnvironment

# choose algorithm
Algorithm = REINFORCE
QNetwork = Network.FC7
PolicyNetwork = Network.Cat3

# environment config
envConfig = {"stateDefinition": "6d-norm", "actionSet": "A9", "rewardFunction": "propReward",
             "acceptance": 5e-3, "targetDiameter": 3e-2, "maxStepsPerEpisode": 50, "successBounty": 10,
             "failurePenalty": -10, "device": "cuda" if torch.cuda.is_available() else "cpu"}
initEnvironment(**envConfig)

# define dummy hyper parameters in order to create trainer-objects for benching
hypPara_RandomBehavior = {"BATCH_SIZE": None, "GAMMA": None, "TARGET_UPDATE": None, "EPS_START": 1, "EPS_END": 1,
                          "EPS_DECAY": 1, "MEMORY_SIZE": None}

hypPara_GreedyBehavior = {"BATCH_SIZE": None, "GAMMA": None, "TARGET_UPDATE": None, "EPS_START": 0, "EPS_END": 0,
                          "EPS_DECAY": 1, "MEMORY_SIZE": None}

# fetch pre-trained agents
trainResults = torch.load("/home/conrad/RL/TempDiff/TargetFocus/src/dump/REINFORCE/6d-states-normalized/ConstantRewardPerStep/6d-norm_9A_RR_Cat3_constantRewardPerStep_2000_agents.tar")
agents = trainResults["agents"]

# save mean returns whereas each entry is the average over the last meanSamples returns
returns = list()
meanReturns = list()
greedyTerminations, randomTerminations = {"successful": list(), "failed": list(), "aborted": list()}, {
    "successful": list(), "failed": list(), "aborted": list()}
meanSamples = 10

# run simulation with greedy behavior
for agent in agents:
    print("greedy run {}".format(agent))
    model = Algorithm.Model(QNetwork=QNetwork, PolicyNetwork=PolicyNetwork)
    model.load_state_dict(agents[agent])
    model.eval()
    trainer = Algorithm.Trainer(model, **hypPara_GreedyBehavior)
    episodeReturns, episodeTerminations = trainer.benchAgent(50)
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
    greedyTerminations["successful"].append(episodeTerminations["successful"])
    greedyTerminations["failed"].append(episodeTerminations["failed"])
    greedyTerminations["aborted"].append(episodeTerminations["aborted"])

# run simulation with random behavior
for i in range(len(agents)):
    print("random run {}".format(i))
    dummyModel = Algorithm.Model(QNetwork=QNetwork, PolicyNetwork=PolicyNetwork)
    dummyModel.eval()
    trainer = Algorithm.Trainer(dummyModel, **hypPara_RandomBehavior)
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
    randomTerminations["successful"].append(episodeTerminations["successful"])
    randomTerminations["failed"].append(episodeTerminations["failed"])
    randomTerminations["aborted"].append(episodeTerminations["aborted"])

# concat to pandas data frame
returns = pd.concat(returns)
meanReturns = pd.concat(meanReturns)

overallResults = {"return": returns, "meanReturn": meanReturns, "greedyTerminations": greedyTerminations,
                  "randomTerminations": randomTerminations}

# dump
with open("/dev/shm/6d-norm_9A_RR_Cat3_constantRewardPerStep_2000_benchmark", "wb") as file:
    pickle.dump(overallResults, file)
