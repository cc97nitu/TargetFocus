"""
Visualize performance during benchmark.
"""
import pickle

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def plotStatistics(data):
    # plot
    sns.set(style="whitegrid")

    sns.boxplot(x="agent_ident", y="termination_count", hue="termination_type", data=data, palette="Set2")

    plt.xlabel(None)
    plt.ylabel("episodes")

    plt.show()
    plt.close()


def buildPdFrame(*args):
    # init dict from which pd frame will be initialized from
    frame = dict()
    frame["termination_count"] = list()
    frame["termination_type"] = list()
    frame["agent_ident"] = list()

    # fill frame with data
    for arg in args:
        data, agent_ident = arg[0], arg[1]

        for key in data["greedyTerminations"].keys():
            frame["termination_count"] += data["greedyTerminations"][key]
            frame["termination_type"] += [str(key) for i in range(len(data["greedyTerminations"][key]))]
            frame["agent_ident"] += [str(agent_ident) for i in range(len(data["greedyTerminations"][key]))]

    for arg in args:
        data, agent_ident = arg[0], arg[1]

        for key in data["randomTerminations"].keys():
            frame["termination_count"] += data["randomTerminations"][key]
            frame["termination_type"] += [str(key) for i in range(len(data["randomTerminations"][key]))]
            frame["agent_ident"] += ["random" for i in range(len(data["randomTerminations"][key]))]

    return pd.DataFrame.from_dict(frame)



if __name__ == "__main__":
    ### benchmark ###
    # # fetch data
    # with open("/home/conrad/RL/TempDiff/TargetFocus/src/dump/DQN/Adam(lr=2e-5)/propReward/gamma=0/6d-norm_4A_RR_FC7_2000_benchmark", "rb") as file:
    #     data = pickle.load(file)
    # data = buildPdFrame((data, "agent_0"))
    # plotStatistics(data,)

    path = "/home/conrad/RL/TempDiff/TargetFocus/src/dump/REINFORCE/6d-states-normalized/propReward/6d-norm_9A_RR_Cat3_propReward_2000_benchmark"
    with open(path, "rb") as file:
        frameA = pickle.load(file)

    path = "/home/conrad/RL/TempDiff/TargetFocus/src/dump/REINFORCE/6d-states-normalized/propRewardStepPenalty/6d-norm_9A_RR_Cat3_propRewardStepPenalty_2000_benchmark"
    with open(path, "rb") as file:
        frameB = pickle.load(file)

    path = "/home/conrad/RL/TempDiff/TargetFocus/src/dump/REINFORCE/6d-states-normalized/propRewardStepPenalty/6d-norm_9A_RR_Cat3_propRewardStepPenalty_2000_benchmark"
    with open(path, "rb") as file:
        frameC = pickle.load(file)


    args = [(frameA, "A"), (frameB, "B"), (frameC, "C")]

    plotStatistics(buildPdFrame(*args))



