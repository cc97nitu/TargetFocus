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
    # with open("/home/conrad/RL/TempDiff/TargetFocus/src/dump/REINFORCE/2d-states-normalized/Cat1/RR_Cat1_400_benchmark", "rb") as file:
    #     data = pickle.load(file)
    # plotStatistics(buildPdFrame(data, "REINFORCE"))

    path = "/home/conrad/RL/TempDiff/TargetFocus/src/dump/DQN/Adam(lr=2e-5)/propReward/gamma=0/6d-norm_4A_RR_FC7_2000_benchmark"
    with open(path, "rb") as file:
        frame_4a = pickle.load(file)

    path = "/home/conrad/RL/TempDiff/TargetFocus/src/dump/DQN/Adam(lr=2e-5)/propReward/gamma=0/6d-norm_9A_RR_FC7_2000_benchmark"
    with open(path, "rb") as file:
        frame_9a = pickle.load(file)

    path = "/home/conrad/RL/TempDiff/TargetFocus/src/dump/DQN/Adam(lr=2e-5)/propReward/gamma=0/6d-norm_25A_RR_FC7_2000_benchmark"
    with open(path, "rb") as file:
        frame_25a = pickle.load(file)

    args = [(frame_4a, r"$\mathit{A}_{4}$"), (frame_9a, r"$\mathit{A}_{9}$"), (frame_25a, r"$\mathit{A}_{25}$")]

    plotStatistics(buildPdFrame(*args))



