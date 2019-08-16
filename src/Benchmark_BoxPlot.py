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

    plt.show()
    plt.close()


def buildPdFrame(data: dict, agent_ident: str):
    # init dict from which pd frame will be initialized from
    frame = dict()
    frame["termination_count"] = list()
    frame["termination_type"] = list()
    frame["agent_ident"] = list()

    # fill frame with data
    for key in data["greedyTerminations"].keys():
        frame["termination_count"] += data["greedyTerminations"][key]
        frame["termination_type"] += [str(key) for i in range(len(data["greedyTerminations"][key]))]
        frame["agent_ident"] += [str(agent_ident) for i in range(len(data["greedyTerminations"][key]))]

    for key in data["randomTerminations"].keys():
        frame["termination_count"] += data["randomTerminations"][key]
        frame["termination_type"] += [str(key) for i in range(len(data["randomTerminations"][key]))]
        frame["agent_ident"] += ["random" for i in range(len(data["randomTerminations"][key]))]

    return pd.DataFrame.from_dict(frame)



if __name__ == "__main__":
    ### benchmark ###
    # # fetch data
    path = "/home/conrad/RL/TempDiff/TargetFocus/src/dump/REINFORCE/6d-states-normalized/6d-norm_4A_RR_Cat1_2000_benchmark"
    with open(path, "rb") as file:
        data = pickle.load(file)

    frame_4a = buildPdFrame(data, "4_actions")

    path = "/home/conrad/RL/TempDiff/TargetFocus/src/dump/REINFORCE/6d-states-normalized/6d-norm_9A_RR_Cat1_2000_benchmark"
    with open(path, "rb") as file:
        data = pickle.load(file)

    frame_9a = buildPdFrame(data, "9_actions")

    path = "/home/conrad/RL/TempDiff/TargetFocus/src/dump/REINFORCE/6d-states-normalized/6d-norm_25A_RR_Cat1_2000_benchmark"
    with open(path, "rb") as file:
        data = pickle.load(file)

    frame_25a = buildPdFrame(data, "25_actions")

    plotStatistics(pd.concat([frame_4a, frame_9a, frame_25a]))



