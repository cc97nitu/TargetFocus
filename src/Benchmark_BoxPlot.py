"""
Visualize performance during benchmark.
"""
import pickle

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import SQL


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

        print(type(data), data.keys())

        for key in data["terminations"].keys():
            frame["termination_count"] += data["terminations"][key]
            frame["termination_type"] += [str(key) for i in range(len(data["terminations"][key]))]
            frame["agent_ident"] += [str(agent_ident) for i in range(len(data["terminations"][key]))]

    # for arg in args:
    #     data, agent_ident = arg[0], arg[1]
    #
    #     for key in data["randomTerminations"].keys():
    #         frame["termination_count"] += data["randomTerminations"][key]
    #         frame["termination_type"] += [str(key) for i in range(len(data["randomTerminations"][key]))]
    #         frame["agent_ident"] += ["random" for i in range(len(data["randomTerminations"][key]))]

    return pd.DataFrame.from_dict(frame)



if __name__ == "__main__":

    frameA = SQL.retrieveBenchmark(13)
    frameB = SQL.retrieveBenchmark(14)
    frameC = SQL.retrieveBenchmark(15)
    frameD = SQL.retrieveBenchmark(16)
    frameZ = SQL.retrieveBenchmark(4)

    args = [(frameA, "REINFORCE"), (frameB, "A2C_noBoot"), (frameC, "A2C_noBoot_v2"), (frameD, "A2C"), (frameZ, "random")]

    plotStatistics(buildPdFrame(*args))

    # frameA = SQL.retrieveBenchmark(13)
    # plotStatistics(buildPdFrame((frameA, "test")))


