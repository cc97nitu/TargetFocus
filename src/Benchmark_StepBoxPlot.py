"""
Visualize episode length during benchmark.
"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import SQL

def buildPdFrame(*args):
    # init dict from which pd frame will be initialized from
    frame = dict()
    frame["algorithm"] = list()
    frame["steps"] = list()

    # fill frame with data
    for arg in args:
        data, algorithm = arg[0], arg[1]

        print(type(data), data.keys())

        steps = data["return"]["steps"]

        frame["algorithm"] += [str(algorithm) for i in range(len(steps))]
        frame["steps"] += steps.values.tolist()

    return pd.DataFrame.from_dict(frame)


def plotStatistics(data):
    # plot
    sns.set(style="whitegrid")

    sns.boxplot(x="algorithm", y="steps", data=data, palette="Set2")

    # plt.xlabel(None)
    # plt.ylabel("episodes")

    plt.show()
    plt.close()



if __name__ == "__main__":
    # retrieve benchmark
    benchA = SQL.retrieveBenchmark(57)
    benchB = SQL.retrieveBenchmark(58)

    # build dataframe
    data = buildPdFrame((benchA, "REINFORCE"), (benchB, "DQN"))

    # show steps per episode
    plotStatistics(data)