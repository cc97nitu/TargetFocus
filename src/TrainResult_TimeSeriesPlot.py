"""
Visualize mean returns achieved during training.
"""
import torch

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def plotTrainResults(data: pd.DataFrame):
    # plot
    sns.set(style="darkgrid")

    sns.lineplot(x="episode", y="return", hue="type", data=data)

    plt.show()
    plt.close()


if __name__ == "__main__":
    ### train results ###
    # fetch data
    # data = torch.load("/home/conrad/RL/TempDiff/TargetFocus/src/dump/randomStart_randomGoal_400_agents.tar")

    # plotTrainResults(data["return"])

    # concat two data frames
    dataFixedFixed = torch.load("/home/conrad/RL/TempDiff/TargetFocus/src/dump/fixedStart_fixedGoal_agents.tar")
    dataRandomFixed = torch.load("/home/conrad/RL/TempDiff/TargetFocus/src/dump/randomStart_fixedGoal_agents.tar")
    dataRandomRandom = torch.load("/home/conrad/RL/TempDiff/TargetFocus/src/dump/randomStart_randomGoal_agents.tar")

    dataFixedFixed = dataFixedFixed["returns"]
    dataFixedFixed["type"] = pd.Series("FF", index=dataFixedFixed.index)
    dataRandomFixed = dataRandomFixed["returns"]
    dataRandomFixed["type"] = pd.Series("RF", index=dataRandomFixed.index)
    dataRandomRandom = dataRandomRandom["returns"]
    dataRandomRandom["type"] = pd.Series("RR", index=dataRandomRandom.index)

    concat = pd.concat([dataFixedFixed, dataRandomFixed, dataRandomRandom])
    plotTrainResults(concat)
