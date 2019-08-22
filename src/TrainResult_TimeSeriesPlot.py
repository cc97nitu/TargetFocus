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

    sns.lineplot(x="episode", y="return", hue="state_def", data=data)

    # plt.title("random start, random goal")
    # plt.yticks([])
    plt.show()
    plt.close()


if __name__ == "__main__":
    # ### train results ###
    # # fetch data
    # data = torch.load("/home/conrad/RL/TempDiff/TargetFocus/src/dump/DQN/Adam(lr=2e-5)/propReward/gamma=0/6d-norm_4A_RR_FC7_2000_benchmark")
    #
    # plotTrainResults(data["returns"])

    # concat two data frames
    dataA = torch.load("/home/conrad/RL/TempDiff/TargetFocus/src/dump/DQN/Adam(lr=2e-5)/propReward/gamma=0/6d-norm_4A_RR_FC7_2000_agents.tar")
    dataB = torch.load("/home/conrad/RL/TempDiff/TargetFocus/src/dump/DQN/Adam(lr=2e-5)/propReward/gamma=0/2d-norm_4A_RR_FC7_400_agents.tar")

    dataA = dataA["returns"]
    dataA["state_def"] = pd.Series("6d-norm", index=dataA.index)
    dataB = dataB["returns"]
    dataB["state_def"] = pd.Series("2d-norm", index=dataB.index)


    concat = pd.concat([dataA, dataB,])
    plotTrainResults(concat)
