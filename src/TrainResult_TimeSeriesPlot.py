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

    sns.lineplot(x="episode", y="return", data=data)

    plt.title("random start, random goal")
    plt.show()
    plt.close()


if __name__ == "__main__":
    ### train results ###
    # fetch data
    data = torch.load("/dev/shm/agents.tar")

    plotTrainResults(data["returns"])

    # # concat two data frames
    # data6dRaw = torch.load("/home/conrad/RL/TempDiff/TargetFocus/src/dump/REINFORCE/6d-states-raw/gamma=0.999/No-Return-Normalization/Cat1/RR_Cat1_400_agents.tar")
    # data2dNormalized = torch.load("/home/conrad/RL/TempDiff/TargetFocus/src/dump/REINFORCE/2d-states-normalized/Cat1/RR_Cat1_400_agents.tar")
    #
    # data6dRaw = data6dRaw["returns"]
    # data6dRaw["states"] = pd.Series("6d-raw", index=data6dRaw.index)
    # data2dNormalized = data2dNormalized["returns"]
    # data2dNormalized["states"] = pd.Series("2d-norm", index=data2dNormalized.index)
    #
    # concat = pd.concat([data6dRaw, data2dNormalized,])
    # plotTrainResults(concat)
