"""
Visualize mean returns achieved during training.
"""
import torch

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def plotTrainResults(data: pd.DataFrame):
    # plot
    sns.set(style="whitegrid")

    sns.lineplot(x="episode", y="return", hue="reward", data=data)

    # plt.title("random start, random goal")
    # plt.yticks([])
    plt.show()
    plt.close()


if __name__ == "__main__":
    # ### train results ###
    # # fetch data
    # data = torch.load("/dev/shm/agents.tar")
    #
    # plotTrainResults(data["returns"])

    # concat two data frames
    dataA = torch.load("/TargetFocus/src/dump/REINFORCE/6d-states-normalized/propReward/6d-norm_9A_RR_Cat3_propReward_2000_agents.tar")
    dataB = torch.load("/TargetFocus/src/dump/REINFORCE/6d-states-normalized/propRewardStepPenalty/6d-norm_9A_RR_Cat3_propRewardStepPenalty_2000_agents.tar")
    dataC = torch.load("/TargetFocus/src/dump/REINFORCE/6d-states-normalized/ConstantRewardPerStep/6d-norm_9A_RR_Cat3_constantRewardPerStep_2000_agents.tar")

    dataA = dataA["returns"]
    dataA["reward"] = pd.Series("A", index=dataA.index)
    dataB = dataB["returns"]
    dataB["reward"] = pd.Series("B", index=dataB.index)
    dataC = dataC["returns"]
    dataC["reward"] = pd.Series("C", index=dataC.index)


    concat = pd.concat([dataA, dataB, dataC])
    plotTrainResults(concat)
