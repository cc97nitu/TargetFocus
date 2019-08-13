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

    sns.lineplot(x="episode", y="return", hue="actions", data=data)

    plt.title("random start, random goal")
    plt.show()
    plt.close()


if __name__ == "__main__":
    ### train results ###
    # # fetch data
    # data = torch.load("/dev/shm/agents.tar")
    #
    # plotTrainResults(data["returns"])

    # concat two data frames
    data4A = torch.load("/home/dylan/RL/TempDiff/TargetFocus/src/dump/REINFORCE/6d-states-normalized/6d-norm_4A_RR_Cat1_2000_agents.tar")
    data9A = torch.load("/home/dylan/RL/TempDiff/TargetFocus/src/dump/REINFORCE/6d-states-normalized/6d-norm_9A_Cat1_2000_agents.tar")
    data25A = torch.load("/home/dylan/RL/TempDiff/TargetFocus/src/dump/REINFORCE/6d-states-normalized/6d-norm_25A_RR_Cat1_2000_agents.tar")

    data4A = data4A["returns"]
    data4A["actions"] = pd.Series("four", index=data4A.index)
    data9A = data9A["returns"]
    data9A["actions"] = pd.Series("nine", index=data9A.index)
    data25A = data25A["returns"]
    data25A["actions"] = pd.Series("twenty-five", index=data25A.index)


    concat = pd.concat([data4A, data9A, data25A ])
    plotTrainResults(concat)
