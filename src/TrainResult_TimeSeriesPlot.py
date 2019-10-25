"""
Visualize mean returns achieved during training.
"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import SQL


def plotTrainResult(data: pd.DataFrame):
    # plot
    sns.set(style="whitegrid")

    sns.lineplot(x="episode", y="return", data=data)

    # plt.title("random start, random goal")
    # plt.yticks([])
    plt.show()
    plt.close()


def multiTrainResults(dataSets: list, hueKeyword: str):
    # build annotated data frame
    dataFrames = []
    for dataSet in dataSets:
        data = SQL.retrieve(dataSet[0])
        data = data["returns"]
        data[hueKeyword] = pd.Series(dataSet[1], index=data.index)
        dataFrames.append(data)

    concatFrame = pd.concat(dataFrames)

    # plot
    print("beginning to plot")
    sns.set(style="whitegrid")

    sns.lineplot(x="episode", y="return", hue=hueKeyword, data=concatFrame)

    # plt.title("random start, random goal")
    # plt.yticks([])
    # plt.ylim(bottom=-40)
    plt.show()
    plt.close()


if __name__ == "__main__":
    ### train results ###

    # # plot single result
    # data = SQL.retrieve(row_id=137)
    # print(data["environmentConfig"])
    # print(data["hyperParameters"])
    #
    # plotTrainResult(data["returns"])

    # plot multiple results
    hueKeyword = "state definition"
    dataSets = [(137, "2d-norm"), (140, "6d-norm"), (134, "6d-norm_6noise"), (135, "6d-norm_60noise")]  # assumes tuples of form (row_id, hueIdentifier)

    multiTrainResults(dataSets, hueKeyword)
