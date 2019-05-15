import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import pandas as pd
import torch


def boxPlot(data, x, y, hue=None, **kwargs):
    """plot results from spatial benchmark"""
    # create figure
    fig, ax = plt.subplots()

    # seaborn boxplot
    ax = sns.boxplot(x=x, y=y, hue=hue, data=data)

    # add some annotation
    if "supTitle" in kwargs.keys():
        fig.suptitle(kwargs["supTitle"], size='xx-large')
    if "axTitle" in kwargs.keys():
        ax.set_title(kwargs["axTitle"], size='x-large')

    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    ax.set_xlabel(ax.get_xlabel(), size='x-large')
    ax.set_ylabel(ax.get_ylabel(), size='x-large')

    # show the plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    plt.close()

    return


def stepsAndSuccess(data):
    # axes
    trainEpisodes = data["trainEpisodes"].unique()
    environmentParameters = data["environmentParameters"].unique()

    successRate = np.empty((len(trainEpisodes), len(environmentParameters)))
    steps = np.empty((len(trainEpisodes), len(environmentParameters)))

    # selectors
    networks = data["network"].unique()
    generators = data["targetGenerator"].unique()
    learningRates = data["learningRate"].unique()
    discounts = data["discount"].unique()

    epsilon = 0.9
    selEpsilon = data["epsilon"] == epsilon

    # do the plots
    for generator in generators:
        selGenerator = data["targetGenerator"] == generator

        for learningRate in learningRates:
            selLearningRate = data["learningRate"] == learningRate

            for discount in discounts:
                selDiscount = data["discount"] == discount

                for network in networks:
                    selNetwork = data["network"] == network

                    # calculate success rate
                    for i in range(0, len(trainEpisodes)):
                        for j in range(0, len(environmentParameters)):
                            selTrainEpisodes = data["trainEpisodes"] == trainEpisodes[i]
                            selEnvironmentParameters = data["environmentParameters"] == environmentParameters[j]

                            # restrict rows according to selectors
                            selection = data.loc[
                                selEpsilon & selGenerator & selLearningRate & selLearningRate & selDiscount & selNetwork & selTrainEpisodes & selEnvironmentParameters]

                            if len(selection) == 0:
                                print(r"no data for {}, $\alpha$={}, {}, $\epsilon$={}".format(generator, learningRate,
                                                                                               network, epsilon))
                            else:
                                successRate[i][j] = sum(selection["success"]) / len(selection["success"])

                    # get steps
                    for i in range(0, len(trainEpisodes)):
                        for j in range(0, len(environmentParameters)):
                            selTrainEpisodes = data["trainEpisodes"] == trainEpisodes[i]
                            selEnvironmentParameters = data["environmentParameters"] == environmentParameters[j]

                            # restrict rows according to selectors
                            selection = data.loc[
                                selEpsilon & selGenerator & selLearningRate & selLearningRate & selNetwork & selTrainEpisodes & selEnvironmentParameters]

                            if len(selection) == 0:
                                print(r"no data for {}, $\alpha$={}, {}, $\epsilon$={}".format(generator, learningRate,
                                                                                               network, epsilon))
                            else:
                                steps[i][j] = sum(selection["steps"]) / len(selection["steps"])

                    # plot it
                    fig, axes = plt.subplots(1, 2, sharey=True)

                    image0 = axes[0].imshow(successRate, cmap="winter")
                    image1 = axes[1].imshow(steps)

                    fig.suptitle(
                        "{}, {}".format(generator, network) + "\n" + r"$\alpha$={}, $\epsilon$={}, $\gamma$={}".format(
                            learningRate, epsilon, discount))
                    axes[0].set_title("success rate")
                    axes[1].set_title("steps")

                    for ax in axes:
                        ax.set_xticks(np.arange(len(successRate[0])))
                        ax.set_yticks(np.arange(len(successRate)))
                        ax.set_yticklabels(trainEpisodes)
                        ax.set_xticklabels(environmentParameters)

                        for tick in ax.get_xticklabels():
                            tick.set_rotation(45)

                        ax.set_xlabel(ax.get_xlabel(), size='x-large')
                        ax.set_ylabel(ax.get_ylabel(), size='x-large')

                    # add colorbars
                    divider = make_axes_locatable(axes[0])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    fig.colorbar(mappable=image0, cax=cax, orientation="vertical")

                    divider = make_axes_locatable(axes[1])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    fig.colorbar(mappable=image1, cax=cax, orientation="vertical")

                    plt.show()
                    plt.close()

                    continue


def distributions(results, kind=None):
    # selectors
    networks = results["network"].unique()
    generators = results["targetGenerator"].unique()
    learningRates = results["learningRate"].unique()
    discounts = results["discount"].unique()

    # restrict to trained agents acting greedy
    epsilon = 0.9
    selEpsilon = results["epsilon"] == epsilon

    maxTrainEpisodes = results["trainEpisodes"].max()
    selTrainEpisodes = results["trainEpisodes"] == maxTrainEpisodes

    # do the plots
    for generator in generators:
        selGenerator = results["targetGenerator"] == generator

        for learningRate in learningRates:
            selLearningRate = results["learningRate"] == learningRate

            for discount in discounts:
                selDiscount = results["discount"] == discount

                for network in networks:
                    selNetwork = results["network"] == network

                    # restrict rows according to selectors
                    selection = results.loc[
                        selEpsilon & selGenerator & selLearningRate & selLearningRate & selDiscount & selNetwork & selTrainEpisodes]

                    # plot
                    if kind is not None:
                        plot = sns.jointplot(x="return", y="steps", data=selection, kind=kind)
                    else:
                        plot = sns.jointplot(x="return", y="steps", data=selection)

                    plot.fig.suptitle(
                        "{}, {}".format(generator, network) + "\n" + r"$\alpha$={}, $\epsilon$={}, $\gamma$={}".format(
                            learningRate, epsilon, discount))

                    plt.show()
                    plt.close()

    return


def plotMemoryDistribution(memory):
    def stateDistribution(memory):
        # extract relative coordinates
        x, y = list(), list()
        for transition in memory:
            x.append(transition.action.state.relCoord[0])
            y.append(transition.action.state.relCoord[1])

        # put them into pandas data frame
        coordinates = [("x", x), ("y", y)]
        coordinates = pd.DataFrame.from_items(coordinates)

        # plot them
        g = sns.jointplot(x="x", y="y", data=coordinates, kind="kde")
        # g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
        # g.ax_joint.collections[0].set_alpha(0)
        # g.set_axis_labels("$X$", "$Y$");

        plt.show()
        plt.close()

        return

    def rewardDistribution(memory):
        # get change in distance to origin
        distances, rewards = list(), list()
        for transition in memory:
            previousDistance = torch.sqrt(torch.sum(transition.action.state.relCoord ** 2)).item()
            afterwardsDistance = torch.sqrt(torch.sum(transition.nextState.relCoord ** 2)).item()
            distances.append(afterwardsDistance - previousDistance)  # negative distance means coming closer to goal
            rewards.append(transition.reward)

        # put them into pandas data frame
        data = [("distance", distances), ("reward", rewards)]
        data = pd.DataFrame.from_items(data)

        # plot them
        sns.distplot(data["distance"])
        plt.show()
        plt.close()

        sns.distplot(data["reward"])
        plt.show()
        plt.close()

        sns.jointplot(x="distance", y="reward", data=data, kind="kde")
        plt.show()
        plt.close()

        return

    def actionDistribution(memory):
        # get actions
        changeX, changeY = list(), list()

        for transition in memory:
            changeX.append(transition.action.changes[0].item())
            changeY.append(transition.action.changes[1].item())

        # put them into pandas data frame
        data = [("changeX", changeX), ("changeY", changeY)]
        data = pd.DataFrame.from_items(data)

        # plot them
        sns.jointplot(x="changeX", y="changeY", data=data)
        plt.show()
        plt.close()
        return

    stateDistribution(memory)
    rewardDistribution(memory)
    actionDistribution(memory)
    return
