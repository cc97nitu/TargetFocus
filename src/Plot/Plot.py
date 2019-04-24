import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns


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


def successPlot(data):
    # axes
    trainEpisodes = data["trainEpisodes"].unique()
    environmentParameters = data["environmentParameters"].unique()

    successRate = np.empty((len(trainEpisodes), len(environmentParameters)))

    # selectors
    networks = data["network"].unique()
    generators = data["targetGenerator"].unique()
    learningRates = data["learningRate"].unique()

    epsilon = 0.9
    selEpsilon = data["epsilon"] == epsilon

    # do the plots
    for generator in generators:
        selGenerator = data["targetGenerator"] == generator

        for learningRate in learningRates:
            selLearningRate = data["learningRate"] == learningRate

            for network in networks:
                selNetwork = data["network"] == network

                # calculate success rate
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
                            successRate[i][j] = sum(selection["success"]) / len(selection["success"])

                # plot it
                fig, ax = plt.subplots()

                ax.pcolormesh(environmentParameters, trainEpisodes, successRate)

                fig.suptitle(r"{}, $\alpha$={}, $\epsilon$={}".format(generator, learningRate, epsilon))
                ax.set_title("{}".format(network))

                for tick in ax.get_xticklabels():
                    tick.set_rotation(45)

                ax.set_xlabel(ax.get_xlabel(), size='x-large')
                ax.set_ylabel(ax.get_ylabel(), size='x-large')

                plt.show()
                plt.close()

                continue


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

    epsilon = 0.9
    selEpsilon = data["epsilon"] == epsilon

    # do the plots
    for generator in generators:
        selGenerator = data["targetGenerator"] == generator

        for learningRate in learningRates:
            selLearningRate = data["learningRate"] == learningRate

            for network in networks:
                selNetwork = data["network"] == network

                # calculate success rate
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

                # axes[0].pcolormesh(environmentParameters, trainEpisodes, successRate)
                # axes[1].pcolormesh(environmentParameters, trainEpisodes, steps)
                image0 = axes[0].imshow(successRate, cmap="winter")
                image1 = axes[1].imshow(steps)

                fig.suptitle(r"{}, {}, $\alpha$={}, $\epsilon$={}".format(generator, network, learningRate, epsilon))
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


def stepPlot(data):
    # axes
    trainEpisodes = data["trainEpisodes"].unique()
    environmentParameters = data["environmentParameters"].unique()

    steps = np.empty((len(trainEpisodes), len(environmentParameters)))

    # selectors
    networks = data["network"].unique()
    generators = data["targetGenerator"].unique()
    learningRates = data["learningRate"].unique()

    epsilon = 0.9
    selEpsilon = data["epsilon"] == epsilon

    # do the plots
    for generator in generators:
        selGenerator = data["targetGenerator"] == generator

        for learningRate in learningRates:
            selLearningRate = data["learningRate"] == learningRate

            for network in networks:
                selNetwork = data["network"] == network

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
                fig, ax = plt.subplots()

                ax.pcolormesh(environmentParameters, trainEpisodes, steps)

                fig.suptitle(r"{}, $\alpha$={}, $\epsilon$={}".format(generator, learningRate, epsilon))
                ax.set_title("{}".format(network))

                for tick in ax.get_xticklabels():
                    tick.set_rotation(45)

                ax.set_xlabel(ax.get_xlabel(), size='x-large')
                ax.set_ylabel(ax.get_ylabel(), size='x-large')

                plt.show()
                plt.close()

                continue


def plotStateDistribution(memory):
    # extract relative coordinates
    relCoords = list()
    for transition in memory:
        relCoords.append(transition.action.state.relCoord)

    # plot them
    fig, ax = plt.subplots()

    for coord in relCoords:
        ax.scatter(coord[0], coord[1])

    ax.axhline(0)
    ax.axvline(0)

    plt.show()
    plt.close()

    return
