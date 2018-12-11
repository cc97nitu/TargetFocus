import pickle
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns


def plotTrainerComparison(results, optimizers):
    # do the plotting
    fig, axes = plt.subplots()

    axes = sns.lineplot(x='epoch', y='loss', hue="Optimizer", style='Optimizer', data=results)

    # add annotation
    axes.set_title("Comparison of Optimizers", size='x-large')

    # ax.set_title("network={0}, training episodes={1}".format(network, trainEpisodes), size='x-large')
    # ax.set_xlabel(ax.get_xlabel(), size='large')
    # ax.set_ylabel(ax.get_ylabel(), size='large')

    plt.savefig("../../dump/supervisedLearning/12.11.18/plot/{}.png".format(optimizers))

    # plt.show()
    # plt.close()

    return


# fetch the data
with open("../../dump/supervisedLearning/12.11.18/trainResults", 'rb') as file:
    data = pickle.load(file)

trainingResults = data.loc[(data['set'] == "training")]
validationResults = data.loc[(data['set'] == "validation")]

# restrict
optimizerList = (("SGD", "Adam", "Adamax"),
                 ("SGD", "RMSprop", "Rprop"),
                 ("SGD", "ASGD",),
                 ("SGD", "Adagrad", "Adadelta"),
                 ("SGD", "Adam", "Adamax", "Rprop", "Adagrad"))

for optimizers in optimizerList:
    results = trainingResults.loc[(trainingResults['Optimizer'].isin(optimizers))]

    plotTrainerComparison(results, optimizers)
