import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import product


def boxPlotExperience(data):
    # create figure
    fig, ax = plt.subplots(figsize=(28/3, 7))

    # seaborn boxplot
    ax = sns.boxplot(data=data)

    # add some annotation
    ax.set_title("agent's performance in dependency of experience", size='xx-large')
    ax.set_xlabel("experienced time steps", size='x-large')
    ax.set_ylabel("received reward", size='x-large')

    # show the plot
    plt.show()
    plt.close()

    return


def boxPlotGPI(data, network=None, trainEpisodes=None):
    """plot results from general policy iteration"""
    # create figure
    fig, ax = plt.subplots()

    # seaborn boxplot
    ax = sns.boxplot(x='policy', y='reward', hue='environmentParameters', data=data)

    # add some annotation
    fig.suptitle("general policy iteration with Sarsa($\lambda$)", size='xx-large')
    ax.set_title("network={0}, training episodes={1}".format(network, trainEpisodes), size='x-large')

    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    ax.set_xlabel(ax.get_xlabel(), size='x-large')
    ax.set_ylabel(ax.get_ylabel(), size='x-large')

    # show the plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    # plt.savefig('../dump/policyIteration/epsilonGreedy/Sarsa(lambda)/12.4.18/plot/trainResults/network={0},trainingEpisodes={1}.png'.format(network, trainEpisodes))
    plt.close()

    return


def boxPlotSpatial(data, network=None, trainEpisodes=None):
    """plot results from spatial benchmark"""
    # create figure
    fig, ax = plt.subplots()

    # seaborn boxplot
    ax = sns.boxplot(x='environmentParameters', y='reward', data=data)

    # add some annotation
    fig.suptitle("agent's performance from different starting points", size='xx-large')
    ax.set_title("network={0}, training episodes={1}".format(network, trainEpisodes), size='x-large')

    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    ax.set_xlabel(ax.get_xlabel(), size='x-large')
    ax.set_ylabel(ax.get_ylabel(), size='x-large')

    ax.axhline(y=-100, color='green', alpha=0.2)

    # show the plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    # plt.savefig('../dump/policyIteration/epsilonGreedy/Sarsa(lambda)/12.4.18/plot/spatialResults/network={0},trainingEpisodes={1}.png'.format(network, trainEpisodes))
    plt.close()

    return


if __name__ == '__main__':
    # fetch data
    with open("../dump/policyIteration/epsilonGreedy/Sarsa(lambda)/benchmarkTrain", 'rb') as file:

        data = pickle.load(file)

    # loop over benchmark data frame
    trainingEpisodes = data['trainingEpisodes'].unique()
    networks = data['network'].unique()

    for trainEpisodes, network in product(trainingEpisodes, networks):
        dataRestricted = data.loc[(data['trainingEpisodes'] == trainEpisodes) & (data['network'] == network)]

        boxPlotGPI(dataRestricted, network=network, trainEpisodes=trainEpisodes)

    # boxPlotGPI(data)
