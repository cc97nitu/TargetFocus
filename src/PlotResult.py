import pickle
import matplotlib.pyplot as plt
import seaborn as sns


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


def boxPlotGPI(data):
    """plot results from general policy iteration"""
    # create figure
    fig, ax = plt.subplots(figsize=(28/3, 7))

    # seaborn boxplot
    ax = sns.boxplot(x='policy', y='reward', hue='environmentParameters', data=data)

    # add some annotation
    fig.suptitle("general policy iteration with offline Sarsa($\lambda$)", size='xx-large')
    ax.set_title("FulCon1, train=200,eval=200")

    # show the plot
    plt.show()
    plt.close()

    return


def boxPlotSpatial(data):
    """plot results from spatial benchmark"""
    # create figure
    fig, ax = plt.subplots()

    # seaborn boxplot
    ax = sns.boxplot(x='environmentParameters', y='reward', data=data)

    # add some annotation
    fig.suptitle("agent's performance from different starting points", size='xx-large')

    # show the plot
    plt.show()
    plt.close()

    return


if __name__ == '__main__':
    # fetch data
    with open("../dump/policyIteration/epsilonGreedy/Sarsa(lambda)/testRun", 'rb') as file:
        data = pickle.load(file)

    boxPlotGPI(data)
