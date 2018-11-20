import pickle
import matplotlib.pyplot as plt
import seaborn as sns


def boxPlotExperience(data):
    # create figure
    fig, ax = plt.subplots(figsize=(12, 9))

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


if __name__ == '__main__':
    # fetch data
    with open("/home/conrad/RL/TempDiff/TargetFocus/dump/policyIteration/epsilonGreedy/Sarsa(lambda)/testRun", 'rb') as file:
        data = pickle.load(file)

    boxPlotExperience(data)
