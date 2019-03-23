import random

import pandas as pd

from Struct import Transition
from Environment import Environment


class Supervisor(object):
    """
    Provides convenience methods to manage agent / environment interaction.
    """

    def __init__(self, agent):
        self.agent = agent

        return

    def walk(self, environment, remember=True, learnOnline=False):
        """
        Experience an episode.
        :param environment: environment to interact with
        :return: number of steps, earned rewards, success
        """
        steps = 0
        rewards = []

        # initial state is provided by the environment
        state = environment.initialState

        # walk until end of episode
        while not state:
            action = self.agent.takeAction(state)
            state, reward = environment.react(action)

            if remember:
                self.agent.remember(Transition(action, reward, state))

            if learnOnline:
                self.agent.learn(agent.shortMemory)

            steps += 1
            rewards.append(reward)

        # did the episode end successfully?
        success = True if rewards[-1] == environment.bounty else False

        return steps, rewards, success

    def benchmark(self, environmentParameters, numEpisodes):
        """
        Let the agent start from different initial positions and observe it's performance.
        :param environmentParameters: list of initial positions
        :param numEpisodes: how many times to start over from each initial position
        :return: pandas data frame with len(environmentParameters) * numEpisodes rows
        """
        # dictionary to build data frame from
        results = dict()

        # add columns containing the agent's attributes
        results["epsilon"] = pd.Series(self.agent.epsilon, index=[i for i in range(0, numEpisodes)])
        results["learningRate"] = pd.Series(self.agent.learningRate, index=[i for i in range(0, numEpisodes)])
        results["discount"] = pd.Series(self.agent.discount, index=[i for i in range(0, numEpisodes)])
        results["traceDecay"] = pd.Series(self.agent.traceDecay, index=[i for i in range(0, numEpisodes)])
        results["memorySize"] = pd.Series(self.agent.memorySize, index=[i for i in range(0, numEpisodes)])
        results["network"] = pd.Series(self.agent.q.network, index=[i for i in range(0, numEpisodes)])
        results["targetGenerator"] = pd.Series(self.agent.targetGenerator.__name__,
                                               index=[i for i in range(0, numEpisodes)])

        # do the actual benchmark
        environmentParametersList, stepList, returnList, successList = [], [], [], []

        for episode in range(0, numEpisodes):
            for param in environmentParameters:
                environmentParametersList.append("[{:.3f}, {:.3f}]".format(*param))

                steps, rewards, success = self.walk(Environment(*param), remember=False, learnOnline=False)
                stepList.append(steps)
                returnList.append(sum(rewards))
                successList.append(success)

        results["environmentParameters"] = pd.Series(environmentParametersList)
        results["steps"] = pd.Series(stepList)
        results["returns"] = pd.Series(returnList)
        results["success"] = pd.Series(successList)

        return pd.DataFrame(results)

    def learnOnline(self, environmentParameters, numEpisodes):
        """
        Learn online from episodes starting from randomly selected initial positions.
        :param environmentParameters: list of initial positions
        :param numEpisodes: number of episodes
        :return: None
        """
        for episode in range(0, numEpisodes):
            # randomly choose starting point
            param = random.choice(environmentParameters)
            self.walk(Environment(*param), remember=True, learnOnline=True)

        return


if __name__ == '__main__':
    from Agent import Agent
    from Environment import EligibleEnvironmentParameters
    from QValue import QNeural
    from FuncApprox.TargetGenerator import sarsaLambda

    # build agent
    agent = Agent(q=QNeural(), epsilon=0.7, discount=0.9, learningRate=0.9, memorySize=1, traceDecay=0,
                  targetGenerator=sarsaLambda)

    # build rover
    rover = Supervisor(agent)

    # generate starting points
    environmentParameters = EligibleEnvironmentParameters(-0.05, 0.05, 0.01)

    # run some episodes
    steps, rewards, success = rover.walk(Environment(0, 0), learnOnline=False)
    print("steps: {0}, return: {1}, success: {2}".format(steps, sum(rewards), success))

    rover.learnOnline([(0, 0)], 100)

    # do a benchmark
    print(rover.benchmark([(0, 0)], 10))
