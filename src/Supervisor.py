import random
import threading
import queue

import pandas as pd

from Struct import Transition
from Environment import Environment


class Walker(threading.Thread):
    def __init__(self, agent, queue, lock, resultList):
        threading.Thread.__init__(self)
        self.agent = agent
        self.queue = queue
        self.lock = lock
        self.resultList = resultList
        return

    def run(self):
        while not self.queue.empty():
            # get job and build environment
            self.lock.acquire()
            param = self.queue.get()

            environment = Environment(*param)
            self.lock.release()

            # initial state is provided by the environment
            state = environment.initialState

            steps = 0
            rewards = []

            # walk until end of episode
            while not state:
                action = self.agent.takeAction(state)
                state, reward = environment.react(action)

                steps += 1
                rewards.append(reward)

            # did the episode end successfully?
            success = True if rewards[-1] == environment.bounty else False

            # store the results
            self.lock.acquire()
            self.resultList["environmentParameters"].append(param)
            self.resultList["steps"].append(steps)
            self.resultList["return"].append(sum(rewards))
            self.resultList["success"].append(success)
            self.lock.release()

        return


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
                self.agent.learn(self.agent.shortMemory)

            steps += 1
            rewards.append(reward)

        # did the episode end successfully?
        success = True if rewards[-1] == environment.bounty else False

        return steps, rewards, success

    def benchmark(self, environmentParameters, numEpisodes, untrained=False):
        """
        Let the agent start from different initial positions and observe it's performance.
        :param environmentParameters: list of initial positions
        :param numEpisodes: how many times to start over from each initial position
        :return: pandas data frame with len(environmentParameters) * numEpisodes rows
        """
        # dictionary to build data frame from
        results = dict()

        # add columns containing the agent's attributes
        results["untrained"] = pd.Series(untrained,
                                         index=[i for i in range(0, numEpisodes * len(environmentParameters))])
        results["epsilon"] = pd.Series(self.agent.epsilon,
                                       index=[i for i in range(0, numEpisodes * len(environmentParameters))])
        results["learningRate"] = pd.Series(self.agent.learningRate,
                                            index=[i for i in range(0, numEpisodes * len(environmentParameters))])
        results["discount"] = pd.Series(self.agent.discount,
                                        index=[i for i in range(0, numEpisodes * len(environmentParameters))])
        results["traceDecay"] = pd.Series(self.agent.traceDecay,
                                          index=[i for i in range(0, numEpisodes * len(environmentParameters))])
        results["memorySize"] = pd.Series(self.agent.memorySize,
                                          index=[i for i in range(0, numEpisodes * len(environmentParameters))])
        results["network"] = pd.Series(self.agent.q.network,
                                       index=[i for i in range(0, numEpisodes * len(environmentParameters))])
        results["targetGenerator"] = pd.Series(self.agent.targetGenerator.__name__,
                                               index=[i for i in range(0, numEpisodes * len(environmentParameters))])

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
        results["return"] = pd.Series(returnList)
        results["success"] = pd.Series(successList)

        return pd.DataFrame(results)

    def benchmarkMultiThreaded(self, environmentParameters, numEpisodes, untrained=False):
        """
        Let the agent start from different initial positions and observe it's performance.
        :param environmentParameters: list of initial positions
        :param numEpisodes: how many times to start over from each initial position
        :return: pandas data frame with len(environmentParameters) * numEpisodes rows
        """
        # dictionary to build data frame from
        results = dict()

        # add columns containing the agent's attributes
        results["untrained"] = pd.Series(untrained,
                                         index=[i for i in range(0, numEpisodes * len(environmentParameters))])
        results["epsilon"] = pd.Series(self.agent.epsilon,
                                       index=[i for i in range(0, numEpisodes * len(environmentParameters))])
        results["learningRate"] = pd.Series(self.agent.learningRate,
                                            index=[i for i in range(0, numEpisodes * len(environmentParameters))])
        results["discount"] = pd.Series(self.agent.discount,
                                        index=[i for i in range(0, numEpisodes * len(environmentParameters))])
        results["traceDecay"] = pd.Series(self.agent.traceDecay,
                                          index=[i for i in range(0, numEpisodes * len(environmentParameters))])
        results["memorySize"] = pd.Series(self.agent.memorySize,
                                          index=[i for i in range(0, numEpisodes * len(environmentParameters))])
        results["network"] = pd.Series(self.agent.q.network,
                                       index=[i for i in range(0, numEpisodes * len(environmentParameters))])
        results["targetGenerator"] = pd.Series(self.agent.targetGenerator.__name__,
                                               index=[i for i in range(0, numEpisodes * len(environmentParameters))])

        # define job queue and result storage
        resultList = {"environmentParameters": list(), "steps": list(), "return": list(), "success": list()}

        jobs = queue.Queue()
        for i in range(0, numEpisodes):
            for param in environmentParameters:
                jobs.put(param)

        resultLock = threading.Lock()

        # create threads
        threadList = []
        for i in range(0, 4):
            threadList.append(Walker(self.agent, jobs, resultLock, resultList))
            threadList[-1].start()

        # wait for them to finish
        for thread in threadList:
            thread.join()

        # append to results
        results["environmentParameters"] = pd.Series(resultList["environmentParameters"])
        results["steps"] = pd.Series(resultList["steps"])
        results["return"] = pd.Series(resultList["return"])
        results["success"] = pd.Series(resultList["success"])

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
