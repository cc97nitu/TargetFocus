import os
import pickle
import pandas as pd

from itertools import product

from Agent import Agent
from QValue import QNeural
import FuncApprox.Network as Network

from Hypervisor import policyIterationV3
from Supervisor import spatialBenchmark

if __name__ == '__main__':
    # save current path to file
    path = os.path.dirname(os.path.abspath(__file__))

    # initialize
    epsilons = (0.3, 0.5, 0.7, 0.9)
    environmentParameters = ((0, 0.01), (0.01, 0), (-0.01, -0.03), (0, -0.04), (-0.04, 0))
    benchEnvironmentParameters = (
        (0, 0.01), (0.01, 0), (-0.01, -0.03), (0, -0.04), (-0.04, 0), (0.02, 0.01), (-0.02, -0.02), (0.03, 0.01),
        (0.04, -0.04), (-0.04, 0.04))

    trainingEpisodes = (int(2e2), int(4e2))
    evaluationEpisodes = int(3e2)

    networks = (Network.FulCon1, Network.FulCon4, Network.FulCon6, Network.FulCon7, Network.FulCon8, Network.FulCon9)

    # do the benchmark
    trainResults = []
    benchResults = []

    for network, trainEpisodes in product(networks, trainingEpisodes):
        # create the agent
        agent = Agent(QNeural(network=network()))

        # train him
        trainResult = policyIterationV3(agent, environmentParameters, epsilons, trainEpisodes, evaluationEpisodes)
        trainResult.loc[:, 'network'] = pd.Series(["{}".format(network()) for i in range(len(trainResult))])
        trainResult.loc[:, 'trainingEpisodes'] = pd.Series(["{}".format(trainEpisodes) for i in range(len(trainResult))])
        trainResults.append(trainResult)

        # do spatial benchmark
        benchResult = spatialBenchmark(agent, benchEnvironmentParameters, evaluationEpisodes)
        benchResult.loc[:, 'network'] = pd.Series(["{}".format(network()) for i in range(len(trainResult))])
        benchResult.loc[:, 'trainingEpisodes'] = pd.Series(["{}".format(trainEpisodes) for i in range(len(trainResult))])
        benchResults.append(benchResult)

    # concat results to pandas data frame
    trainResults = pd.concat(trainResults)
    benchResults = pd.concat(benchResults)

    # save to disk
    os.chdir(path)

    with open("../dump/policyIteration/epsilonGreedy/Sarsa(lambda)/benchmarkTrain", 'wb') as file:
        pickle.dump(trainResults, file)

    with open("../dump/policyIteration/epsilonGreedy/Sarsa(lambda)/benchmarkBench", 'wb') as file:
        pickle.dump(benchResults, file)
