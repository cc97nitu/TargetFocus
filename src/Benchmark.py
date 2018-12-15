import os
import pickle
import pandas as pd

from itertools import product

from Agent import Agent
from QValue import QNeural
import FuncApprox.Network as Network
import FuncApprox.Trainer as Trainer

from Hypervisor import createEnvironmentParameters, policyIterationV3, policyIterationV4
from Supervisor import spatialBenchmark

if __name__ == '__main__':
    # save current path to file
    path = os.path.dirname(os.path.abspath(__file__))

    # initialize
    epsilons = (0.3, 0.5, 0.7, 0.9)
    environmentParameters = createEnvironmentParameters()
    benchEnvironmentParameters = (
        (0, 0.01), (0.01, 0), (-0.01, -0.03), (0, -0.04), (-0.04, 0), (0.02, 0.01), (-0.02, -0.02), (0.03, 0.01),
        (0.04, -0.04), (-0.04, 0.04))

    trainingEpisodes = (int(1e1), int(2e1), int(5e1), int(8e1))
    evaluationEpisodes = int(3e1)

    networks = (Network.FulCon10,)

    # do the benchmark
    trainResults = []
    benchResults = []

    for network, trainEpisodes in product(networks, trainingEpisodes):
        print("network={}, trainEpisodes={}".format(network, trainEpisodes))

        # create the agent
        agent = Agent(QNeural(network=network, trainer=Trainer.Rprop, epochs=5))

        # train him
        trainResult = policyIterationV4(agent, environmentParameters, epsilons, trainEpisodes, evaluationEpisodes)
        trainResult.loc[:, 'network'] = pd.Series(["{}".format(network()) for i in range(len(trainResult))])
        trainResult.loc[:, 'trainingEpisodes'] = pd.Series([trainEpisodes for i in range(len(trainResult))])
        trainResults.append(trainResult)

        # do spatial benchmark
        benchResult = spatialBenchmark(agent, benchEnvironmentParameters, evaluationEpisodes)
        benchResult.loc[:, 'network'] = pd.Series(["{}".format(network()) for i in range(len(trainResult))])
        benchResult.loc[:, 'trainingEpisodes'] = pd.Series([trainEpisodes for i in range(len(trainResult))])
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
