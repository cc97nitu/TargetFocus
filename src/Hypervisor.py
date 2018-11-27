import pandas as pd
import pickle
import os

from Supervisor import benchmark, spatialBenchmark, trainAgent, trainAgentOffline
from Agent import Agent
from QValue import QNeural
import FuncApprox.Network as Network


def experienceComparison(agent, environmentParameters, trainingStages, evaluationEpisodes):
    """compare agent's performance in dependence of his experience"""

    # check performance before training
    results = {'before': benchmark(agent, environmentParameters, evaluationEpisodes)}

    # train the agent for first stage
    print("train to stage {:.0e}".format(trainingStages[0]))

    agent = trainAgent(agent, environmentParameters, trainingStages[0])
    results["{:.0e}".format(trainingStages[0])] = benchmark(agent, environmentParameters, evaluationEpisodes)

    # continue
    for i in range(1, len(trainingStages)):
        print("train to stage {:.0e}".format(trainingStages[i]))

        trainingEpisodes = trainingStages[i] - trainingStages[i - 1]
        agent = trainAgent(agent, environmentParameters, trainingEpisodes)
        results["{:.0e}".format(trainingStages[i])] = benchmark(agent, environmentParameters, evaluationEpisodes)

    # build pandas data frame
    results = pd.DataFrame(results)

    return results


def policyIteration(agent, environmentParameters, epsilons, trainingEpisodes, evaluationEpisodes):
    """general policy iteration with different epsilon-greedy policies"""
    # returns data as pandas data frame in wide-form

    # check performance before training
    results = {"before={:.1f}".format(agent.epsilon): benchmark(agent, environmentParameters, evaluationEpisodes)}

    # train under policies
    for epsilon in epsilons:
        print("epsilon={:.1f}".format(epsilon))
        agent.epsilon = epsilon

        agent = trainAgent(agent, environmentParameters, trainingEpisodes)
        results["epsilon={:.1f}".format(agent.epsilon)] = benchmark(agent, environmentParameters, evaluationEpisodes)

    # build pandas data frame
    results = pd.DataFrame(results)

    return results


def policyIterationV2(agent, environmentParameters, epsilons, trainingEpisodes, evaluationEpisodes):
    """general policy iteration with different epsilon-greedy policies"""
    # returns data as pandas data frame in long-form

    frames = []

    # check performance before training
    result = {"before={:.1f}".format(agent.epsilon): benchmark(agent, environmentParameters, evaluationEpisodes)}
    result = pd.melt(pd.DataFrame(result), var_name='policy', value_name='reward')
    result.loc[:, 'environmentParameters'] = pd.Series(["{}".format(environmentParameters) for i in range(len(result))])
    frames.append(result)

    # train under policies
    for epsilon in epsilons:
        print("epsilon={:.1f}".format(epsilon))
        agent.epsilon = epsilon

        agent = trainAgent(agent, environmentParameters, trainingEpisodes)
        result = {"epsilon={:.1f}".format(agent.epsilon): benchmark(agent, environmentParameters, evaluationEpisodes)}
        result = pd.melt(pd.DataFrame(result), var_name='policy', value_name='reward')
        result.loc[:, 'environmentParameters'] = pd.Series(["{}".format(environmentParameters) for i in range(len(result))])
        frames.append(result)

    # build common pandas data frame
    commonFrame = frames[0]
    for i in range(1, len(frames)):
        commonFrame = commonFrame.append(frames[i])

    return commonFrame


def policyIterationV3(agent, environmentParameters, epsilons, trainingEpisodes, evaluationEpisodes):
    """general policy iteration with different epsilon-greedy policies for different starting points"""
    # returns data as pandas data frame in long-form

    frames = []

    # check performance before training
    for parameters in environmentParameters:
        result = {"before={:.1f}".format(agent.epsilon): benchmark(agent, parameters, evaluationEpisodes)}
        result = pd.melt(pd.DataFrame(result), var_name='policy', value_name='reward')
        result.loc[:, 'environmentParameters'] = pd.Series(["{}".format(parameters) for i in range(len(result))])
        frames.append(result)

    # train under policies
    for epsilon in epsilons:
        print("epsilon={:.1f}".format(epsilon))
        agent.epsilon = epsilon

        # train for each starting point
        for parameters in environmentParameters:
            agent = trainAgentOffline(agent, parameters, trainingEpisodes)

        # measure performance from each starting point
        for parameters in environmentParameters:
            result = {"epsilon={:.1f}".format(agent.epsilon): benchmark(agent, parameters, evaluationEpisodes)}
            result = pd.melt(pd.DataFrame(result), var_name='policy', value_name='reward')
            result.loc[:, 'environmentParameters'] = pd.Series(["{}".format(parameters) for i in range(len(result))])
            frames.append(result)

    return pd.concat(frames)


if __name__ == '__main__':
    # save current path to file
    path = os.path.dirname(os.path.abspath(__file__))

    # build agent
    agent = Agent(QNeural(network=Network.FulCon1()), epsilon=0.3)

    # parameters
    # environmentParameters = (-0.01, -0.03)
    # trainingStages = (int(5e2), int(1e3), int(5e3), int(1e4), int(5e4), int(1e5))
    trainingStages = (int(2e2), int(5e2), int(8e2),)
    evaluationEpisodes = int(2e2)

    # # evaluate agent's performance
    # perf = experienceComparison(agent, environmentParameters, trainingStages, evaluationEpisodes)

    # general policy iteration
    epsilons = (0.3, 0.5, 0.7, 0.9)
    trainingEpisodes = int(2e2)
    # environmentParameters = ((0, 0.01), (0.01, 0), (-0.01, -0.03), (0, -0.04), (-0.04, 0))
    environmentParameters = ((0, 0.01), (0.01, 0), (-0.01, -0.03),)

    perf = policyIterationV3(agent, environmentParameters, epsilons, trainingEpisodes, evaluationEpisodes)

    # save to disk
    os.chdir(path)

    with open("../dump/policyIteration/epsilonGreedy/Sarsa(lambda)/testRun", 'wb') as file:
        pickle.dump(perf, file)

    # # do a spatial benchmark
    # benchEnvironmentParameters = ((0, 0.01), (0.01, 0), (-0.01, -0.03), (0, -0.04), (-0.04, 0), (0.02, 0.01), (-0.02, -0.02), (0.03, 0.01), (0.04, -0.04), (-0.04, 0.04))
    #
    # spatialPerf = spatialBenchmark(agent, benchEnvironmentParameters, evaluationEpisodes)
    #
    # os.chdir(path)
    #
    # with open("../dump/policyIteration/epsilonGreedy/Sarsa(lambda)/testSpatial", 'wb') as file:
    #     pickle.dump(spatialPerf, file)
