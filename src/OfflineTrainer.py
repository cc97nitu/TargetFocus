import pickle
import os
import torch

from Agent import Agent
from QValue import QNeural
import FuncApprox.Network as Network
from Supervisor import spatialBenchmark

from PlotResult import boxPlotSpatial


def trainFromReplayMemory(agent, epochs):
    # train from replay memory
    for epoch in range(epochs):
        allInput, allLabels = [], []

        for shortMemory in agent.replayMemory:
            netInput, labels = agent.getDQN(shortMemory)
            allInput.append(netInput)
            allLabels.append(labels)

        allInput = torch.cat(allInput)
        allLabels = torch.cat(allLabels)

        agent.learn(allInput, allLabels)

    return agent


if __name__ == '__main__':
    # save current path to file
    path = os.path.dirname(os.path.abspath(__file__))

    # fetch the replay memory
    with open("../dump/policyIteration/epsilonGreedy/Sarsa(lambda)/replayMemory_traceLength=1", 'rb') as file:
        replayMemory = pickle.load(file)

    # build the agent
    agent = Agent(QNeural(network=Network.FulCon4()), epsilon=0.3)
    agent.replayMemory = replayMemory

    # train the agent from his replay memory
    agent = trainFromReplayMemory(agent, epochs=int(1e1))

    # do a spatial benchmark
    # benchEnvironmentParameters = ((0, 0.01), (0.01, 0), (-0.01, -0.03), (0, -0.04), (-0.04, 0), (0.02, 0.01), (-0.02, -0.02), (0.03, 0.01), (0.04, -0.04), (-0.04, 0.04))
    benchEnvironmentParameters = ((0, 0.01), (0.01, 0), (-0.01, -0.03),)
    evaluationEpisodes = int(1e2)

    spatialPerf = spatialBenchmark(agent, benchEnvironmentParameters, evaluationEpisodes)

    # show
    boxPlotSpatial(spatialPerf)
