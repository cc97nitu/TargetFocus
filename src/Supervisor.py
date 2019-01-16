from Environment import Environment
from Agent import Agent
from Struct import Transition
from QValue import QNeural
from FuncApprox.Network import FulCon2

import torch
import pandas as pd
import random
from copy import deepcopy


def episode(agent, environment):
    """run an episode"""
    state = environment.initialState

    totalReward, steps = 0, 0

    while not state:
        action = agent.takeAction(state)
        state, reward = environment.react(action)
        agent.remember(Transition(action, reward, state))

        totalReward += reward
        steps += 1

    distanceToGoal = torch.sqrt(torch.sum((state.focus - environment.focusGoal) ** 2)).item()

    return totalReward, steps


def learnFromEpisode(agent, environment):
    """run an episode and learn after each step"""
    state = environment.initialState

    totalReward, steps = 0, 0

    while not state:
        action = agent.takeAction(state)
        state, reward = environment.react(action)
        agent.remember(Transition(action, reward, state))

        agent.learn(*agent.getDQN(agent.shortMemory))

        totalReward += reward
        steps += 1

    return totalReward, steps


def experienceEpisode(agent, environment):
    """run an episode and remember it without learning"""
    state = environment.initialState

    while not state:
        action = agent.takeAction(state)
        state, reward = environment.react(action)
        agent.remember(Transition(action, reward, state))

        if len(agent.replayMemory) < agent.replayMemorySize:
            agent.replayMemory.append(deepcopy(agent.shortMemory))
        else:
            del agent.replayMemory[0]
            agent.replayMemory.append(deepcopy(agent.shortMemory))

    return


def benchmark(agent, environmentParameters, episodes):
    """get average reward over episodes"""
    rewards = []

    for run in range(episodes):
        reward, steps = episode(agent, Environment(*environmentParameters))
        rewards.append(reward)

    return rewards


def spatialBenchmark(agent, environmentParameters, episodes):
    """get average reward for different starting positions"""
    frames = []

    for parameters in environmentParameters:
        result = {"{}".format(parameters): benchmark(agent, parameters, episodes)}
        result = pd.melt(pd.DataFrame(result), var_name="environmentParameters", value_name="reward")
        frames.append(result)

    return pd.concat(frames)


def trainAgent(agent, environmentParameters, trainingEpisodes):
    """train an agent and measure its performance"""

    # learn from episodes
    for run in range(trainingEpisodes):
        totalReward, steps = learnFromEpisode(agent, Environment(*environmentParameters))
        agent.wipeShortMemory()

    return agent


def trainAgent_random(agent, environmentParameters, trainingEpisodes):
    """train an agent and measure its performance"""

    # learn from episodes
    for run in range(trainingEpisodes):
        # select initial environment parameters
        initialParameters = random.choice(environmentParameters)

        totalReward, steps = learnFromEpisode(agent, Environment(*initialParameters))
        agent.wipeShortMemory()

    return agent


def trainAgentOffline(agent, environmentParameters, trainingEpisodes):
    """train an agent and measure its performance"""

    # learn from episodes
    for run in range(trainingEpisodes):
        experienceEpisode(agent, Environment(*environmentParameters))
        agent.wipeShortMemory()

        if run % 10 == 0:
            # train from replay memory
            allInput, allLabels = [], []

            for shortMemory in agent.replayMemory:
                netInput, labels = agent.getSarsaLambda(shortMemory)
                allInput.append(netInput)
                allLabels.append(labels)

            allInput = torch.cat(allInput)
            allLabels = torch.cat(allLabels)

            agent.learn(allInput, allLabels)

    return agent


def trainAgentOffline_random(agent, environmentParameters, trainingEpisodes):
    """train an agent and measure its performance"""

    # learn from episodes
    for run in range(trainingEpisodes):
        # select initial environment parameters
        initialParameters = random.choice(environmentParameters)

        experienceEpisode(agent, Environment(*initialParameters))
        agent.wipeShortMemory()

        if run % 10 == 0:
            # train from replay memory
            allInput, allLabels = [], []

            for shortMemory in agent.replayMemory:
                netInput, labels = agent.getSarsaLambda(shortMemory)
                allInput.append(netInput)
                allLabels.append(labels)

            allInput = torch.cat(allInput)
            allLabels = torch.cat(allLabels)

            agent.learn(allInput, allLabels)

    return agent


if __name__ == '__main__':
    # initialize
    agent = Agent(QNeural(network=FulCon2()), epsilon=0.5)
    agent.q.trainer.epochs = 10
    environmentParameters = ((0, 0.01), (0.01, 0), (-0.01, -0.03))
