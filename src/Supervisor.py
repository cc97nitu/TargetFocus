from Environment import Environment
from Agent import Agent
from Struct import Transition
from QValue import QNeural
import torch


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
    """run an episode"""
    state = environment.initialState

    totalReward, steps = 0, 0

    while not state:
        action = agent.takeAction(state)
        state, reward = environment.react(action)
        agent.remember(Transition(action, reward, state))

        agent.learn(*agent.getSarsaLambda())

        totalReward += reward
        steps += 1

    return totalReward, steps


def benchmark(agent, environmentParameters, episodes):
    """get average reward over episodes"""
    rewards = []

    for run in range(episodes):
        reward, steps = episode(agent, Environment(*environmentParameters))
        rewards.append(reward)

    return rewards


def trainAgent(agent, environmentParameters, trainingEpisodes):
    """train an agent and measure its performance"""

    # learn from episodes
    for run in range(trainingEpisodes):
        totalReward, steps = learnFromEpisode(agent, Environment(*environmentParameters))
        agent.wipeShortMemory()

    return agent


if __name__ == '__main__':
    # initialize
    agent = Agent(QNeural(), epsilon=0.5)
    agent.q.trainer.epochs = 10

