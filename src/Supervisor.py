from Environment import Environment
from Agent import Agent
from Struct import Transition
from QValue import QNeural


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


if __name__ == '__main__':
    # initialize
    agent = Agent(QNeural())
    agent.epsilon = 0.9

    evaluationEpisodes = int(2e2)

    # rewards before training
    print("evaluating agent performance before training")
    rewardsBeforeTraining = benchmark(agent, (0, 0.01), evaluationEpisodes)

    # learn from episodes
    print("begin training")

    episodes = int(1e1)

    for run in range(episodes):
        totalReward, steps = learnFromEpisode(agent, Environment(0, 0.01))
        agent.wipeShortMemory()

    # rewards after training
    print("evaluating agent performance after training")
    rewardsAfterTraining = benchmark(agent, (0, 0.01), evaluationEpisodes)

    print("average reward: before={0:.2f}, after={1:.2f}".format(sum(rewardsBeforeTraining) / len(rewardsBeforeTraining), sum(rewardsAfterTraining) / len(rewardsAfterTraining)))



