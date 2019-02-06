import torch

from Struct import Transition


def experienceEpisode(agent, environment):
    """
    Experience a whole episode and store observed transitions in episode memory
    :param agent: the acting agent
    :param environment: the environment to interact with
    :return: total reward accumulated over the episode
    """
    # flush episode memory at first
    agent.episodeMemory = []

    state = environment.initialState

    totalReward = 0

    # continue until terminate state is encountered
    while not state:
        action = agent.act(state)
        state, reward = environment.react(action)

        agent.episodeMemory.append(Transition(action, reward, state))
        totalReward += reward

    return totalReward

def learnDuringEpisode(agent, environment):
    # flush episode memory at first
    agent.episodeMemory = []

    state = environment.initialState

    totalReward = 0

    # continue until terminate state is encountered
    while not state:
        action = agent.act(state)
        state, reward = environment.react(action)
        transition = Transition(action, reward, state)

        agent.learn_Q([transition])

        agent.episodeMemory.append(transition)
        totalReward += reward

    return totalReward


if __name__ == '__main__':
    import random
    from itertools import product

    from Environment import Environment, createEnvironmentParameters
    from QValue import QValue
    from FuncApprox.Network import FulConI1
    from FuncApprox.Trainer import SGD
    from Agent import Agent

    # initialize environment
    resolutionTarget, resolutionOversize = 80, 20
    resolution = resolutionTarget + resolutionOversize

    environmentParameters = createEnvironmentParameters()
    environment = Environment(*random.choice(environmentParameters), resolutionTarget, resolutionOversize)

    # action set
    possibleChangesPerMagnet = (1e-2, 1e-3, 0, -1e-2, -1e-3)
    actionSet = tuple(torch.tensor((x, y), dtype=torch.float) for x, y in
                      product(possibleChangesPerMagnet, possibleChangesPerMagnet))


    # initialize value function and agent
    valueFunction = QValue(FulConI1, resolution, actionSet, SGD, 20)
    agent = Agent(valueFunction, 0.9, actionSet, 0.9, 0.5)

    # experience an episode
    print(experienceEpisode(agent, Environment(*random.choice(environmentParameters), resolutionTarget, resolutionOversize)))

    # learn during an episode
    print(learnDuringEpisode(agent, Environment(*random.choice(environmentParameters), resolutionTarget, resolutionOversize)))
