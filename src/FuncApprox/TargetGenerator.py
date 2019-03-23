import random
import torch


def sarsaLambda(parent, memory):
    """
    Create update targets from memory according to SARSA rule.
    :param parent: agent object subject to update targets
    :param memory: memory to create targets upon
    :return: targets and labels as torch tensor
    """
    # get temporal difference error
    delta = memory[-1].reward + parent.discount * parent.q.evaluate(
        parent.takeAction(memory[-1].nextState)) - parent.q.evaluate(memory[-1].action)

    # states
    netInput = []
    for transition in memory:
        netInput.append(
            torch.cat((transition.action.state.strengths, transition.action.state.relCoord, transition.action.changes)))

    netInput = torch.stack(netInput)

    # updates for every state in memory with respect to its eligibility
    labels = []
    for transition in memory:
        labels.append(parent.learningRate * delta * transition.action.eligibility)

    labels = torch.tensor(labels)
    labels = torch.unsqueeze(labels, 1)

    return netInput, labels


def dqn(parent, memory):
    """
    Create update targets from memory according to naive Q-learning.
    :param parent: agent object subject to update targets
    :param memory: memory to create targets upon
    :return: targets and labels as torch tensor
    """
    # sampleSize = self.memorySize // 5  # use only with traces (= short memory larger than 5 entries)
    sampleSize = 1

    if len(memory) < sampleSize:
        sample = memory
    else:
        sample = random.sample(memory, sampleSize)

    # states
    netInput = []
    for memory in sample:
        netInput.append(
            torch.cat((memory.action.state.strengths, memory.action.state.relCoord, memory.action.changes)))

    netInput = torch.stack(netInput)

    # updates for Q-values
    labels = []
    for memory in sample:
        if memory.nextState:
            labels.append(memory.reward)
        else:
            currentQ = parent.q.evaluate(memory.action)
            labels.append(currentQ + parent.learningRate * (
                    parent.discount * parent.q.evaluateMax(memory.nextState, parent.actionSet) - currentQ))

    labels = torch.tensor(labels)
    labels = torch.unsqueeze(labels, 1)

    return netInput.float(), labels.float()  # casting added due to occasional occurrence of LongTensors <- why?
