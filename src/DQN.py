import random
import math
from itertools import count, product
from collections import namedtuple

import torch
import torch.optim
import matplotlib.pyplot as plt

import Struct
import Network
from Environment import Environment

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# number features describing a state
numberFeatures = Environment.features


class ActionSet(object):
    eps_start = 0.1
    eps_end = 0.9
    eps_decay = 200

    def __init__(self):
        # possible changes in focusing strengths
        posChanges = [-1e-2, -1e-3, 0, 1e-3, 1e-2]
        posChanges = [i for i in product(posChanges, posChanges)]
        self.actions = torch.tensor(posChanges, dtype=torch.float, device=device)
        self.numDecisions = 0
        return

    def __len__(self):
        return len(self.actions)

    def select(self, model, state):
        sample = random.random()
        eps_threshold = ActionSet.eps_end + (ActionSet.eps_start - ActionSet.eps_end) * math.exp(
            -1. * self.numDecisions / ActionSet.eps_decay)
        if sample > eps_threshold:
            with torch.no_grad():
                action = model(state).argmax()
                return action
        else:
            return self.actions[random.randrange(len(self))]



# initialize
memory = Struct.ReplayMemory(int(1e4))
actionSet = ActionSet()

BATCH_SIZE = 128
GAMMA = 0.999
TARGET_UPDATE = 10

policy_net = Network.FC1(numberFeatures, len(actionSet)).to(device)
target_net = Network.FC1(numberFeatures, len(actionSet)).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# store accumulated reward
episodeReturns = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episodeReturns, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy(), color="orange", label="duration")
    # Take 10 episode averages and plot them too
    if len(durations_t) >= 10:
        means = durations_t.unfold(0, 10, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(9), means))
        plt.plot(means.numpy(), color="blue", label="mean")

    plt.legend()
    plt.pause(0.001)  # pause a bit so that plots are updated


optimizer = torch.optim.RMSprop(policy_net.parameters())

# let the agent learn
num_episodes = 200
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env = Environment(0, 0)
    state = env.initialState
    episodeReturn = 0

    episodeTerminated = False
    while not episodeTerminated:
        # Select and perform an action
        action = actionSet.select(policy_net, state)
        nextState, reward, episodeTerminated = env.react(action)
        episodeReturn += reward
        reward = torch.tensor([reward], device=device)

        # Store the transition in memory
        memory.push(state, action, nextState, reward)

        # Move to the next state
        state = nextState

        # # Perform one step of the optimization (on the target network)
        # optimize_model()

    episodeReturns.append(episodeReturn)
    plot_durations()

    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
