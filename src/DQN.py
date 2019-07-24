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
    eps_decay = 1000

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
        if sample > 1 - eps_threshold:  # bug in original code??
            with torch.no_grad():
                actionIndex = model(state).argmax()
                return self.actions[actionIndex]
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
    returns_t = torch.tensor(episodeReturns, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(returns_t.numpy(), color="orange", label="return")
    # Take 10 episode averages and plot them too
    if len(returns_t) >= 10:
        means = returns_t.unfold(0, 10, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(9), means))
        plt.plot(range(len(means))[10:], means.numpy()[10:], color="blue", label="mean")

    plt.legend()
    plt.pause(0.001)  # pause a bit so that plots are updated


optimizer = torch.optim.RMSprop(policy_net.parameters())


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Struct.Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    # state_action_values = policy_net(state_batch).gather(1, action_batch)


    return action_batch


# let the agent learn
num_episodes = 30
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
        optimize_model()

    episodeReturns.append(episodeReturn)

    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
        plot_durations()

