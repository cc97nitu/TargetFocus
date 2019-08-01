import random
import math

import torch
import torch.optim
import torch.nn.functional
import matplotlib.pyplot as plt

import DQN.Struct as Struct
from DQN import Environment
import DQN.Network as Network

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# number features describing a state
numberFeatures = Environment.features
numberActions = len(Environment.actionSet)

# initialize
memory = Struct.ReplayMemory(int(1e4))

BATCH_SIZE = 128
GAMMA = 0.999
TARGET_UPDATE = 30

policy_net = Network.FC2(numberFeatures, numberActions).to(device)
target_net = Network.FC2(numberFeatures, numberActions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

EPS_START = 0.5
EPS_END = 0
EPS_DECAY = 500

# store accumulated reward
episodeReturns = []

# store epsilon at the episode's beginning
episodeEpsilon = []

fig, axes = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})


def plot_durations():
    # clear axes
    axes[0].clear()
    axes[1].clear()

    # show return on first axes
    returns_t = torch.tensor(episodeReturns, dtype=torch.float)
    axes[0].set_title('Training...')
    axes[0].set_ylabel('Return')
    axes[0].plot(returns_t.numpy(), color="orange", label="return")
    # Take 10 episode averages and plot them too
    if len(returns_t) >= 20:
        means = returns_t.unfold(0, 20, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(19), means))
        axes[0].plot(range(len(means))[20:], means.numpy()[20:], color="blue", label="mean")

    axes[0].yaxis.grid(linestyle="-")
    axes[0].legend()

    # show epsilon on second axes
    axes[1].plot(episodeEpsilon, color="black")
    x = range(0, len(episodeEpsilon))
    axes[1].fill_between(x, episodeEpsilon, color='#539ecd')

    axes[1].set_ylim(0, 1)
    axes[1].yaxis.grid()
    axes[1].set_ylabel("epsilon")
    axes[1].set_xlabel('Episode')

    fig.tight_layout()
    plt.pause(0.001)  # pause a bit so that plots are updated


stepsDone = 0


def selectAction(model, state):
    global stepsDone
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * stepsDone / EPS_DECAY)
    stepsDone += 1
    if sample > eps_threshold:  # bug in original code??
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).argmax().unsqueeze_(0).unsqueeze_(0)
    else:
        return torch.tensor([[random.randrange(numberActions)]], device=device, dtype=torch.long)

    return torch.tensor([[random.randrange(numberActions)]], device=device, dtype=torch.long)


optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-4)


def optimizeModel():
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
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = torch.nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return


# let the agent learn
num_episodes = 300
for i_episode in range(num_episodes):
    # store current epsilon
    episodeEpsilon.append(EPS_END + (EPS_START - EPS_END) * math.exp(-1. * stepsDone / EPS_DECAY))

    # Initialize the environment and state
    env = Environment(0, 0)
    state = env.initialState
    episodeReturn = 0

    episodeTerminated = False
    while not episodeTerminated:
        # Select and perform an action
        action = selectAction(policy_net, state)
        nextState, reward, episodeTerminated = env.react(action)
        episodeReturn += reward

        # Store the transition in memory
        memory.push(state, action, nextState, reward)

        # Move to the next state
        state = nextState

        # # Perform one step of the optimization (on the target network)
        optimizeModel()

    episodeReturns.append(episodeReturn)

    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
        plot_durations()

print('Complete')
plt.ioff()
plt.show()

#########
env = Environment(0, 0)
state = env.initialState

transitions = memory.sample(BATCH_SIZE)
# Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
# detailed explanation). This converts batch-array of Transitions
# to Transition of batch-arrays.
batch = Struct.Transition(*zip(*transitions))

print("episode terminations: successful {}, failed {}, aborted {}".format(*Environment.terminations.values()))
