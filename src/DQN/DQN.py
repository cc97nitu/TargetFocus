import random
import math

import torch
import torch.optim
import torch.nn.functional
import matplotlib.pyplot as plt

from DQN import Struct
from DQN import Environment, Termination
from DQN import Network

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# number features describing a state
numberFeatures = Environment.features
numberActions = len(Environment.actionSet)


class Model(object):
    """Class describing a model consisting of two neural networks."""

    def __init__(self):
        self.policy_net = Network.FC4(numberFeatures, numberActions).to(device)
        self.target_net = Network.FC4(numberFeatures, numberActions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        return

    def to_dict(self):
        return {"policy_net_state_dict": self.policy_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict()}

    def load_state_dict(self, dictionary: dict):
        try:
            self.policy_net.load_state_dict(dictionary["policy_net_state_dict"])
            self.target_net.load_state_dict(dictionary["target_net_state_dict"])
        except KeyError as e:
            raise ValueError("missing state_dict: {}".format(e))

    def eval(self):
        self.policy_net.eval()
        self.target_net.eval()

    def train(self):
        self.policy_net.train()
        self.target_net.eval()  ## target net is never trained but updated by copying weights


class Trainer(object):
    """Class used to train a model under given hyper parameters."""

    def __init__(self, model: Model, **kwargs):
        """
        Set up trainer.
        :param model: model to train
        :param kwargs: dictionary containing hyper parameters
        """
        self.model = model

        # extract hyper parameters from kwargs
        try:
            self.BATCH_SIZE = kwargs["BATCH_SIZE"]
            self.GAMMA = kwargs["GAMMA"]
            self.TARGET_UPDATE = kwargs["TARGET_UPDATE"]
            self.EPS_START = kwargs["EPS_START"]
            self.EPS_END = kwargs["EPS_END"]
            self.EPS_DECAY = kwargs["EPS_DECAY"]
            self.MEMORY_SIZE = kwargs["MEMORY_SIZE"]
        except KeyError as e:
            raise ValueError("Cannot read hyper parameters: {}".format(e))

        # set up replay memory
        self.memory = Struct.ReplayMemory(self.MEMORY_SIZE)

        # count steps in order to become greedier
        self.stepsDone = 0

        # define optimizer
        self.optimizer = torch.optim.Adam(self.model.policy_net.parameters(), lr=1e-4)
        return

    def selectAction(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.stepsDone / self.EPS_DECAY)
        self.stepsDone += 1
        if sample > eps_threshold:  # bug in original code??
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.model.policy_net(state).argmax().unsqueeze_(0).unsqueeze_(0)
        else:
            return torch.tensor([[random.randrange(numberActions)]], device=device, dtype=torch.long)

    def optimizeModel(self):
        if len(self.memory) < self.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.BATCH_SIZE)
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
        state_action_values = self.model.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.model.target_net(non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        loss = torch.nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return

    def trainAgent(self, num_episodes):
        # keep track of epsilon and received return
        episodeEpsilon, episodeReturns = [], []

        # count how episodes terminate
        episodeTerminations = {"successful": 0, "failed": 0, "aborted": 0}

        # let the agent learn
        for i_episode in range(num_episodes):
            # store current epsilon
            episodeEpsilon.append(
                self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.stepsDone / self.EPS_DECAY))

            # Initialize the environment and state
            env = Environment()  # no arguments => random initialization of starting point
            state = env.initialState
            episodeReturn = 0

            episodeTerminated = Termination.INCOMPLETE
            while episodeTerminated == Termination.INCOMPLETE:
                # Select and perform an action
                action = self.selectAction(state)
                nextState, reward, episodeTerminated = env.react(action)
                episodeReturn += reward

                # Store the transition in memory
                self.memory.push(state, action, nextState, reward)

                # Move to the next state
                state = nextState

                # # Perform one step of the optimization (on the target network)
                self.optimizeModel()

            episodeReturns.append(episodeReturn)
            if episodeTerminated == Termination.SUCCESSFUL:
                episodeTerminations["successful"] += 1
            elif episodeTerminated == Termination.FAILED:
                episodeTerminations["failed"] += 1
            elif episodeTerminated == Termination.ABORTED:
                episodeTerminations["aborted"] += 1

            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.TARGET_UPDATE == 0:
                self.model.target_net.load_state_dict(self.model.policy_net.state_dict())

        print("Complete")
        return episodeReturns, episodeTerminations

    def benchAgent(self, num_episodes):
        # keep track of epsilon and received return
        episodeReturns = []

        # count how episodes terminate
        episodeTerminations = {"successful": 0, "failed": 0, "aborted": 0}

        # episodes
        for i_episode in range(num_episodes):
            # Initialize the environment and state
            env = Environment(0, 0)
            state = env.initialState
            episodeReturn = 0

            episodeTerminated = Termination.INCOMPLETE
            while episodeTerminated == Termination.INCOMPLETE:
                # Select and perform an action
                action = self.selectAction(state)
                nextState, reward, episodeTerminated = env.react(action)
                episodeReturn += reward

                # Move to the next state
                state = nextState

            episodeReturns.append(episodeReturn)
            if episodeTerminated == Termination.SUCCESSFUL:
                episodeTerminations["successful"] += 1
            elif episodeTerminated == Termination.FAILED:
                episodeTerminations["failed"] += 1
            elif episodeTerminated == Termination.ABORTED:
                episodeTerminations["aborted"] += 1


        print("Complete")
        return episodeReturns, episodeTerminations



if __name__ == "__main__":
    # create model
    model = Model()

    # define hyper parameters
    hyperParamsDict = {"BATCH_SIZE": 128, "GAMMA": 0.999, "TARGET_UPDATE": 10, "EPS_START": 0.5, "EPS_END": 0,
                       "EPS_DECAY": 500, "MEMORY_SIZE": int(1e4)}

    # set up trainer
    trainer = Trainer(model, **hyperParamsDict)

    # train model under hyper parameters
    trainer.trainAgent(5)
