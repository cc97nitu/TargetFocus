import random
import math
import numpy as np

import torch
import torch.optim
import torch.nn.functional
from torch.autograd import Variable

from SteeringPair import Struct
from SteeringPair import Environment, Termination
from SteeringPair import Network

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# number features describing a state
numberFeatures = Environment.features
numberActions = len(Environment.actionSet)


class Model(object):
    """Class describing a model consisting of two neural networks."""

    def __init__(self):
        self.qTrainNet = Network.FC7(numberFeatures, numberActions).to(device)
        self.qTargetNet = Network.FC7(numberFeatures, numberActions).to(device)
        self.qTargetNet.load_state_dict(self.qTrainNet.state_dict())
        self.qTargetNet.eval()

        self.policyNet = Network.Cat1(numberFeatures, numberActions).to(device)
        return

    def to_dict(self):
        return {"qTrainNet_state_dict": self.qTrainNet.state_dict(),
                "qTargetNet_state_dict": self.qTargetNet.state_dict(),
                "policyNet_state_dict": self.policyNet.state_dict()}

    def load_state_dict(self, dictionary: dict):
        try:
            self.qTrainNet.load_state_dict(dictionary["qTrainNet_state_dict"])
            self.qTargetNet.load_state_dict(dictionary["qTargetNet_state_dict"])
            self.policyNet.load_state_dict(dictionary["policyNet_state_dict"])
        except KeyError as e:
            raise ValueError("missing state_dict: {}".format(e))

    def eval(self):
        self.qTrainNet.eval()
        self.qTargetNet.eval()
        self.policyNet.eval()

    def train(self):
        self.qTrainNet.train()
        self.qTargetNet.eval()  ## target net is never trained but updated by copying weights
        self.policyNet.train()


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
            self.MEMORY_SIZE = kwargs["MEMORY_SIZE"]
        except KeyError as e:
            raise ValueError("Cannot read hyper parameters: {}".format(e))

        # set up replay memory
        self.memory = Struct.ReplayMemory(self.MEMORY_SIZE)

        # define optimizer
        self.optimizerQTrain = torch.optim.Adam(self.model.qTrainNet.parameters(), lr=2e-5)
        self.optimizerPolicy = torch.optim.Adam(self.model.policyNet.parameters(), lr=3e-4)
        return

    def selectAction(self, state):
        log_probs = self.model.policyNet.forward(Variable(state))
        probs = torch.exp(log_probs)
        highest_prob_action = np.random.choice(len(Environment.actionSet), p=np.squeeze(probs.detach().numpy()))
        log_prob = log_probs.squeeze(0)[highest_prob_action]
        highest_prob_action = torch.tensor([highest_prob_action], dtype=torch.long)
        return highest_prob_action, log_prob

    def optimizeModel(self, rewards, log_probs):
        ### optimize policy at first ###
        # calculate observed returns from observed rewards
        observedReturns = []

        for t in range(len(rewards)):
            Gt = 0
            pw = 0
            for r in rewards[t:]:
                Gt = Gt + self.GAMMA ** pw * r
                pw = pw + 1
            observedReturns.append(Gt)

        observedReturns = torch.tensor(observedReturns)
        # observedReturns = (observedReturns - observedReturns.mean()) / (
        #         observedReturns.std() + 1e-9)  # normalize discounted rewards

        # calculate policy gradient
        policy_gradient = []
        for log_prob, Gt in zip(log_probs, observedReturns):
            policy_gradient.append(-log_prob * Gt)

        self.optimizerPolicy.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        policy_gradient.backward()
        self.optimizerPolicy.step()

        ### optimize Q-function ###
        if len(self.memory) < self.BATCH_SIZE:
            return

        # put model in training mode
        self.model.train()

        # sample from replay memory
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
        state_action_values = self.model.qTrainNet(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.model.qTargetNet(non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        loss = torch.nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizerQTrain.zero_grad()
        loss.backward()
        for param in self.model.qTrainNet.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizerQTrain.step()

        # put model in evaluation mode again
        self.model.eval()
        return







