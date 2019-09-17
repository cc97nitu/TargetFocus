"""
Agent acting by randomly choosing an action. All actions are obtained from an uniform distribution. This algorithm is resistant to learning.
"""

import numpy as np
import torch

import SteeringPair.Network as Network
from QuadLens import Environment, Termination, initEnvironment
from QuadLens.AbstractAlgorithm import AbstractModel, AbstractTrainer


class Model(AbstractModel):
    def __init__(self, PolicyNetwork, **kwargs):
        super().__init__(**kwargs)
        self.policy_net = PolicyNetwork(Environment.features, len(Environment.actionSet)).to(self.device)

    def to_dict(self):
        return {"policy_net_state_dict": self.policy_net.state_dict(), }

    def load_state_dict(self, dictionary: dict):
        try:
            self.policy_net.load_state_dict(dictionary["policy_net_state_dict"])
        except KeyError as e:
            raise ValueError("missing state_dict: {}".format(e))

    def eval(self):
        self.policy_net.eval()

    def train(self):
        self.policy_net.train()

    def __repr__(self):
        return "PolicyNetwork={}".format(str(self.policy_net.__class__.__name__))


class Trainer(AbstractTrainer):
    def __init__(self, model, optimizer, stepSize, **kwargs):
        super().__init__()

        self.model = model

    def selectAction(self, state):
        actionIndex = np.random.choice(len(Environment.actionSet))
        return torch.tensor([actionIndex])

    def trainAgent(self, num_episodes):

        # keep track of received return
        episodeReturns = []

        # count how episodes terminate
        episodeTerminations = {"successful": 0, "failed": 0, "aborted": 0}

        # let the agent learn
        for i_episode in range(num_episodes):
            # keep track of rewards and logarithmic probabilities
            rewards = list()

            # Initialize the environment and state
            while True:
                try:
                    env = Environment("random")  # no arguments => random initialization of starting point
                    break
                except ValueError:
                    continue

            state = env.state
            episodeReturn = 0

            episodeTerminated = Termination.INCOMPLETE
            while episodeTerminated == Termination.INCOMPLETE:
                # Select and perform an action
                action = self.selectAction(state)
                nextState, reward, episodeTerminated = env.react(action)

                rewards.append(reward)
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

            # status report
            print("episode: {}/{}".format(i_episode + 1, num_episodes), end="\r")

        print("Complete")
        return episodeReturns, episodeTerminations

    def benchAgent(self, num_episodes):
        # keep track of received return
        episodeReturns = []

        # count how episodes terminate
        episodeTerminations = {"successful": 0, "failed": 0, "aborted": 0}

        # episodes
        for i_episode in range(num_episodes):
            # Initialize the environment and state
            while True:
                try:
                    env = Environment("random")  # no arguments => random initialization of starting point
                    break
                except ValueError:
                    continue

            state = env.state
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
        return episodeReturns, episodeTerminations, None  # None stands for accuracy of value function


if __name__ == "__main__":
    import torch.optim

    # environment config
    envConfig = {"stateDefinition": "RAW_16", "actionSet": "A9", "rewardFunction": "propRewardStepPenalty",
                 "acceptance": 2e-2, "targetDiameter": 3e-2, "maxIllegalStateCount": 0, "maxStepsPerEpisode": 50,
                 "successBounty": 10,
                 "failurePenalty": -10, "device": torch.device("cpu")}
    initEnvironment(**envConfig)

    model = Model(PolicyNetwork=Network.Cat1)
    train = Trainer(model, torch.optim.Adam, 3e-4, **{"GAMMA": 0.999})
    train.trainAgent(1)
    _, terminations, _ = train.benchAgent(100)
    print(terminations)
