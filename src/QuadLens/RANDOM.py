"""
Agent acting by randomly choosing an action. All actions are obtained from an uniform distribution. This algorithm is resistant to learning.
"""

import numpy as np
import torch

from QuadLens import Network, Environment, Termination, initEnvironment
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
    def __init__(self, model: Model, optimizer, stepSize, **kwargs):
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
            # status report
            print("episode: {}/{}".format(i_episode + 1, num_episodes), end="\r")

            # keep track of rewards and logarithmic probabilities
            rewards = list()

            # Initialize the environment and state
            while True:
                try:
                    env = Environment()
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

        print("Complete")
        return episodeReturns, episodeTerminations

    def benchAgent(self, num_episodes):
        # keep track of received return
        episodeReturns = []

        # count how episodes terminate
        episodeTerminations = {"successful": 0, "failed": 0, "aborted": 0}

        # episodes
        for i_episode in range(num_episodes):
            # status report
            print("episode: {}/{}".format(i_episode + 1, num_episodes), end="\r")

            # Initialize the environment and state
            while True:
                try:
                    env = Environment()
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
    from SteeringPair import Network
    import matplotlib.pyplot as plt

    # environment config
    envConfig = {"stateDefinition": "RAW_16", "actionSet": "A9", "rewardFunction": "propRewardStepPenalty",
                 "acceptance": 1e-3, "targetDiameter": 3e-2, "maxIllegalStateCount": 0, "maxStepsPerEpisode": 50,
                 "successBounty": 10,
                 "failurePenalty": -10, "device": torch.device("cpu")}
    initEnvironment(**envConfig)

    # define hyper parameters
    hyperParams = {"BATCH_SIZE": 128, "GAMMA": 0.9, "TARGET_UPDATE": 10, "EPS_START": 0.5, "EPS_END": 0,
                   "EPS_DECAY": 500, "MEMORY_SIZE": int(1e4)}



    # set up trainer
    model = Model(PolicyNetwork=Network.Cat1)
    trainer = Trainer(model, torch.optim.Adam, 3e-4, **hyperParams)

    # train model under hyper parameters
    episodeReturns, _ = trainer.trainAgent(1000)

    # plot mean return
    meanSamples = 20
    episodeReturns = torch.tensor(episodeReturns)
    meanReturns = torch.empty(len(episodeReturns) - meanSamples, dtype=torch.float)
    for i in reversed(range(len(meanReturns))):
        meanReturns[i] = episodeReturns[i:i+meanSamples].sum() / meanSamples

    plt.plot(range(meanSamples, len(episodeReturns)), meanReturns.numpy())
    plt.show()
    plt.close()

    trainer.benchAgent(50)

