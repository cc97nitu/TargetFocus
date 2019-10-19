import random
import math

import torch
import torch.optim
import torch.nn.functional

from SteeringPair import Struct
from SteeringPair_Stochastic import Environment, Termination, initEnvironment
from SteeringPair_Stochastic.AbstractAlgorithm import AbstractModel, AbstractTrainer


class Model(AbstractModel):
    """Class describing a model consisting of two neural networks."""

    def __init__(self, QNetwork, **kwargs):
        super().__init__(**kwargs)

        self.policy_net = QNetwork(self.numberFeatures, self.numberActions).to(self.device)
        self.target_net = QNetwork(self.numberFeatures, self.numberActions).to(self.device)
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

    def __repr__(self):
        return "QNetwork={}".format(str(self.policy_net.__class__.__name__))


class Trainer(AbstractTrainer):
    """Class used to train a model under given hyper parameters."""

    def __init__(self, model: Model, optimizer, stepSize, **kwargs):
        """
        Set up trainer.
        :param model: model to train
        :param kwargs: dictionary containing hyper parameters
        """
        super().__init__()
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

        self.model = model

        # set up replay memory
        self.memory = Struct.ReplayMemory(self.MEMORY_SIZE)

        # count steps in order to become greedier
        self.stepsDone = 0

        # define optimizer
        self.optimizer = optimizer(self.model.policy_net.parameters(), lr=stepSize)
        # self.optimizer = torch.optim.Adam(self.model.policy_net.parameters(), lr=2e-5)

        # tool to normalize states
        self.stat = Struct.RunningStat(Environment.features)

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
                self.model.eval()
                return self.model.policy_net(state).argmax().unsqueeze_(0).unsqueeze_(0)
        else:
            return torch.tensor([[random.randrange(self.model.numberActions)]], device=Environment.device, dtype=torch.long)

    def optimizeModel(self):
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
                                                batch.next_state)), device=Environment.device, dtype=torch.uint8)
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
        next_state_values = torch.zeros(self.BATCH_SIZE, device=Environment.device)
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

        # put model in evaluation mode again
        self.model.eval()
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
            while True:
                try:
                    env = Environment("random")  # no arguments => random initialization of starting point
                    break
                except ValueError:
                    continue

            state = self.stat.runningNorm(env.initialState.squeeze(0)).unsqueeze(0)
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
                state = self.stat.runningNorm(nextState.squeeze(0)).unsqueeze(0) if not nextState is None else None

                # # Perform one step of the optimization (on the target network)
                self.optimizeModel()

            episodeReturns.append(episodeReturn)
            if episodeTerminated == Termination.SUCCESSFUL:
                episodeTerminations["successful"] += 1
            elif episodeTerminated == Termination.FAILED:
                episodeTerminations["failed"] += 1
            elif episodeTerminated == Termination.ABORTED:
                episodeTerminations["aborted"] += 1

            # Update the target network
            if self.TARGET_UPDATE < 1:
                # Update the target network by applying a soft update
                policyNetDict, targetNetDict = self.model.policy_net.state_dict(), self.model.target_net.state_dict()
                for param in targetNetDict.keys():
                    targetNetDict[param] = (1 - self.TARGET_UPDATE) * targetNetDict[param] + self.TARGET_UPDATE * policyNetDict[param]

                self.model.target_net.load_state_dict(targetNetDict)
            else:
                # update by copying every parameter
                if i_episode % self.TARGET_UPDATE == 0:
                    self.model.target_net.load_state_dict(self.model.policy_net.state_dict())

            # status report
            print("episode: {}/{}".format(i_episode+1, num_episodes), end="\r")

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
            while True:
                try:
                    env = Environment("random")  # no arguments => random initialization of starting point
                    break
                except ValueError:
                    continue

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
        return episodeReturns, episodeTerminations, None  # None stands for accuracy of value function



if __name__ == "__main__":
    import torch.optim
    from SteeringPair import Network
    import matplotlib.pyplot as plt

    # environment config
    envConfig = {"stateDefinition": "6d-norm", "actionSet": "A9", "rewardFunction": "stochasticPropRewardStepPenalty",
                 "acceptance": 5e-3, "targetDiameter": 3e-2, "maxIllegalStateCount": 0, "maxStepsPerEpisode": 50,
                 "stateNoiseAmplitude": 1e-14, "rewardNoiseAmplitude": 1e-14, "successBounty": 10,
                 "failurePenalty": -10, "device": torch.device("cpu")}
    initEnvironment(**envConfig)

    # define hyper parameters
    hyperParams = {"BATCH_SIZE": 128, "GAMMA": 0.9, "TARGET_UPDATE": 10, "EPS_START": 0.5, "EPS_END": 0,
                   "EPS_DECAY": 500, "MEMORY_SIZE": int(1e4)}

    # create model
    model = Model(QNetwork=Network.FC7)

    # set up trainer
    trainer = Trainer(model, torch.optim.Adam, 3e-4, **hyperParams)

    # train model under hyper parameters
    episodeReturns, _ = trainer.trainAgent(500)

    # plot mean return
    meanSamples = 10
    episodeReturns = torch.tensor(episodeReturns)
    meanReturns = torch.empty(len(episodeReturns) - meanSamples, dtype=torch.float)
    for i in reversed(range(len(meanReturns))):
        meanReturns[i] = episodeReturns[i:i+meanSamples].sum() / meanSamples

    plt.plot(range(meanSamples, len(episodeReturns)), meanReturns.numpy())
    plt.show()
    plt.close()
