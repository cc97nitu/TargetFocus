import numpy as np

import torch
import torch.nn.functional
from torch.autograd import Variable

from SteeringPair import Struct
from SteeringPair import Environment, Termination, initEnvironment
from SteeringPair.AbstractAlgorithm import AbstractModel, AbstractTrainer


class Model(AbstractModel):
    """Class describing a model consisting of two neural networks."""

    def __init__(self, QNetwork, PolicyNetwork, **kwargs):
        super().__init__(**kwargs)
        self.vTrainNet = QNetwork(self.numberFeatures, 1).to(self.device)
        self.vTargetNet = QNetwork(self.numberFeatures, 1).to(self.device)
        self.vTargetNet.load_state_dict(self.vTrainNet.state_dict())
        self.vTargetNet.eval()

        self.policyNet = PolicyNetwork(self.numberFeatures, self.numberActions).to(self.device)
        return

    def to_dict(self):
        return {"vTrainNet_state_dict": self.vTrainNet.state_dict(),
                "vTargetNet_state_dict": self.vTargetNet.state_dict(),
                "policyNet_state_dict": self.policyNet.state_dict()}

    def load_state_dict(self, dictionary: dict):
        try:
            self.vTrainNet.load_state_dict(dictionary["vTrainNet_state_dict"])
            self.vTargetNet.load_state_dict(dictionary["vTargetNet_state_dict"])
            self.policyNet.load_state_dict(dictionary["policyNet_state_dict"])
        except KeyError as e:
            raise ValueError("missing state_dict: {}".format(e))

    def eval(self):
        self.vTrainNet.eval()
        self.vTargetNet.eval()
        self.policyNet.eval()

    def train(self):
        self.vTrainNet.train()
        self.vTargetNet.eval()  ## target net is never trained but updated by copying weights
        self.policyNet.train()

    def __repr__(self):
        return "VNetwork={}, PolicyNetwork={}".format(str(self.vTrainNet.__class__.__name__),
                                                      str(self.policyNet.__class__.__name__))


class Trainer(AbstractTrainer):
    """Class used to train a model under given hyper parameters."""

    def __init__(self, model: Model, optimizer, stepSize, **kwargs):
        """
        Set up trainer.
        :param model: model to train
        :param kwargs: dictionary containing hyper parameters
        """
        super().__init__()

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
        self.optimizerVTrain = optimizer(self.model.vTrainNet.parameters(), lr=stepSize)
        self.optimizerPolicy = optimizer(self.model.policyNet.parameters(), lr=stepSize)
        return

    def selectAction(self, state):
        log_probs = self.model.policyNet.forward(Variable(state))
        probs = torch.exp(log_probs)
        selectedAction = np.random.choice(len(Environment.actionSet), p=np.squeeze(probs.detach().numpy()))
        log_prob = log_probs.squeeze(0)[selectedAction]
        selectedAction = torch.tensor([selectedAction], dtype=torch.long, device=Environment.device)
        return selectedAction, log_prob

    def optimizePolicy(self, states, nextStates, rewards, log_probs):
        # catch episodes consisting of one single step
        if len(states) == 1:
            advantageValue = rewards[0] - self.model.vTrainNet(states[0]).squeeze(1).detach()
            # calculate policy gradient
            log_probs = torch.stack(log_probs)
            policy_gradient = -1 * log_probs * advantageValue

            self.optimizerPolicy.zero_grad()
            policy_gradient = policy_gradient.sum()
            policy_gradient.backward()
            self.optimizerPolicy.step()
            return

        # concatenate lists to tensors
        states = torch.cat(states)
        rewards = torch.cat(rewards)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                nextStates)), device=Environment.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in nextStates if s is not None])

        # calculate state-values
        stateValues = self.model.vTrainNet(states).squeeze_(1)
        nextStateValues = torch.zeros(len(states), device=Environment.device)
        nextStateValues[non_final_mask] = self.model.vTrainNet(non_final_next_states).squeeze_(1)

        # calculate advantage-values
        advantageValues = rewards + (self.GAMMA * nextStateValues) - stateValues
        advantageValues.detach_()

        # calculate policy gradient
        log_probs = torch.stack(log_probs)
        policy_gradient = -1 * log_probs * advantageValues

        self.optimizerPolicy.zero_grad()
        policy_gradient = policy_gradient.sum()
        policy_gradient.backward()
        self.optimizerPolicy.step()

    def optimizeVTrainNet(self):
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
        states = torch.cat(batch.state)
        rewards = torch.cat(batch.reward)

        # get current state-values
        stateValues = self.model.vTrainNet(states)

        # get next state's state-values
        nextStateValues = torch.zeros((self.BATCH_SIZE, 1), device=Environment.device)
        nextStateValues[non_final_mask] = self.model.vTargetNet(non_final_next_states)

        # Compute the expected Q values
        expectedStateValues = (nextStateValues * self.GAMMA) + rewards

        # Compute Huber loss
        loss = torch.nn.functional.smooth_l1_loss(stateValues, expectedStateValues)

        # Optimize the model
        self.optimizerVTrain.zero_grad()
        loss.backward()
        for param in self.model.vTrainNet.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizerVTrain.step()

        # put model in evaluation mode again
        self.model.eval()
        return

    def trainAgent(self, num_episodes):
        # keep track of received return
        episodeReturns = []

        # count how episodes terminate
        episodeTerminations = {"successful": 0, "failed": 0, "aborted": 0}

        # let the agent learn
        for i_episode in range(num_episodes):
            # keep track of states, selected actions and logarithmic probabilities
            states, nextStates, rewards, log_probs = list(), list(), list(), list()

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
                action, log_prob = self.selectAction(state)
                nextState, reward, episodeTerminated = env.react(action)

                # Store the transition in memory
                self.memory.push(state, action, nextState, reward)

                # log
                states.append(state)
                nextStates.append(nextState)
                rewards.append(reward)
                log_probs.append(log_prob)
                episodeReturn += reward

                # Move to the next state
                state = nextState

                # optimize Q-values
                self.optimizeVTrainNet()

            # update policy
            self.optimizePolicy(states, nextStates, rewards, log_probs)

            # Update the target network, copying all weights and biases in SteeringPair
            if i_episode % self.TARGET_UPDATE == 0:
                self.model.vTargetNet.load_state_dict(self.model.vTrainNet.state_dict())

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

            state = env.initialState
            episodeReturn = 0

            episodeTerminated = Termination.INCOMPLETE
            while episodeTerminated == Termination.INCOMPLETE:
                # Select and perform an action
                action, _ = self.selectAction(state)
                nextState, reward, episodeTerminated = env.react(action)
                episodeReturn += reward

                # Move to the next state
                state = nextState

            episodeReturns.append(torch.tensor([[episodeReturn,]]))
            if episodeTerminated == Termination.SUCCESSFUL:
                episodeTerminations["successful"] += 1
            elif episodeTerminated == Termination.FAILED:
                episodeTerminations["failed"] += 1
            elif episodeTerminated == Termination.ABORTED:
                episodeTerminations["aborted"] += 1

        print("Complete")
        return episodeReturns, episodeTerminations


if __name__ == "__main__":
    import torch.optim
    from SteeringPair import Network

    # environment config
    envConfig = {"stateDefinition": "6d-norm", "actionSet": "A4", "rewardFunction": "propReward",
                 "acceptance": 5e-3, "targetDiameter": 3e-2, "maxStepsPerEpisode": 50, "successBounty": 10,
                 "failurePenalty": -10, "device": "cuda" if torch.cuda.is_available() else "cpu"}
    initEnvironment(**envConfig)

    # create model
    model = Model(QNetwork=Network.FC7, PolicyNetwork=Network.Cat3)

    # define hyper parameters
    hyperParamsDict = {"BATCH_SIZE": 128, "GAMMA": 0.999, "TARGET_UPDATE": 10, "EPS_START": 0.5, "EPS_END": 0,
                       "EPS_DECAY": 500, "MEMORY_SIZE": int(1e4)}

    # set up trainer
    trainer = Trainer(model, torch.optim.Adam, 3e-4, **hyperParamsDict)

    # train model under hyper parameters
    trainer.trainAgent(500)
