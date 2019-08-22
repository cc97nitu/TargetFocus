import numpy as np

import torch
import torch.optim
import torch.nn.functional
from torch.autograd import Variable

from SteeringPair import Struct
from SteeringPair import Environment, Termination, initEnvironment
from SteeringPair import Network
from SteeringPair.AbstractAlgorithm import AbstractModel, AbstractTrainer

# if gpu is to be used
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")


class Model(AbstractModel):
    """Class describing a model consisting of two neural networks."""

    def __init__(self, QNetwork, PolicyNetwork, **kwargs):
        super().__init__(**kwargs)
        self.qTrainNet = QNetwork(self.numberFeatures, self.numberActions).to(device)
        self.qTargetNet = QNetwork(self.numberFeatures, self.numberActions).to(device)
        self.qTargetNet.load_state_dict(self.qTrainNet.state_dict())
        self.qTargetNet.eval()

        self.policyNet = PolicyNetwork(self.numberFeatures, self.numberActions).to(device)
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

    def __repr__(self):
        return "QNetwork={}, PolicyNetwork={}".format(str(self.qTrainNet.__class__.__name__), str(self.policyNet.__class__.__name__))



class Trainer(AbstractTrainer):
    """Class used to train a model under given hyper parameters."""

    def __init__(self, model: Model, **kwargs):
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
        self.optimizerQTrain = torch.optim.Adam(self.model.qTrainNet.parameters(), lr=2e-5)
        self.optimizerPolicy = torch.optim.Adam(self.model.policyNet.parameters(), lr=3e-4)
        return

    def selectAction(self, state):
        log_probs = self.model.policyNet.forward(Variable(state))
        probs = torch.exp(log_probs)
        selectedAction = np.random.choice(len(Environment.actionSet), p=np.squeeze(probs.detach().numpy()))
        log_prob = log_probs.squeeze(0)[selectedAction]
        selectedAction = torch.tensor([selectedAction], dtype=torch.long, device=device)
        return selectedAction, log_prob

    def optimizeQTrainNet(self):
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
        action_batch = torch.cat(batch.action).unsqueeze(1)
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
        next_state_values[non_final_mask] = self.model.qTrainNet(non_final_next_states).max(1)[0].detach()

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

    def optimizePolicy(self, states, actions, log_probs):
        # get Q-values
        states = torch.cat(states)
        actions = torch.cat(actions).unsqueeze(1)
        qValues = self.model.qTrainNet(states).gather(1, actions)
        qValues.detach_()

        # calculate policy gradient
        log_probs = torch.stack(log_probs)
        policy_gradient = -1 * log_probs * qValues.squeeze(1)

        self.optimizerPolicy.zero_grad()
        policy_gradient = policy_gradient.sum()
        policy_gradient.backward()
        self.optimizerPolicy.step()

    def trainAgent(self, num_episodes):
        # keep track of received return
        episodeReturns = []

        # count how episodes terminate
        episodeTerminations = {"successful": 0, "failed": 0, "aborted": 0}

        # let the agent learn
        for i_episode in range(num_episodes):
            # keep track of states, selected actions and logarithmic probabilities
            states, selectedActions, log_probs = [], [], []

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
                selectedActions.append(action)
                log_probs.append(log_prob)
                episodeReturn += reward

                # Move to the next state
                state = nextState

                # optimize Q-values
                self.optimizeQTrainNet()

            # update policy
            self.optimizePolicy(states, selectedActions, log_probs)

            # Update the target network, copying all weights and biases in SteeringPair
            if i_episode % self.TARGET_UPDATE == 0:
                self.model.qTargetNet.load_state_dict(self.model.qTrainNet.state_dict())

            episodeReturns.append(episodeReturn)
            if episodeTerminated == Termination.SUCCESSFUL:
                episodeTerminations["successful"] += 1
            elif episodeTerminated == Termination.FAILED:
                episodeTerminations["failed"] += 1
            elif episodeTerminated == Termination.ABORTED:
                episodeTerminations["aborted"] += 1

            # status report
            print("episode: {}/{}".format(i_episode+1, num_episodes), end="\r")

        print("Complete")
        # plt.plot(episodeReturns)
        # plt.show()
        # plt.close()
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
        return episodeReturns, episodeTerminations


if __name__ == "__main__":
    # environment config
    envConfig = {"stateDefinition": "6d-norm", "actionSet": "A4", "rewardFunction": "propReward",
                 "acceptance": 5e-3, "targetDiameter": 3e-2, "maxStepsPerEpisode": 50, "successBounty": 10,
                 "failurePenalty": -10, "device": "cuda" if torch.cuda.is_available() else "cpu"}
    initEnvironment(**envConfig)

    # create model
    model = Model()

    # define hyper parameters
    hyperParamsDict = {"BATCH_SIZE": 128, "GAMMA": 0.999, "TARGET_UPDATE": 10, "EPS_START": 0.5, "EPS_END": 0,
                       "EPS_DECAY": 500, "MEMORY_SIZE": int(1e4)}

    # set up trainer
    trainer = Trainer(model, **hyperParamsDict)

    # train model under hyper parameters
    trainer.trainAgent(500)

    _, terminations = trainer.benchAgent(50)
    print(terminations)







