import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable

import SteeringPair.Network as Network
from SteeringPair import Environment, Termination


class Model(object):
    def __init__(self):
        self.policy_net = Network.Cat1(Environment.features, len(Environment.actionSet))
        # self.policy_net = PolicyNetwork(Environment.features, len(Environment.actionSet), 128)

    def to_dict(self):
        return {"policy_net_state_dict": self.policy_net.state_dict(),}

    def load_state_dict(self, dictionary: dict):
        try:
            self.policy_net.load_state_dict(dictionary["policy_net_state_dict"])
        except KeyError as e:
            raise ValueError("missing state_dict: {}".format(e))

    def eval(self):
        self.policy_net.eval()

    def train(self):
        self.policy_net.train()


class Trainer(object):
    def __init__(self, model, **kwargs):
        self.model = model
        self.optimizer = optim.Adam(self.model.policy_net.parameters(), lr=3e-4)

        # extract hyper parameters from kwargs
        try:
            self.GAMMA = kwargs["GAMMA"]
        except KeyError as e:
            raise ValueError("Cannot read hyper parameters: {}".format(e))

    def selectAction(self, state):
        log_probs = self.model.policy_net.forward(Variable(state))
        probs = torch.exp(log_probs)
        highest_prob_action = np.random.choice(len(Environment.actionSet), p=np.squeeze(probs.detach().numpy()))
        # log_prob = torch.log(log_probs.squeeze(0)[highest_prob_action])
        log_prob = log_probs.squeeze(0)[highest_prob_action]
        highest_prob_action = torch.tensor([highest_prob_action], dtype=torch.long)
        return highest_prob_action, log_prob

    def optimizeModel(self, rewards, log_probs):
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

        policy_gradient = []
        for log_prob, Gt in zip(log_probs, observedReturns):
            policy_gradient.append(-log_prob * Gt)

        self.optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        policy_gradient.backward()
        self.optimizer.step()

    def trainAgent(self, num_episodes):

        # keep track of received return
        episodeReturns = []

        # count how episodes terminate
        episodeTerminations = {"successful": 0, "failed": 0, "aborted": 0}

        # let the agent learn
        for i_episode in range(num_episodes):
            # keep track of rewards and logarithmic probabilities
            rewards, log_probs = [], []

            # Initialize the environment and state
            env = Environment()  # no arguments => random initialization of starting point
            state = env.initialState
            episodeReturn = 0

            episodeTerminated = Termination.INCOMPLETE
            while episodeTerminated == Termination.INCOMPLETE:
                # Select and perform an action
                action, log_prob = self.selectAction(state)
                nextState, reward, episodeTerminated = env.react(action)

                log_probs.append(log_prob)
                rewards.append(reward)
                episodeReturn += reward

                # Move to the next state
                state = nextState

            # optimize
            self.optimizeModel(rewards, log_probs)

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
        # keep track of received return
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
                action, _ = self.selectAction(state)
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
    model = Model()
    train = Trainer(model, **{"GAMMA": 0.999})
    train.trainAgent(400)
    _, terminations = train.benchAgent(50)
    print(terminations)