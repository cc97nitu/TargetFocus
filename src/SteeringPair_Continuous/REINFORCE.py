import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal

from SteeringPair_Continuous.AbstractAlgorithm import AbstractModel, AbstractTrainer
from SteeringPair_Continuous.SingleState_MDP import Environment, Termination, initEnvironment


class Model(AbstractModel):
    def __init__(self, PolicyNetwork, **kwargs):
        super().__init__(**kwargs)
        self.policy_net = PolicyNetwork(Environment.features, 4).to(self.device)

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
        # self.optimizer = optim.Adam(self.model.policy_net.parameters(), lr=3e-4)
        self.optimizer = optimizer(self.model.policy_net.parameters(), lr=stepSize)

        # extract hyper parameters from kwargs
        try:
            self.GAMMA = kwargs["GAMMA"]
        except KeyError as e:
            raise ValueError("Cannot read hyper parameters: {}".format(e))

    def selectAction(self, state):
        pdfParam = self.model.policy_net.forward(Variable(state)).squeeze(0)

        # rescale parameters to match action-interval's size
        pdfParam = torch.cat([pdfParam[:2] * 0.066, pdfParam[2:] * 0.066 / 3])

        # build pdf without destroying the autograd graph
        means = pdfParam[:2]
        covariances = torch.zeros(2, 2)
        covariances[0, 0] = pdfParam[2]**2
        covariances[1, 1] = pdfParam[3]**2

        pdf = MultivariateNormal(means, covariances)

        action = pdf.sample()
        logProb = pdf.log_prob(action)
        return action, logProb

    def optimizeModel(self, rewards, log_probs):
        # calculate observed returns from observed rewards
        observedReturns = []

        for t in range(len(rewards)):
            Gt = 0
            pw = 0
            for r in rewards[t:]:
                Gt = Gt + self.GAMMA ** pw * r
                pw = pw + 1
            observedReturns.append(Gt)

        observedReturns = torch.tensor(observedReturns, device=Environment.device)
        # observedReturns = (observedReturns - observedReturns.mean()) / (
        #         observedReturns.std() + 1e-9)  # normalize discounted rewards

        # calculate policy gradient
        policy_gradient = []
        for log_prob, Gt in zip(log_probs, observedReturns):
            policy_gradient.append(-log_prob * Gt)

        self.optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        policy_gradient.backward()
        self.optimizer.step()

    def trainAgent(self, num_episodes):

        # keep track of received return
        episodeReturns = list()

        # count how episodes terminate
        episodeTerminations = {"successful": 0, "failed": 0, "aborted": 0}

        # let the agent learn
        for i_episode in range(num_episodes):
            # keep track of rewards and logarithmic probabilities
            rewards, log_probs = [], []

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

                log_probs.append(log_prob)
                rewards.append(reward)
                episodeReturn += reward

                # Move to the next state
                state = nextState

            # optimize
            self.optimizeModel(rewards, log_probs)

            episodeReturns.append(torch.tensor([[episodeReturn,]]))
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
    import matplotlib.pyplot as plt

    # environment config
    envConfig = {"stateDefinition": "6d-norm", "actionSet": "A4", "rewardFunction": "propReward",
                 "acceptance": 5e-3, "targetDiameter": 3e-2, "maxStepsPerEpisode": 50, "successBounty": 10,
                 "failurePenalty": -10, "device": "cuda" if torch.cuda.is_available() else "cpu"}
    initEnvironment(**envConfig)

    # create model
    model = Model(PolicyNetwork=Network.PDF1)

    # define hyper parameters
    hyperParamsDict = {"BATCH_SIZE": 128, "GAMMA": 0.999, "TARGET_UPDATE": 10, "EPS_START": 0.5, "EPS_END": 0,
                       "EPS_DECAY": 500, "MEMORY_SIZE": int(1e4)}

    # set up trainer
    trainer = Trainer(model, torch.optim.Adam, 3e-4, **hyperParamsDict)

    # train model under hyper parameters
    episodeReturns, _ = trainer.trainAgent(2000)
    plt.plot(episodeReturns)
    plt.show()
    plt.close()

    # set up env and state for interactive mode
    env = Environment("random")
    state = env.initialState
    net = model.policy_net
    pdfParam = model.policy_net.forward(Variable(state)).squeeze(0)
    pdfParam = torch.cat([pdfParam[:2] * 0.066, pdfParam[2:] * 0.066 / 3])

    pdf = MultivariateNormal(pdfParam[:2], torch.tensor([[0.001, 0], [0, 0.001]]))
