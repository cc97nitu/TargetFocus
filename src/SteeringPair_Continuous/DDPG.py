import torch
import torch.nn.functional
import torch.optim as optim

from SteeringPair_Continuous.AbstractAlgorithm import AbstractModel, AbstractTrainer
from SteeringPair_Continuous.Ornstein_Uhlenbeck import OUNoise
from SteeringPair_Continuous.Environment import Environment, Termination, initEnvironment
from SteeringPair import Struct


class Model(AbstractModel):
    """Class describing a model consisting of two neural networks."""

    def __init__(self, QNetwork, PolicyNetwork, **kwargs):
        super().__init__(**kwargs)
        self.qTrainNet = QNetwork(self.numberFeatures + 2, 1).to(self.device)
        self.qTargetNet = QNetwork(self.numberFeatures + 2, 1).to(self.device)
        self.qTargetNet.load_state_dict(self.qTrainNet.state_dict())
        self.qTargetNet.eval()

        self.policyTrainNet = PolicyNetwork(self.numberFeatures, 2).to(self.device)
        self.policyTargetNet = PolicyNetwork(self.numberFeatures, 2).to(self.device)
        self.policyTargetNet.load_state_dict(self.policyTrainNet.state_dict())
        self.policyTargetNet.eval()
        return

    def to_dict(self):
        return {"qTrainNet_state_dict": self.qTrainNet.state_dict(),
                "qTargetNet_state_dict": self.qTargetNet.state_dict(),
                "policyTrainNet_state_dict": self.policyTrainNet.state_dict(),
                "policyTargetNet_state_dict": self.policyTargetNet.state_dict()}

    def load_state_dict(self, dictionary: dict):
        try:
            self.qTrainNet.load_state_dict(dictionary["qTrainNet_state_dict"])
            self.qTargetNet.load_state_dict(dictionary["qTargetNet_state_dict"])
            self.policyTrainNet.load_state_dict(dictionary["policyTrainNet_state_dict"])
            self.policyTargetNet.load_state_dict(dictionary["policyTargetNet_state_dict"])
        except KeyError as e:
            raise ValueError("missing state_dict: {}".format(e))

    def eval(self):
        self.qTrainNet.eval()
        self.qTargetNet.eval()
        self.policyTrainNet.eval()
        self.policyTargetNet.eval()

    def train(self):
        self.qTrainNet.train()
        self.qTargetNet.eval()  ## target net is never trained but updated by copying weights
        self.policyTrainNet.train()
        self.policyTargetNet.eval()  ## target net is never trained but updated by copying weights

    def __repr__(self):
        return "QNetwork={}, PolicyNetwork={}".format(str(self.qTrainNet.__class__.__name__),
                                                      str(self.policyTrainNet.__class__.__name__))


class Trainer(AbstractTrainer):
    def __init__(self, model: Model, optimizer, stepSize, **kwargs):
        super().__init__()

        self.model = model
        self.optimizerQTrainNet = optimizer(self.model.qTrainNet.parameters(), lr=stepSize)
        self.optimizerPolicyTrainNet = optimizer(self.model.policyTrainNet.parameters(), lr=stepSize)

        # extract hyper parameters from kwargs
        try:
            self.GAMMA = kwargs["GAMMA"]
            self.TARGET_UPDATE = kwargs["TARGET_UPDATE"]
            self.BATCH_SIZE = kwargs["BATCH_SIZE"]
            self.MEMORY_SIZE = kwargs["MEMORY_SIZE"]
        except KeyError as e:
            raise ValueError("Cannot read hyper parameters: {}".format(e))

        # noise generator for exploration
        self.noise = OUNoise(-1, 1, dim=2)

        # set up replay memory
        self.memory = Struct.ReplayMemory(self.MEMORY_SIZE)

    def selectAction(self, state):
        # action chosen by the deterministic policy
        action = self.model.policyTrainNet(state).squeeze(0)

        # add noise from Ornstein-Uhlenbeck process
        action = self.noise(action)

        # rescale to action interval
        action = 0.066 * action

        return action

    def optimizeModel(self):
        # sample from memory
        if len(self.memory) < self.BATCH_SIZE:
            return

        batch = self.memory.sample(self.BATCH_SIZE)
        batch = Struct.Transition(*zip(*batch))

        stateBatch = torch.cat(batch.state).detach()
        actionBatch = torch.stack(batch.action).detach()
        rewardBatch = torch.cat(batch.reward).squeeze(1).detach()

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=Environment.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None]).detach()

        # put model in training mode
        self.model.train()

        # compute next actions and their q-values
        nextActions = self.model.policyTargetNet(non_final_next_states)
        nextStateActionValues = torch.zeros(self.BATCH_SIZE, dtype=torch.float, device=Environment.device)
        nextStateActionValues[non_final_mask] = self.model.qTargetNet(
            torch.cat((non_final_next_states, nextActions), dim=1)).squeeze(1)

        # compute expected q-values via the Bellman-equation
        expectedStateActionValues = rewardBatch + self.GAMMA * nextStateActionValues
        expectedStateActionValues.detach_()

        # compute q-values
        stateActionValues = self.model.qTrainNet(torch.cat((stateBatch, actionBatch), dim=1))

        # Compute Huber loss
        qLoss = torch.nn.functional.smooth_l1_loss(stateActionValues, expectedStateActionValues)

        # Optimize the model
        self.optimizerQTrainNet.zero_grad()

        qLoss.backward()
        for param in self.model.qTrainNet.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizerQTrainNet.step()

        # update policy network
        statesAndActions = torch.cat((stateBatch, self.model.policyTrainNet(stateBatch)), dim=1)
        policyLoss = -1 * self.model.qTrainNet(statesAndActions).mean()

        self.optimizerPolicyTrainNet.zero_grad()
        policyLoss.backward()
        self.optimizerPolicyTrainNet.step()

        # put model in evaluation mode again
        self.model.eval()

        return

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
                action = self.selectAction(state)
                nextState, reward, episodeTerminated = env.react(action)

                # log_probs.append(log_prob)
                rewards.append(reward)
                episodeReturn += reward

                # Store the transition in memory
                self.memory.push(state, action, nextState, reward)

                # Move to the next state
                state = nextState

                # Update the target networks
                if self.TARGET_UPDATE < 1:
                    # update by applying a soft update
                    policyTrainNetDict, policyTargetNetDict = self.model.policyTrainNet.state_dict(), self.model.policyTargetNet.state_dict()
                    for param in policyTargetNetDict.keys():
                        policyTargetNetDict[param] = (1 - self.TARGET_UPDATE) * policyTargetNetDict[
                            param] + self.TARGET_UPDATE * policyTrainNetDict[param]

                    self.model.policyTargetNet.load_state_dict(policyTargetNetDict)

                    qTrainNetDict, qTargetNetDict = self.model.qTrainNet.state_dict(), self.model.qTargetNet.state_dict()
                    for param in qTargetNetDict.keys():
                        qTargetNetDict[param] = (1 - self.TARGET_UPDATE) * qTargetNetDict[param] + self.TARGET_UPDATE * \
                                                qTrainNetDict[param]
                else:
                    # update by copying every parameter
                    if i_episode % self.TARGET_UPDATE == 0:
                        self.model.policyTargetNet.load_state_dict(self.model.policyTrainNet.state_dict())
                        self.model.qTargetNet.load_state_dict(self.model.qTrainNet.state_dict())

            # optimize
            self.optimizeModel()

            episodeReturns.append(torch.tensor([[episodeReturn, ]]))
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

            episodeReturns.append(torch.tensor([[episodeReturn, ]]))
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
    envConfig = {"stateDefinition": "6d-norm", "actionSet": "A4", "rewardFunction": "propRewardStepPenalty",
                 "acceptance": 5e-3, "targetDiameter": 3e-2, "maxStepsPerEpisode": 50, "successBounty": 10,
                 "failurePenalty": -10, "device": "cuda" if torch.cuda.is_available() else "cpu"}
    initEnvironment(**envConfig)

    # create model
    model = Model(PolicyNetwork=Network.PDF3, QNetwork=Network.FC7)

    # define hyper parameters
    hyperParamsDict = {"BATCH_SIZE": 128, "GAMMA": 0.999, "TARGET_UPDATE": 20, "EPS_START": 0.5, "EPS_END": 0,
                       "EPS_DECAY": 500, "MEMORY_SIZE": int(1e4)}

    # set up trainer
    trainer = Trainer(model, torch.optim.Adam, 3e-4, **hyperParamsDict)

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

    # stuff for interactive testing
    env = Environment()
    state = env.initialState
    action = lambda state: trainer.selectAction(state)

