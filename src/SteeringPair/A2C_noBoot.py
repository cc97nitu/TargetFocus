import torch
import torch.nn.functional

import SteeringPair.A2C
from SteeringPair import Environment, Termination, initEnvironment


class Model(SteeringPair.A2C.Model):
    pass

class Trainer(SteeringPair.A2C.Trainer):
    def optimizePolicy(self, rewards, log_probs, states):
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

        # do supervised learning
        states = torch.cat(states)
        predictedReturns = self.model.vTrainNet(states)
        loss = torch.nn.functional.smooth_l1_loss(predictedReturns, observedReturns)
        self.optimizerVTrain.zero_grad()
        loss.backward()
        for param in self.model.vTrainNet.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizerVTrain.step()

        # observedReturns = (observedReturns - observedReturns.mean()) / (
        #         observedReturns.std() + 1e-9)  # normalize discounted rewards

        # calculate policy gradient
        policy_gradient = []
        for log_prob, Gt, state in zip(log_probs, observedReturns, states):
            policy_gradient.append(-log_prob * (Gt - self.model.vTrainNet(state)))

        self.optimizerPolicy.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
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

                # log
                states.append(state)
                nextStates.append(nextState)
                rewards.append(reward)
                log_probs.append(log_prob)
                episodeReturn += reward

                # Move to the next state
                state = nextState

            # update policy
            self.optimizePolicy(rewards, log_probs, states)

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
    model = Model(QNetwork=Network.FC7, PolicyNetwork=Network.Cat3)

    # define hyper parameters
    hyperParamsDict = {"BATCH_SIZE": 128, "GAMMA": 0.999, "TARGET_UPDATE": 10, "EPS_START": 0.5, "EPS_END": 0,
                       "EPS_DECAY": 500, "MEMORY_SIZE": int(1e4)}

    # set up trainer
    trainer = Trainer(model, torch.optim.Adam, 3e-4, **hyperParamsDict)

    # train model under hyper parameters
    episodeReturns, _ = trainer.trainAgent(500)
    plt.plot(episodeReturns)
    plt.show()
    plt.close()
