import torch
import torch.nn.functional

import SteeringPair.A2C_noBoot
from SteeringPair import Environment, Termination, initEnvironment, Struct

class Model(SteeringPair.A2C_noBoot.Model):
    """No modifications needed."""
    pass

class Trainer(SteeringPair.A2C_noBoot.Trainer):
    def __init__(self, model: Model, optimizer, stepSize, **kwargs):
        """
        Set up trainer.
        :param model: model to train
        :param kwargs: dictionary containing hyper parameters
        """
        super().__init__(model, optimizer, stepSize, **kwargs)

        # set up replay memory
        self.memory = Struct.CyclicBuffer(self.MEMORY_SIZE)

        return

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

        # store states and their observed rewards in the replay memory
        for state, observedReturn in zip(states, observedReturns):
            self.memory.push(tuple([state, observedReturn]))

        observedReturns = torch.tensor(observedReturns, device=Environment.device)


        # observedReturns = (observedReturns - observedReturns.mean()) / (
        #         observedReturns.std() + 1e-9)  # normalize discounted rewards

        # calculate policy gradient
        policy_gradient = []
        for log_prob, Gt, state in zip(log_probs, observedReturns, states):
            policy_gradient.append(-log_prob * (Gt - self.model.vTargetNet(state)))

        self.optimizerPolicy.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        policy_gradient.backward()
        self.optimizerPolicy.step()

        # improve state-value function
        try:
            expBatch = self.memory.sample(self.BATCH_SIZE)
        except ValueError:
            # the are less than BATCH_SIZE samples in the memory
            return

        states, observedReturns = zip(*expBatch)
        states = torch.cat(states)
        observedReturns = torch.tensor(observedReturns, device=Environment.device)

        # do supervised learning
        predictedReturns = self.model.vTrainNet(states)
        loss = torch.nn.functional.smooth_l1_loss(predictedReturns, observedReturns)
        self.optimizerVTrain.zero_grad()
        loss.backward()
        for param in self.model.vTrainNet.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizerVTrain.step()

        # apply soft-update to minimize impact of statistical outliers in the memory
        vTrainDict, vTargetDict = self.model.vTrainNet.state_dict(), self.model.vTargetNet.state_dict()
        for param in vTargetDict.keys():
            vTargetDict[param] = (1 - self.TARGET_UPDATE) * vTargetDict[param] + self.TARGET_UPDATE * vTrainDict[param]

        self.model.vTargetNet.load_state_dict(vTargetDict)


if __name__ == "__main__":
    import torch.optim
    from SteeringPair import Network
    import matplotlib.pyplot as plt

    # environment config
    envConfig = {"stateDefinition": "6d-norm", "actionSet": "A9", "rewardFunction": "propRewardStepPenalty",
                 "acceptance": 5e-3, "targetDiameter": 3e-2, "maxStepsPerEpisode": 50, "successBounty": 10,
                 "failurePenalty": -10, "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")}
    initEnvironment(**envConfig)

    # create model
    model = Model(QNetwork=Network.FC7, PolicyNetwork=Network.Cat3)

    # define hyper parameters
    hyperParams = {"BATCH_SIZE": 128, "GAMMA": 0.9, "TARGET_UPDATE": 0.1, "EPS_START": 0.5, "EPS_END": 0,
                   "EPS_DECAY": 500, "MEMORY_SIZE": int(1e4)}

    # set up trainer
    trainer = Trainer(model, torch.optim.Adam, 3e-4, **hyperParams)

    # train model under hyper parameters
    episodeReturns, _ = trainer.trainAgent(1000)
    plt.plot(episodeReturns)
    plt.show()
    plt.close()

    # bench model
    returns, terminations, comparison = trainer.benchAgent(50)

    comparison = tuple([*zip(*comparison)])

    # try to visualize comparison
    import pandas as pd
    comparison = pd.DataFrame.from_dict({"observed": comparison[0], "predicted": comparison[1]})

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.jointplot(x="observed", y="predicted", data=comparison)
    plt.show()
    plt.close()

    print(terminations)

