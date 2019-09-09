import enum
import torch


class Termination(enum.Enum):
    INCOMPLETE = enum.auto()
    SUCCESSFUL = enum.auto()
    FAILED = enum.auto()
    ABORTED = enum.auto()


def initEnvironment(**kwargs):
    pass


class Environment(object):
    goal = torch.tensor((4e-3, -4e-3), dtype=torch.float)
    flightLength = 0.15
    features = 2
    initialState = torch.zeros(1, 2)

    device = torch.device("cpu")

    def __init__(self, *args):
        pass

    def react(self, action: torch.tensor):
        position = action * Environment.flightLength
        distance = position - self.goal
        return None, -1 * (10**3 * distance.norm())**2, Termination.FAILED


if __name__ == "__main__":
    import torch.optim
    from SteeringPair import Network
    import matplotlib.pyplot as plt
    import seaborn as sns

    from torch.autograd import Variable
    from torch.distributions.multivariate_normal import MultivariateNormal

    from SteeringPair_Continuous.REINFORCE import Model, Trainer

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
    def createPDF(pdfParam):
        pdfParam = torch.cat([pdfParam[:2] * 0.066, pdfParam[2:] * 0.066 / 3])
        pdf = MultivariateNormal(pdfParam[:2], torch.tensor([[pdfParam[2] ** 2, 0], [0, pdfParam[3] ** 2]]))

        # pdf = MultivariateNormal(pdfParam[:2], torch.tensor([[10**-8, 0], [0, 10**-8]], dtype=torch.float))
        return pdf


    def plotPDF(pdf):
        sample = pdf.sample((10000,))
        sample.transpose_(1, 0)
        x, y = sample[0].numpy(), sample[1].numpy()

        sns.kdeplot(x, label="x", color="blue")
        sns.kdeplot(y, label="y", color="orange")

        # mark optimal means
        optMeans = Environment.goal / Environment.flightLength
        plt.axvline(optMeans[0], color="blue", dashes=[2,2,8,2])
        plt.axvline(optMeans[1], color="orange", dashes=[2,2,8,2])

        plt.legend()
        plt.show()
        plt.close()


    def plotEpisodeReturns(episodeReturns):
        x = torch.tensor(episodeReturns)
        backwardEpisodes = 50
        y = torch.empty(len(x), dtype=torch.float)

        for i in range(0, len(y)):
            y[i] = x[i: i + backwardEpisodes].mean()

        plt.plot(y.numpy())
        plt.show()
        plt.close()


    pdfParam = model.policy_net.forward(Variable(Environment.initialState)).squeeze(0)
    pdf = createPDF(pdfParam)

    plotPDF(pdf)

    for i in range(0, 3):
        episodeReturns, _ = trainer.trainAgent(1500)

        pdfParam = model.policy_net.forward(Variable(Environment.initialState)).squeeze(0)
        pdf = createPDF(pdfParam)

        plotPDF(pdf)
        plotEpisodeReturns(episodeReturns)
