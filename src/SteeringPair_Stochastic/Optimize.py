import torch

from SteeringPair_Stochastic.Environment import Environment as Env
from SteeringPair_Stochastic.Environment import initEnvironment, Termination

# environment interface
class EnvInterface(object):
    def __init__(self):
        self.env: Env = Env("random")
        initialState = self.env.initialState[0, -2:]
        self.relCoords = [initialState[0].item(), initialState[1].item()]

    def react(self, deflections):
        deflections = torch.tensor(deflections, dtype=torch.float)
        state, reward, termination = self.env.react(deflections, continuousAction=True)

        if termination != Termination.INCOMPLETE:
            raise TerminationException(termination)

        print(reward)
        # distance to goal
        distance = state[0, -2:].norm().item()

        return distance

# signal termination of episode
class TerminationException(Exception):
    def __init__(self, termination: Termination, *args):
        super(TerminationException, self).__init__(termination, *args)
        self.termination: Termination = termination

if __name__ == "__main__":
    from scipy.optimize import minimize

    # environment config
    envConfig = {"stateDefinition": "6d-raw", "actionSet": "A9", "rewardFunction": "stochasticPropRewardStepPenalty",
                 "acceptance": 5e-3, "targetDiameter": 3e-2, "maxIllegalStateCount": 0, "maxStepsPerEpisode": 5e2,
                 "stateNoiseAmplitude": 5e-9, "rewardNoiseAmplitude": 2e-1, "successBounty": 10,
                 "failurePenalty": -10, "device": torch.device("cpu")}
    initEnvironment(**envConfig)

    # set up interface to run optimizer
    interface = EnvInterface()

    try:
        res = minimize(lambda x: interface.react(x), interface.relCoords, method="trust-ncg", tol=1e-350, options={"maxiter": 1e3})
        print(res)
    except TerminationException as t:
        print(t)

