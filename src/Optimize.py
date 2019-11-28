"""Check performance of optimization algorithms from SciPy."""
import io
import torch
import pandas as pd
import pickle
from scipy.optimize import minimize

import SQL

from SteeringPair_Stochastic.Environment import Environment as Env
from SteeringPair_Stochastic.Environment import initEnvironment, Termination

# environment interface
class EnvInterface(object):
    def __init__(self):
        self.env: Env = Env("random")
        initialState = self.env.initialState[0, -2:]
        self.relCoords = [initialState[0].item(), initialState[1].item()]
        self.episodeLength = 0

    def react(self, deflections):
        self.episodeLength += 1
        deflections = torch.tensor(deflections, dtype=torch.float)
        state, reward, termination = self.env.react(deflections, continuousAction=True)

        if termination != Termination.INCOMPLETE:
            raise TerminationException(termination)

        # distance to goal
        distance = state[0, -2:].norm().item()

        return distance

# signal termination of episode
class TerminationException(Exception):
    def __init__(self, termination: Termination, *args):
        super(TerminationException, self).__init__(termination, *args)
        self.termination: Termination = termination


def experiment(method: str, agents: int, episodes: int):
    # store results
    terminations = {"successful": list(), "failed": list(), "aborted": list()}
    episodeLengths = list()

    # benchmark different "agents"
    for i in range(agents):
        print("agent {}/{}".format(i+1, agents), end="\r")

        term_count = {"successful": 0, "failed": 0, "aborted": 0}

        # test upon episodes
        for j in range(episodes):
            interface = EnvInterface()

            try:
                out = minimize(lambda x: interface.react(x), interface.relCoords, method=method, tol=1e-350,
                               options={"maxiter": 1e4})

                # some algorithms refuse to work if they got stuck
                if not out.success:
                    raise TerminationException(Termination.FAILED)

                print("Warning: no exception encountered!")
                print(out)
            except TerminationException as t:
                # log number of steps
                episodeLengths.append(interface.episodeLength)

                # log termination
                if t.termination == Termination.SUCCESSFUL:
                    term_count["successful"] += 1
                elif t.termination == Termination.FAILED:
                    term_count["failed"] += 1
                elif t.termination == Termination.ABORTED:
                    term_count["aborted"] += 1
                else:
                    print(t)

        # append to terminations
        for key in term_count:
            terminations[key].append(term_count[key])

    return terminations, episodeLengths


def benchmark(method: str, agents: int, episodes: int, envConfig: dict):
    # set up environment
    initEnvironment(**envConfig)

    # check performance
    terminations, steps = experiment(method, agents, episodes)

    # construct data frame to store steps in
    returnFrame = {"episode": [i + 1 for i in range(len(steps))],
                   "steps": steps}
    returnFrame = pd.DataFrame(returnFrame)

    # construct dictionary similar to those found in SQL
    result = {"envConfig": envConfig, "return": returnFrame, "terminations": terminations, "method": method}

    return result


if __name__ == "__main__":

    # environment config
    envConfig = {"stateDefinition": "6d-raw", "actionSet": "A9", "rewardFunction": "stochasticPropRewardStepPenalty",
                 "acceptance": 5e-3, "targetDiameter": 3e-2, "maxIllegalStateCount": 0, "maxStepsPerEpisode": 5e2,
                 "stateNoiseAmplitude": 1e-1, "rewardNoiseAmplitude": 1, "successBounty": 10,
                 "failurePenalty": -10, "device": torch.device("cpu")}


    ### run benchmark

    # methods = ["Nelder-Mead",]
    methods = ["Nelder-Mead", "SLSQP", "CG", "Powell", "BFGS", "L-BGFS-B", "trust-krylov"]

    agents, episodes = 20, 100

    for method in methods:
        print("benching method {}".format(method))
        result = benchmark(method, agents, episodes, envConfig)

        ### insert into database
        envConfig["device"] = str(envConfig["device"])

        # dump into file like buffer
        buffer = io.BytesIO()
        dumpDict = {"environmentConfig": envConfig, "method": method,
                    "bench_episodes": episodes, "agents": agents, "result": result}
        pickle.dump(dumpDict, buffer)

        # dump into SQL
        columnData = {**envConfig, "method": method,
                      "bench_episodes": episodes, "number_agents": agents, }

        SQL.insertOptimizeResult(columnData, buffer.getvalue())

    # # dump results
    # with open("/TargetFocus/src/dump/Optimize/trust-constr_deterministic.dump", "wb") as file:
    #     pickle.dump(result, file)



