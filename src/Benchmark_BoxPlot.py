"""
Visualize performance during benchmark.
"""
import pickle
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import SQL


def plotStatistics(data):
    # plot
    sns.set(style="whitegrid")

    sns.boxplot(x="agent_ident", y="termination_count", hue="termination_type", data=data, palette="Set2")

    plt.xlabel(None)
    plt.ylabel("episodes")

    plt.show()
    plt.close()


def buildPdFrame(*args):
    # init dict from which pd frame will be initialized from
    frame = dict()
    frame["termination_count"] = list()
    frame["termination_type"] = list()
    frame["agent_ident"] = list()

    # fill frame with data
    for arg in args:
        data, agent_ident = arg[0], arg[1]

        for key in data["terminations"].keys():
            frame["termination_count"] += data["terminations"][key]
            frame["termination_type"] += [str(key) for i in range(len(data["terminations"][key]))]
            frame["agent_ident"] += [str(agent_ident) for i in range(len(data["terminations"][key]))]

    # for arg in args:
    #     data, agent_ident = arg[0], arg[1]
    #
    #     for key in data["randomTerminations"].keys():
    #         frame["termination_count"] += data["randomTerminations"][key]
    #         frame["termination_type"] += [str(key) for i in range(len(data["randomTerminations"][key]))]
    #         frame["agent_ident"] += ["random" for i in range(len(data["randomTerminations"][key]))]

    return pd.DataFrame.from_dict(frame)



if __name__ == "__main__":
    frame = lambda x: SQL.retrieveBenchmark(x)

    # fetch results from scipy.Optimize
    with open("/home/dylan/RL/TempDiff/TargetFocus/src/dump/Optimize/Nelder-Mead_deterministic.dump", "rb") as file:
        optA = pickle.load(file)


    # plot multiple benchmarks
    args = [(frame(105), "A4"), (frame(106), "A9"), (optA, "Nelder-Mead")]

    plotStatistics(buildPdFrame(*args))

    # # plot a single benchmark
    # frameA = SQL.retrieveBenchmark(58)
    # data = buildPdFrame((frameA, "test"))
    # plotStatistics(data)

    # # try optimize
    # import Optimize
    # import torch
    #
    # # environment config
    # envConfig = {"stateDefinition": "6d-raw", "actionSet": "A9", "rewardFunction": "stochasticPropRewardStepPenalty",
    #              "acceptance": 5e-3, "targetDiameter": 3e-2, "maxIllegalStateCount": 0, "maxStepsPerEpisode": 5e1,
    #              "stateNoiseAmplitude": 1e-1, "rewardNoiseAmplitude": 1, "successBounty": 10,
    #              "failurePenalty": -10, "device": torch.device("cpu")}
    #
    # result = Optimize.benchmark("Nelder-Mead", 4, 10, envConfig)
    # opt = buildPdFrame((result, "Nelder-Mead",))
    # plotStatistics(opt)


