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
    with open("/TargetFocus/src/dump/Optimize/Nelder-Mead_stochastic.dump", "rb") as file:
        optA = pickle.load(file)

    with open("/TargetFocus/src/dump/Optimize/deterministic-CG.dump", "rb") as file:
        optB = pickle.load(file)

    with open("/TargetFocus/src/dump/Optimize/Powell-deterministic.dump", "rb") as file:
        optC = pickle.load(file)

    with open("/TargetFocus/src/dump/Optimize/BFGS_deterministic.dump", "rb") as file:
        optD = pickle.load(file)

    with open("/TargetFocus/src/dump/Optimize/L-BFGS-B_deterministic.dump", "rb") as file:
        optE = pickle.load(file)

    with open("/TargetFocus/src/dump/Optimize/trust-constr_deterministic.dump", "rb") as file:
        optF = pickle.load(file)

    with open("/TargetFocus/src/dump/Optimize/SLSQP_deterministic.dump", "rb") as file:
        optG = pickle.load(file)



    # plot multiple benchmarks
    args = [(frame(119), "A2C_temp"), (frame(120), "DQN"), (frame(121), "A2C_mont"), (frame(122), "REINFORCE"), (frame(118), "random")]
    plotStatistics(buildPdFrame(*args))

    # # plot a single benchmark
    # frameA = SQL.retrieveBenchmark(58)
    # data = buildPdFrame((frameA, "test"))
    # plotStatistics(data)
