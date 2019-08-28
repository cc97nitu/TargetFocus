"""
Visualize performance during benchmark.
"""
import pickle
import torch
import numpy as np

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def plotBenchmark(data):
    # plot
    sns.set(style="darkgrid")

    sns.lineplot(x="episode", y="return", hue="behavior", data=data)

    plt.show()
    plt.close()


def statistics(data):
    """
    Calculate statistics.
    """
    def stats(key: str):
        print(key)

        for prop in data[key].keys():
            statistical = [np.mean(data[key][prop]), np.std(data[key][prop]), np.min(data[key][prop]), np.max(data[key][prop])]
            print("\t" + prop + "\t", "mean={:.1f}, std={:.1f}, min={}, max={}".format(*statistical))

    stats("greedyTerminations")
    stats("randomTerminations")


if __name__ == "__main__":
    ### benchmark ###
    # fetch data
    path = "/dev/shm/benchmark"
    with open(path, "rb") as file:
        data = pickle.load(file)

    plotBenchmark(data["meanReturn"])
    statistics(data)
