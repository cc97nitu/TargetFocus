import pickle
import torch
import numpy as np

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def calculateRunningMean(data, meanSamples):
    mean = list()
    for j in range(meanSamples, len(data)):
        mean.append(np.mean(data[j - meanSamples:j + 1]))

    return pd.Series(mean)

# fetch data
with open("/dev/shm/benchmark", "rb") as file:
    data = pickle.load(file)

meanReturns = data["meanReturn"]

# data = torch.load("/home/dylan/RL/TempDiff/TargetFocus/src/dump/randomStart_randomGoal_agents.tar")
# meanReturns = data["returns"]

# plot
sns.set(style="darkgrid")

sns.lineplot(x="episode", y="return", data=meanReturns)

plt.show()
plt.close()
