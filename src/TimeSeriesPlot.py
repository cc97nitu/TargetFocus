import pickle
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

# plot
sns.set(style="darkgrid")

sns.lineplot(x="episode", y="return", hue="behavior", data=meanReturns)

plt.show()
plt.close()
