# plot of the greediness as function of experience

import numpy as np
import matplotlib.pyplot as plt

eps_start = 0.1
eps_end = 0.9
eps_decay = 100

def eps_threshold(x):
    return eps_end + (eps_start - eps_end) * np.exp(-1. * x / eps_decay)

numDecisions = np.arange(0, 1000)
eps = eps_threshold(numDecisions)

plt.plot(numDecisions, eps, label="eps_threshold")
plt.plot(numDecisions, 1 - eps, label="1 - eps_threshold")

plt.xlabel("decisions")
plt.legend()

plt.show()
plt.close()
