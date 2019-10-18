import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

# x-axis
x = torch.arange(-6, 6 + 1e-2, 1e-2, dtype=torch.float).unsqueeze(1)

def getYAxis(func):
    m = func()
    y = m(x)
    return y.squeeze(1).numpy()

# plot
xVal = x.squeeze(1).numpy()

rangeMin, rangeMax = -3, 3

sns.set_style("whitegrid")
fix, axes = plt.subplots(3, 2, sharex=True, sharey=True)

plotElements = [[(0,0), nn.Sigmoid], [(0, 1), nn.Tanh], [(1, 0), nn.ReLU], [(1, 1), nn.ELU], [(2, 0), nn.Hardtanh], [(2, 1), nn.Tanhshrink]]

for element in plotElements:
    axes[element[0]].plot(xVal, getYAxis(element[1]))
    axes[element[0]].set_ylim(rangeMin, rangeMax)
    # axes[element[0]].set_xlim(rangeMin, rangeMax)

plt.show()
plt.close()