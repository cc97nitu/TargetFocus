import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from Environment import Environment

# test arguments
xVal, yVal = torch.arange(-0.1, 0.1 + 1e-3, step=1e-3), torch.arange(-0.1, 0.1 + 1e-3, step=1e-3)
# xVal, yVal = torch.arange(-0.1, 0.1, step=2e-2), torch.arange(-0.1, 0.1, step=2e-2)

testResults = torch.zeros(len(xVal), len(yVal), dtype=torch.float)

print(testResults.shape)

for i in range(len(xVal)):
    for j in range(len(yVal)):
        try:
            env = Environment(xVal[i], xVal[j])
            testResults[i][j] = 1
        except ValueError:
            testResults[i][j] = 0

# show them

fig, axes = plt.subplots()

im = axes.pcolormesh(xVal, yVal, testResults,
                     cmap='copper', norm=colors.PowerNorm(gamma=1. / 1.))
# cax = make_axes_locatable(axes).append_axes('right', size='5%', pad=0.05)
# fig.colorbar(im, cax=cax)

# add some notation
axes.set_title("valid arguments", size='xx-large')
axes.set_xlabel("x deflection", size='x-large')
axes.set_ylabel("y deflection", size='x-large')

plt.show()
plt.close()
