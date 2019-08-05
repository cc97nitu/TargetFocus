import pickle

import seaborn as sns
import matplotlib.pyplot as plt


# fetch data
with open("/dev/shm/benchmark", "rb") as file:
    data = pickle.load(file)

#
returns = data["returns"]
# plot
sns.set(style="darkgrid")

sns.lineplot(x="episode", y="return", hue="behavior", data=returns)

plt.show()
plt.close()
