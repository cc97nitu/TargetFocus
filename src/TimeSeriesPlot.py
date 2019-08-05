import pickle

import seaborn as sns
import matplotlib.pyplot as plt


# fetch data
with open("dump/benchmark", "rb") as file:
    data = pickle.load(file)

# plot
sns.set(style="darkgrid")

sns.lineplot(x="episode", y="meanReturn", hue="behavior", data=data)

plt.show()
plt.close()
