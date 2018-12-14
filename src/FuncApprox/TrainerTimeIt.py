import timeit
import matplotlib.pyplot as plt
from copy import deepcopy

setup = """
import torch
import pickle
import Trainer
import Network


# fetch the data
with open("/home/conrad/RL/TempDiff/TargetFocus/dump/supervisedLearning/12.12.18/supervisedTrainingSet", 'rb') as file:
    data = pickle.load(file)

trainInput = data[0]
trainLabels = data[1]

network = Network.FulCon10()
trainer = Trainer.{}(network, epochs=1)

epochs = 20
"""

stmt = """
for epoch in range(1, epochs + 1):
    # train
    trainer.applyUpdate(trainInput, trainLabels)
"""

trainers = ("SGD", "ASGD", "Adam", "Adamax", "Adagrad", "Adadelta", "Rprop", "RMSprop",)
times = []

for trainer in trainers:
    times.append(timeit.timeit(stmt=stmt, setup=setup.format(trainer), number=10))

print(times)
# normalize times on SGD time
timeSGD = deepcopy(times[0])

for i in range(len(times)):
    times[i] = times[i] / timeSGD * 100

print(times)
# plot the results
fig, ax = plt.subplots()

ax.bar(range(len(trainers)), times)

# add annotation
ax.set_title("Computation Time for various Optimizers", size='x-large')
ax.set_xticklabels(("", *trainers))

for tick in ax.get_xticklabels():
    tick.set_rotation(45)

yTicks = ax.get_yticks()
ax.set_yticklabels(['{}%'.format(x) for x in yTicks])

ax.set_xlabel("optimizer")
ax.set_ylabel("rel. computation time")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
plt.close()
