import matplotlib.pyplot as plt
import numpy as np
import torch

import SteeringPair

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define hyper parameters
hyperParams = {"BATCH_SIZE": 128, "GAMMA": 0.999, "TARGET_UPDATE": 10, "EPS_START": 0.5, "EPS_END": 0,
               "EPS_DECAY": 500, "MEMORY_SIZE": int(1e4)}

# train an agent and plot his return per episode during training
trainEpisodes = 400

model = SteeringPair.Model()
model.train()
trainer = SteeringPair.Trainer(model, **hyperParams)

episodeReturns, _ = trainer.trainAgent(trainEpisodes)
episodeReturns = [x[0].item() for x in episodeReturns]

# calculate mean return
meanSamples = 10

meanEpisodeReturn = list()
for j in range(meanSamples, len(episodeReturns)):
    meanEpisodeReturn.append(np.mean(episodeReturns[j - meanSamples:j + 1]))

# make a plot
fig, ax = plt.subplots()

ax.plot(range(trainEpisodes), episodeReturns, label="return", color="orange")
ax.plot(range(trainEpisodes)[meanSamples:], meanEpisodeReturn, label="avg_return", color="blue")

ax.set_xlabel("episode")
ax.set_ylabel("return")

ax.legend()

plt.show()
plt.close()




