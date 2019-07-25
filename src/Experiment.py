import pickle
import pandas as pd

from DQN.DQN import DQN

# run simulation
returns = []

# random behavior
for i in range(10):
    print("random run {}".format(i))
    dqn = DQN()
    dqn.EPS_START = 1
    dqn.EPS_END = 1
    episodeReturns = dqn.trainAgent(50)
    episodeReturns = [x[0].item() for x in episodeReturns]
    returns.append(pd.DataFrame({"episode": [i+1 for i in range(len(episodeReturns))],
                                 "behavior": ["random" for i in range(len(episodeReturns))],
                                 "return": episodeReturns}))

# decreasing epsilon
for i in range(10):
    print("dec_eps run {}".format(i))
    dqn = DQN()
    episodeReturns = dqn.trainAgent(50)
    episodeReturns = [x[0].item() for x in episodeReturns]
    returns.append(pd.DataFrame({"episode": [i+1 for i in range(len(episodeReturns))],
                                 "behavior": ["dec_eps" for i in range(len(episodeReturns))],
                                 "return": episodeReturns}))



# concat to pandas data frame
returns = pd.concat(returns)

# dump
with open("dump/test", "rb") as file:
    pickle.dump(returns, file)
