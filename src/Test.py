import torch

import SteeringPair


def Env(*args):
    env = SteeringPair.Environment(*args)
    state = env.initialState
    return env, state

# define hyper parameters
hyperParams = {"BATCH_SIZE": 128, "GAMMA": 0.999, "TARGET_UPDATE": 10, "EPS_START": 0.5, "EPS_END": 0,
               "EPS_DECAY": 500, "MEMORY_SIZE": int(1e4)}

# fetch pre-trained agents
trainResults = torch.load("/dev/shm/agents.tar")
agents = trainResults["agents"]

model = SteeringPair.REINFORCE.Model()
model.load_state_dict(agents["agent_0"])
model.eval()

if __name__ == "__main__":
    states = list()
    for i in range(10):
        env, state = Env("random")
        states.append(state)

    states = torch.cat(states, dim=0).transpose(1, 0)
    print(states)
    states /= 1e-2
    print(states)
