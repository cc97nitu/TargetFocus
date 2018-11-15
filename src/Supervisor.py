from Environment import Environment
from Agent import Agent


if __name__ == '__main__':
    # initialize
    environment = Environment(0, 0.01)
    agent = Agent(None)

    state = environment.initialState

    # run
    n = 1

    while not state:
        print(n)
        action = agent.takeAction(state)
        state, reward = environment.react(action)
        n += 1

