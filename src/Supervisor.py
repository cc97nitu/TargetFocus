from Environment import Environment
from Agent import Agent
from Struct import Transition
from QValue import QNeural


if __name__ == '__main__':
    # initialize
    environment = Environment(0, 0.01)
    agent = Agent(QNeural())

    state = environment.initialState

    # run
    n = 1

    while not state:
        print(n)
        action = agent.takeAction(state)
        state, reward = environment.react(action)
        agent.remember(Transition(action, reward, state))
        n += 1

