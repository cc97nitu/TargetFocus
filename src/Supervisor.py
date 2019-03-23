from Struct import Transition


class Rover(object):

    def __init__(self, agent):
        self.agent = agent

        return

    def walk(self, environment, learnOnline=False):
        """
        Experience an episode.
        :param environment: environment to interact with
        :return: number of steps, earned rewards, success
        """
        steps = 0
        rewards = []

        # initial state is provided by the environment
        state = environment.initialState

        # walk until end of episode
        while not state:
            action = self.agent.takeAction(state)
            state, reward = environment.react(action)

            self.agent.remember(Transition(action, reward, state))

            if learnOnline:
                self.agent.learn(agent.shortMemory)

            steps += 1
            rewards.append(reward)

        # did the episode end successfully?
        success = True if rewards[-1] == environment.bounty else False

        return steps, rewards, success


if __name__ == '__main__':
    from Agent import Agent
    from Environment import Environment
    from QValue import QNeural
    from FuncApprox.TargetGenerator import sarsaLambda

    # build agent
    agent = Agent(q=QNeural(), epsilon=1, discount=0.9, learningRate=0.9, memorySize=1, traceDecay=0, targetGenerator=sarsaLambda)

    # build rover
    rover = Rover(agent)

    # run some episodes
    steps, rewards, success = rover.walk(Environment(0, 0), learnOnline=True)
    print("steps: {0}, return: {1}, success: {2}".format(steps, sum(rewards), success))

    for i in range(0, 10):
        for j in range(0, 10):
            steps, rewards, success = rover.walk(Environment(0, 0), learnOnline=True)

        print("steps: {0}, return: {1}, success: {2}".format(steps, sum(rewards), success))

