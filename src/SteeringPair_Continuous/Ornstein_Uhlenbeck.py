"""
Taken from https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
"""
import torch


class OUNoise(object):
    def __init__(self, low, high, dim, mu=0.0, theta=0.5, max_sigma=0.15, min_sigma=0.15, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = dim
        self.low = low
        self.high = high
        self.state = None
        self.reset()

    def reset(self):
        self.state = torch.ones(self.action_dim, dtype=torch.float) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * torch.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def __call__(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return torch.clamp(action + ou_state, self.low, self.high)



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")

    noise = OUNoise(-1, 1, 1, theta=0.15, min_sigma=0.05, max_sigma=0.1)

    paths = list()
    for i in range(3):
        noise.reset()
        path = list()

        for j in range(100):
            path.append(noise(0))

        paths.append(path)

    for path in paths:
        plt.plot(path)

    plt.xlabel("time step")
    plt.show()
    plt.close()

    # test torch implementation
    noise = OUNoise(-1, 1, 2)

    action = torch.tensor([0, 0], dtype=torch.float)

    disturbedAction = noise(action)

