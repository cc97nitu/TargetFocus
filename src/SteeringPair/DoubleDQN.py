import torch

import SteeringPair
from SteeringPair import Struct, Environment


class Model(SteeringPair.DQN.Model):
    """No changes required."""
    pass


class Trainer(SteeringPair.DQN.Trainer):
    def optimizeModel(self):
        if len(self.memory) < self.BATCH_SIZE:
            return

        # put model in training mode
        self.model.train()

        # sample from replay memory
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Struct.Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=Environment.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.model.policy_net(state_batch).gather(1, action_batch)

        # next actions as predicted by the policy net
        nextActionBatch = self.model.policy_net(non_final_next_states).argmax(1).unsqueeze(1)

        # get next actions Q-value computed by the target network
        next_state_values = torch.zeros(self.BATCH_SIZE, device=Environment.device)
        next_state_values[non_final_mask] = self.model.target_net(non_final_next_states).gather(1, nextActionBatch).squeeze(1).detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        loss = torch.nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # put model in evaluation mode again
        self.model.eval()
        return


if __name__ == "__main__":
    import torch.optim
    from SteeringPair import Network, initEnvironment
    import matplotlib.pyplot as plt

    # environment config
    envConfig = {"stateDefinition": "6d-norm", "actionSet": "A4", "rewardFunction": "propReward",
                 "acceptance": 5e-3, "targetDiameter": 3e-2, "maxStepsPerEpisode": 50, "successBounty": 10,
                 "failurePenalty": -10, "device": "cuda" if torch.cuda.is_available() else "cpu"}
    initEnvironment(**envConfig)

    # create model
    model = Model(QNetwork=Network.FC7)

    # define hyper parameters
    hyperParamsDict = {"BATCH_SIZE": 128, "GAMMA": 0.999, "TARGET_UPDATE": 10, "EPS_START": 0.5, "EPS_END": 0,
                       "EPS_DECAY": 500, "MEMORY_SIZE": int(1e4)}

    # set up trainer
    trainer = Trainer(model, torch.optim.Adam, 3e-4, **hyperParamsDict)

    # train model under hyper parameters
    episodeReturns, _ = trainer.trainAgent(500)
    plt.plot(episodeReturns)
    plt.show()
    plt.close()
