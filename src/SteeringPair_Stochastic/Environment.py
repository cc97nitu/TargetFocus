import os
import subprocess as sp
import threading
import enum
import torch
import torch.distributions
from collections import namedtuple

from time import time
from shutil import copy, rmtree
from itertools import product
from sdds import SDDS

terminations = namedtuple("terminations", ["successful", "failed", "aborted"])
path2defns = "-rpnDefns=" + os.path.expanduser("~/.defns.rpn")

class RewardFunctions(object):
    """Storage for reward functions."""

    def propReward(self, distanceChange, scale):
        """
        Gives reward according to the change in distance to goal in the latest step.
        :param distanceChange: change in distance to goal
        :param scale: scalar value to adjust reward size
        :return: reward
        """
        return -scale * distanceChange

    def propRewardStepPenalty(self, distanceChange, scale):
        """
        Gives reward according to the change in distance to goal in the latest step.
        :param distanceChange: change in distance to goal
        :param scale: scalar value to adjust reward size
        :return: reward
        """
        return -scale * distanceChange - 0.5

    def stochasticPropRewardStepPenalty(self, distanceChange, scale):
        """
        Gives reward according to the change in distance to goal in the latest step.
        :param distanceChange: change in distance to goal
        :param scale: scalar value to adjust reward size
        :return: reward
        """
        mean = scale * distanceChange
        deviation = Environment.rewardNoiseAmplitude * abs(mean)
        rewardDist = torch.distributions.normal.Normal(mean, deviation)
        return -1 * rewardDist.sample().item() - 0.5

    def constantRewardPerStep(self, distanceChange, scale):
        """
        Gives reward according to the change in distance to goal in the latest step.
        :param distanceChange: change in distance to goal
        :param scale: scalar value to adjust reward size
        :return: reward
        """
        return -1


class Termination(enum.Enum):
    INCOMPLETE = enum.auto()
    SUCCESSFUL = enum.auto()
    FAILED = enum.auto()
    ABORTED = enum.auto()


class StateDefinition(enum.Enum):
    SIXDRAW = enum.auto()
    SIXDNORM = enum.auto()
    TWODNORM = enum.auto()


def initEnvironment(**kwargs):
    """Set up Environment class."""
    try:
        # if gpu is to be used
        device = kwargs["device"]
        Environment.device = device

        # choose reward function
        if kwargs["rewardFunction"] == "propReward":
            Environment.reward = RewardFunctions.propReward
        elif kwargs["rewardFunction"] == "propRewardStepPenalty":
            Environment.reward = RewardFunctions.propRewardStepPenalty
        elif kwargs["rewardFunction"] == "stochasticPropRewardStepPenalty":
            Environment.reward = RewardFunctions.stochasticPropRewardStepPenalty
        elif kwargs["rewardFunction"] == "constantRewardPerStep":
            Environment.reward = RewardFunctions.constantRewardPerStep

        Environment.bounty = torch.tensor([kwargs["successBounty"]], dtype=torch.float, device=device).unsqueeze_(0)
        Environment.penalty = torch.tensor([kwargs["failurePenalty"]], dtype=torch.float, device=device).unsqueeze_(0)

        # environment properties
        Environment.targetDiameter = kwargs["targetDiameter"]
        Environment.targetRadius = Environment.targetDiameter / 2
        Environment.acceptance = kwargs["acceptance"]
        Environment.maxStepsPerEpisodes = kwargs["maxStepsPerEpisode"]
        Environment.rewardNoiseAmplitude = kwargs["rewardNoiseAmplitude"]
        Environment.stateNoiseAmplitude = kwargs["stateNoiseAmplitude"]

        # state definition
        if kwargs["stateDefinition"] == "2d-norm":
            Environment.features = 2
            Environment.stateDefinition = StateDefinition.TWODNORM
        elif kwargs["stateDefinition"] == "6d-raw":
            Environment.features = 6
            Environment.stateDefinition = StateDefinition.SIXDRAW
        elif kwargs["stateDefinition"] == "6d-norm":
            Environment.features = 6
            Environment.stateDefinition = StateDefinition.SIXDNORM
        else:
            raise ValueError("cannot interpret state definition")

        # create action set
        if kwargs["actionSet"] == "A4":
            posChanges = [-5e-3, 5e-3]
        elif kwargs["actionSet"] == "A9":
            posChanges = [-5e-3, 0, 5e-3]
        elif kwargs["actionSet"] == "A25":
            posChanges = [-1e-2, -1e-3, 0, 1e-3, 1e-2]
        elif kwargs["actionSet"] == "A49":
            posChanges = [-1e-2, -5e-3, -1e-3, 0, 1e-3, 5e-3, 1e-2]
        elif kwargs["actionSet"] == "A81":
            posChanges = [-1e-2, -6e-3, -3e-3, -1e-3, 0, 1e-3, 3e-3, 6e-3, 1e-2]
        else:
            raise ValueError("cannot interpret action set!")

        Environment.actionSet = [torch.tensor([x, y], dtype=torch.float, device=device) for x, y in
                                 product(posChanges, posChanges)]
    except KeyError:
        raise ValueError("incomplete environment configuration!")

    Environment.configured = True


class Environment(object):
    """
    Simulated Environment.
    """
    # has initEnvironment() been called?
    configured = False

    # if gpu is to be used
    device = None

    # number of variables representing a state
    features = None
    stateDefinition = None

    # define action set
    actionSet = None

    # define focus goal
    focusGoal = torch.tensor((8e-3, 8e-3), dtype=torch.float, device=device)  # historic dummy value

    acceptance = None  # max distance between focus goal and beam focus of a state for the state to be considered terminal
    targetDiameter = None  # diameter of target
    targetRadius = None

    # reward on success / penalty on failure
    bounty = None
    penalty = None

    # abort if episode takes to long
    maxStepsPerEpisodes = None

    # path to run.ele
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    pathRunEle = os.path.abspath("../SteeringPair/res/") + "/run.ele"

    def __init__(self, *args):
        """
        Creates an environment.
        :param args: if empty: choose random start; if  one single arg: random start, random goal; if two args: use as start coordinates
        """
        # has initEnvironment() been called?
        if not self.configured:
            raise ImportWarning("Environment class needs to be configured with initEnvironment before usage!")

        # count how often the environment reacted
        self.reactCount = 0

        # working directory
        workDir = "/dev/shm/"

        # create temporary storage
        while True:
            dir = str(threading.get_ident()) + "_" + str(int(1e7 * time()))

            if not os.path.exists(workDir + dir):
                os.makedirs(workDir + dir)
                break

        self.dir = workDir + dir

        # copy elegant config file to working directory
        copy(Environment.pathRunEle, self.dir)

        # distance to goal gets updated upon call of react()
        self.distanceToGoal = 0

        # initial magnets' deflection and beam spot's position
        if len(args) == 2:
            self.deflections = torch.tensor((args[0], args[1]), dtype=torch.float, device=Environment.device)

            # get initial focus / initial state from them
            initialState, _, episodeTerminated = self.react(torch.zeros(1, device=Environment.device).unsqueeze(0),
                                                            initialize=True)  # action is a dummy action

            if episodeTerminated != Termination.INCOMPLETE:
                raise ValueError("starting in terminal state")
            else:
                self.initialState = initialState

        elif len(args) == 0:
            # find random starting point
            while True:

                self.deflections = torch.randn(2, dtype=torch.float, device=Environment.device) * 0.1  # rescale to fit onto target

                # get initial focus / initial state from them
                initialState, _, episodeTerminated = self.react(torch.zeros(1, device=Environment.device).unsqueeze(0),
                                                                initialize=True)  # action is a dummy action

                if episodeTerminated == Termination.INCOMPLETE:
                    # starting in a non-terminal state
                    self.initialState = initialState
                    break

        elif len(args) == 1:
            # find center of goal which lies on the target
            def findGoal():
                while True:
                    focusGoal = torch.randn(2, dtype=torch.float, device=Environment.device) * 0.1  # rescale to match target size

                    if focusGoal.norm() < Environment.targetRadius:
                        return focusGoal

            self.focusGoal = findGoal()

            # find random starting point
            stuck = False
            attempts = 0
            while True:
                attempts += 1
                self.deflections = torch.randn(2, dtype=torch.float, device=Environment.device) * 0.1  # rescale to fit onto target

                # get initial focus / initial state from them
                initialState, _, episodeTerminated = self.react(torch.zeros(1, device=Environment.device).unsqueeze(0),
                                                                initialize=True)  # action is a dummy action

                if episodeTerminated == Termination.INCOMPLETE:
                    # starting in a non-terminal state
                    self.initialState = initialState
                    if stuck:
                        print("init successful")
                    break

                if attempts > 100:
                    print("Environment stuck at initialization")
                    raise ValueError("unable to find random starting points")

        else:
            # illegal number of arguments
            raise ValueError("check arguments passed to Environment!")

        return

    def react(self, action: torch.tensor, initialize=False):
        """
        Simulate response to an action performed by the agent.
        :param action: index of action to respond to
        :return: next_state, reward, termination: bool, type_termination: Termination
        """

        # update magnet strengths according to chosen action
        if not initialize:
            self.deflections = self.deflections + Environment.actionSet[action[0].item()]

        # create lattice
        self.__createLattice(self.deflections)

        # run elegant simulation
        with open(os.devnull, "w") as f:
            sp.call(["elegant", "run.ele", path2defns], stdout=f, cwd=self.dir)

        # read elegant output
        os.chdir(self.dir)
        dataSet = SDDS(0)
        dataSet.load(self.dir + "/run.out")

        # get absolute coordinates of beam center
        absCoords = torch.tensor((dataSet.columnData[0], dataSet.columnData[2]), dtype=torch.float, device=Environment.device).mean(
            1)
        absCoords = absCoords.mean(1)

        # add noise to coordinates
        deviation = Environment.stateNoiseAmplitude * absCoords
        variance = torch.diag(deviation ** 2)
        noiseDistribution = torch.distributions.multivariate_normal.MultivariateNormal(absCoords, variance)
        absCoords = noiseDistribution.sample()

        # calculate relative distance between beam focus and focus goal
        relCoords = absCoords - self.focusGoal

        #### create state tensor
        if Environment.stateDefinition == StateDefinition.SIXDRAW:
            state = torch.cat((self.deflections, absCoords, relCoords)).unsqueeze(0)
        elif Environment.stateDefinition == StateDefinition.SIXDNORM:
            # substract mean (which is zero) and divide through standard deviation
            state = torch.cat((self.deflections / 3.33e-2, absCoords / 7.5e-3, relCoords / 1e-2)).unsqueeze(0)
        elif Environment.stateDefinition == StateDefinition.TWODNORM:
            state = relCoords / 1.225e-2  # shall normalize to mean=0 and std=1
            state.unsqueeze_(0)

        # return terminal state if maximal amount of reactions exceeded
        if not self.reactCount < self.maxStepsPerEpisodes:
            # print("forced abortion of episode, max steps exceeded")
            return None, self.penalty, Termination.ABORTED
        else:
            self.reactCount += 1

        # # add noise to coordinates
        # deviation = Environment.stateNoiseAmplitude * state[0, 2:]
        # variance = torch.diag(deviation ** 2)
        # noiseDistribution = torch.distributions.multivariate_normal.MultivariateNormal(state[0, 2:], variance)
        # state = torch.cat((state[0, :2], noiseDistribution.sample())).unsqueeze(0)

        # return state and reward
        newDistanceToGoal = relCoords.norm().item()
        distanceChange = newDistanceToGoal - self.distanceToGoal
        self.distanceToGoal = newDistanceToGoal

        if newDistanceToGoal < self.acceptance:
            # goal reached
            return None, self.bounty, Termination.SUCCESSFUL
        elif absCoords.norm().item() >= self.targetRadius:
            # beam spot left target
            return None, self.penalty, Termination.FAILED
        else:
            # reward according to distanceChange
            return state, torch.tensor([self.reward(distanceChange, 10 ** 3)], dtype=torch.float,
                                       device=Environment.device).unsqueeze_(
                0), Termination.INCOMPLETE

    def __createLattice(self, strengths):
        """
        Creates the lattice.lte file.
        :param strengths: deflection angles for the steering magnets
        :return: None
        """
        strengthA, strengthB = strengths[0].item(), strengths[1].item()

        os.chdir(self.dir)
        with open("lattice.lte", 'w') as lattice:
            content = ("D1: Drift, L=0.5",
                       "Steer: QUAD, L=0.15, HKICK={0}, VKICK={1}".format(strengthA, strengthB),
                       "D2: Drift, L=0.15",
                       "beamline: line=(D1,Steer,D2)")

            content = "\n".join(content)  # create one string from content with newline symbols in between
            lattice.write(content)

            return

    def __del__(self):
        """clean up before destruction"""
        rmtree(self.dir)

        return


class EligibleEnvironmentParameters(list):
    """
    List containing valid initial environment parameters.
    """

    def __init__(self, low, high, step):
        """
        Constructor.
        :param low: lower boundary for deflection
        :param high: upper boundary for deflection
        :param step: step size
        """
        list.__init__(self)
        self.__createEnvironmentParameters(low, high, step)

        return

    def __createEnvironmentParameters(self, low, high, step):
        # deflection range
        defRange = torch.arange(low, high, step)

        # find eligible starting points
        for x, y in product(defRange, defRange):
            try:
                Environment(x, y)
                self.append((x, y))
            except ValueError:
                continue

        return


if __name__ == '__main__':
    # environment config
    envConfig = {"stateDefinition": "6d-norm", "actionSet": "A9", "rewardFunction": "stochasticPropRewardStepPenalty",
                 "acceptance": 5e-3, "targetDiameter": 3e-2, "maxStepsPerEpisode": 50, "stateNoiseAmplitude": 1e-3, "rewardNoiseAmplitude": 0.2, "successBounty": 10,
                 "failurePenalty": -10, "device": "cuda" if torch.cuda.is_available() else "cpu"}
    initEnvironment(**envConfig)

    torch.set_default_tensor_type(torch.FloatTensor)

    env = Environment(0, 0)
    state = env.initialState

    # sample deflections and absolute coordinates
    Environment.acceptance = 0

    # deflection range
    defRange = torch.arange(-0.1, 0.1, 5e-3)

    states = list()
    # find eligible starting points
    for x, y in product(defRange, defRange):
        try:
            env = Environment(x, y)
            states.append(env.initialState)
        except ValueError:
            continue

    states = torch.cat(states)
    statesMean = states.transpose(1, 0).mean(1)
    statesStd = states.transpose(1, 0).std(1)
