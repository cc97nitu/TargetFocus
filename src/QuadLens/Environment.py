import os
import subprocess as sp
import threading
import enum
import torch
from collections import namedtuple

from time import time
from shutil import copy, rmtree
from itertools import product
from sdds import SDDS

terminations = namedtuple("terminations", ["successful", "failed", "aborted"])
path2defns = "-rpnDefns=" + os.path.expanduser("~/.defns.rpn")


class RewardFunctions(object):
    """Storage for reward functions."""

    def propReward(self, spotSizeChange, distanceChange, scale):
        """
        Gives reward according to the change in distance to goal in the latest step.
        :param spotSizeChange: change in size of beam spot
        :param distanceChange: change in distance to goal
        :param scale: scalar value to adjust reward size
        :return: reward
        """
        return -scale * (spotSizeChange + distanceChange)

    def propRewardStepPenalty(self, spotSizeChange, distanceChange, scale):
        """
        Gives reward according to the change in distance to goal in the latest step.
        :param spotSizeChange: change in size of beam spot
        :param distanceChange: change in distance to goal
        :param scale: scalar value to adjust reward size
        :return: reward
        """
        return -scale * (spotSizeChange + distanceChange) - 0.5


class Termination(enum.Enum):
    INCOMPLETE = enum.auto()
    SUCCESSFUL = enum.auto()
    FAILED = enum.auto()
    ABORTED = enum.auto()


class StateDefinition(enum.Enum):
    RAW_16 = enum.auto()
    NORM_16 = enum.auto()


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
        elif kwargs["rewardFunction"] == "constantRewardPerStep":
            Environment.reward = RewardFunctions.constantRewardPerStep

        Environment.bounty = torch.tensor([kwargs["successBounty"]], dtype=torch.float, device=device).unsqueeze_(0)
        Environment.penalty = torch.tensor([kwargs["failurePenalty"]], dtype=torch.float, device=device).unsqueeze_(0)

        # environment properties
        Environment.targetDiameter = kwargs["targetDiameter"]
        Environment.targetRadius = Environment.targetDiameter / 2
        Environment.acceptance = kwargs["acceptance"]
        Environment.maxStepsPerEpisodes = kwargs["maxStepsPerEpisode"]
        Environment.maxIllegalStateCount = kwargs["maxIllegalStateCount"]

        # state definition
        if kwargs["stateDefinition"] == "NORM_16":
            raise NotImplementedError("think about meaningful normalization for deflections")
            # Environment.features = 16
            # Environment.stateDefinition = StateDefinition.NORM_16
        elif kwargs["stateDefinition"] == "RAW_16":
            Environment.features = 16
            Environment.stateDefinition = StateDefinition.RAW_16
        else:
            raise ValueError("cannot interpret state definition")

        # create action set
        if kwargs["actionSet"] == "A4":
            posChanges = [-1e-2, 1e-2]
        elif kwargs["actionSet"] == "A9":
            posChanges = [-1e-2, 0, 1e-2]
        else:
            raise ValueError("cannot interpret action set!")

        Environment.actionSet = [[a, b, c, d] for a, b, c, d in
                                 product(posChanges, posChanges, posChanges, posChanges)]
        Environment.actionSet = torch.tensor(Environment.actionSet, dtype=torch.float, device=device)

    except KeyError:
        raise ValueError("incomplete environment configuration!")

    Environment.configured = True
    return


class Environment(object):
    """
    Simulated Environment.
    """
    configured = False

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
        self.illegalStateCount = 0

        # working directory
        workDir = "/dev/shm/"

        # create temporary storage
        while True:
            directory = str(threading.get_ident()) + "_" + str(int(1e7 * time()))

            if not os.path.exists(workDir + directory):
                os.makedirs(workDir + directory)
                break

        self.directory = workDir + directory

        # initialize beam
        centroid = torch.zeros(6, dtype=torch.float)
        meanVal = 0.1 * torch.randn(4)  # rescale to meaningful values
        centroid[0:4] = meanVal
        self.__createCommandFile(centroid)

        # observe initial state
        self.deflections = torch.zeros(4)
        self.state = self.__runSimulation(deflections=self.deflections)
        self.spotSize = self.state[0, 10] + self.state[0, 11]
        self.distanceToGoal = self.state[0, 8:10].norm()

        # is initial state a final state?
        if self.__validateState(self.state) != Termination.INCOMPLETE:
            raise ValueError("starting in terminal state")

        return

    def react(self, action: torch.tensor, dummy=False):
        """
        Environment's answer to an agent's action.
        :param action: action to respond to
        :param dummy: omit adjustment of steerers, used only by constructor
        :return: new state, reward, termination type of new state
        """
        # return terminal state if maximal amount of reactions exceeded
        if not self.reactCount < self.maxStepsPerEpisodes:
            # print("forced abortion of episode, max steps exceeded")
            return None, self.penalty, Termination.ABORTED
        else:
            self.reactCount += 1

        # adjust steerers
        if not dummy:
            self.deflections += self.actionSet[action[0].item()]

        # obtain new state
        self.state = self.__runSimulation(self.deflections)

        # is new state a final state?
        termination = self.__validateState(self.state)

        if termination == Termination.SUCCESSFUL:
            return None, self.bounty, Termination.SUCCESSFUL
        elif termination == Termination.FAILED:
            self.illegalStateCount +=1
            if self.illegalStateCount >= self.maxIllegalStateCount:
                return None, self.penalty, Termination.FAILED
        else:
            self.illegalStateCount = 0

        # return new state
        newSpotSize = self.state[0, 10] * self.state[0, 11]
        newDistanceToGoal = self.state[0, 8:10].norm()

        spotSizeChange = newSpotSize - self.spotSize
        distanceChange = newDistanceToGoal - self.distanceToGoal
        self.spotSize = newSpotSize
        self.distanceToGoal = newDistanceToGoal

        return self.state, torch.tensor([self.reward(spotSizeChange, distanceChange, 10 ** 3)], dtype=torch.float,
                                        device=Environment.device).unsqueeze_(
            0), Termination.INCOMPLETE

    def __createLattice(self, deflections: torch.tensor):
        """
        Creates the lattice.lte file.
        :param deflections: deflection angles for the steering magnets
        :return: None
        """
        os.chdir(self.directory)
        with open("lattice.lte", 'w') as lattice:
            content = ("D1: Drift, L=0.2",
                       "W1: Watch, Mode=Parameter, Filename=W1.SDDS",
                       "Steerer1: QUAD, L=0.15, K1=0, HKICK={}, VKICK={}".format(*deflections[0:2]),
                       "D2: Drift, L=0.2",
                       "W2: Watch, Mode=Parameter, Filename=W2.SDDS",
                       "Steerer2: QUAD, L=0.15, K1=0, HKICK={}, VKICK={}".format(*deflections[2:4]),
                       "D3: Drift, L=0.2",
                       "QH: QUAD, L=0.15, K1=5.296345936776492e+01",
                       "D4: Drift, L=0.05",
                       "QV: QUAD, L=0.15, K1=-5.884268589682184e+01",
                       "D5: Drift, L=0.2",
                       "beamline: line=(D1,W1,Steerer1,D2,W2,Steerer2,QH,D4,QV,D5)")

            content = "\n".join(content)  # create one string from content with newline symbols in between
            lattice.write(content)
            return

    def __createCommandFile(self, centroid: torch.tensor):
        """
        Creates the run.ele file.
        :param centroid: centroid for each coordinate
        :return: None
        """
        os.chdir(self.directory)
        with open("run.ele", 'w') as run:
            content = ("&run_setup",
                       "lattice=lattice.lte",
                       "use_beamline=beamline",
                       "p_central_mev=0.25",
                       "sigma=%s.sig",
                       "centroid=%s.cen",
                       "output = %s.out",
                       "&end",
                       "",
                       "&run_control &end",
                       "",
                       "&bunched_beam",
                       "n_particles_per_bunch=1000",
                       "emit_x=0.45e-6",
                       "beta_x=.24",
                       "alpha_x=-2.13",
                       "emit_y=0.51e-6",
                       "beta_y=2.1",
                       "alpha_y=5.9",
                       "distribution_type[0]=\"gaussian\",\"gaussian\"",
                       "enforce_rms_values[0]=1,1,1",
                       "centroid[0]={},{},{},{},{},{}".format(*centroid),
                       "&end",
                       "",
                       "&track &end",)

            content = "\n".join(content)  # create one string from content with newline symbols in between
            run.write(content)
            return

    def __runSimulation(self, deflections: torch.tensor) -> torch.tensor:
        """
        Execute simulation and read results.
        :param centroid: centroid for each coordinate
        :param deflections: steerers' deflection angles
        :return: current state
        """
        self.__createLattice(deflections)

        # run elegant simulation
        with open(os.devnull, "w") as f:
            sp.call(["elegant", "run.ele", path2defns], stdout=f, cwd=self.directory)

        # read elegant output
        os.chdir(self.directory)
        dataSet = SDDS(0)

        # beam on target
        dataSet.load(self.directory + "/run.sig")
        targetSx = dataSet.columnData[dataSet.columnName.index("Sx")][0][-1]
        targetSy = dataSet.columnData[dataSet.columnName.index("Sy")][0][-1]

        dataSet.load(self.directory + "/run.cen")
        targetCx = dataSet.columnData[dataSet.columnName.index("Cx")][0][-1]
        targetCy = dataSet.columnData[dataSet.columnName.index("Cy")][0][-1]

        # beam on first watch element
        dataSet.load(self.directory + "/W1.SDDS")
        w1Sx = dataSet.columnData[dataSet.columnName.index("Sx")][0][-1]
        w1Sy = dataSet.columnData[dataSet.columnName.index("Sy")][0][-1]
        w1Cx = dataSet.columnData[dataSet.columnName.index("Cx")][0][-1]
        w1Cy = dataSet.columnData[dataSet.columnName.index("Cy")][0][-1]

        # beam on second watch element
        dataSet.load(self.directory + "/W2.SDDS")
        w2Sx = dataSet.columnData[dataSet.columnName.index("Sx")][0][-1]
        w2Sy = dataSet.columnData[dataSet.columnName.index("Sy")][0][-1]
        w2Cx = dataSet.columnData[dataSet.columnName.index("Cx")][0][-1]
        w2Cy = dataSet.columnData[dataSet.columnName.index("Cy")][0][-1]

        # create state tensor
        state = torch.tensor([w1Cx, w1Cy, w1Sx, w1Sy, w2Cx, w2Cy, w2Sx, w2Sy, targetCx, targetCy, targetSx, targetSy],
                             dtype=torch.float)
        return torch.cat([state, deflections]).unsqueeze(0)

    def __validateState(self, state: torch.tensor) -> Termination:
        # is beam still visible on every target?
        if state[0, 0:2].norm() > self.targetRadius or state[0, 4:6].norm() > self.targetRadius or state[
                                                                                                             0,
                                                                                                             8:10].norm() > self.targetRadius:
            # beam left target / pipe
            return Termination.FAILED

        # has beam the desired properties?
        newSpotSize = state[0, 10] * state[0, 11]
        newDistanceToGoal = state[0, 8:10].norm()

        if newDistanceToGoal < self.acceptance and newSpotSize < 2 * 1.61e-7:
            # 1.61e-7 is (rounded) minimal spot size for straight weft according to optimization via elegant
            return Termination.SUCCESSFUL

        return Termination.INCOMPLETE


    def __del__(self):
        """clean up before destruction"""
        rmtree(self.directory)

        return


if __name__ == "__main__":
    # environment config
    envConfig = {"stateDefinition": "RAW_16", "actionSet": "A9", "rewardFunction": "propRewardStepPenalty",
                 "acceptance": 5e-3, "targetDiameter": 3e-2, "maxIllegalStateCount": 0, "maxStepsPerEpisode": 50,
                 "successBounty": 10,
                 "failurePenalty": -10, "device": torch.device("cpu")}
    initEnvironment(**envConfig)

    # for i in range(100):
    #     try:
    #         env = Environment()
    #     except ValueError:
    #         print("final state")

    env = Environment()
