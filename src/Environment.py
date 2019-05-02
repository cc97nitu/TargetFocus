import os
import subprocess as sp
import threading
import torch

from time import time
from itertools import product
from shutil import copy, rmtree
from sdds import SDDS

from Struct import State, Action


class Environment(object):
    """
    Simulated Environment.
    """
    # define focus goal
    focusGoal = torch.tensor((8e-3, 8e-3), dtype=torch.float)

    def __init__(self, strengthA, strengthB):
        """
        Constructor.
        :param strengthA: initial horizontal deflection
        :param strengthB:   initial vertical deflection
        """
        # reward on success / penalty on failure
        self.bounty = 100
        self.penalty = -100

        # count how often the environment reacted
        self.reactCountMax = 50
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
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        copy("../res/run.ele", self.dir)

        # define focus goal

        self.acceptance = 5e-3  # max distance between focus goal and beam focus of a state for the state to be considered terminal
        targetDiameter = 3e-2  # diameter of target
        self.targetRadius = targetDiameter / 2

        self.distanceToGoal = 0  # distance to goal gets updated upon call of react()

        # initial strengths of magnets
        self.strengths = torch.tensor((strengthA, strengthB), dtype=torch.float)

        # get initial focus / initial state from them
        initialState = self.react(
            Action(State(torch.tensor((strengthA, strengthB), dtype=torch.float), torch.zeros(2, dtype=torch.float)),
                   torch.zeros(2, dtype=torch.float)))[0]  # action is a dummy action

        if initialState.terminalState:
            raise ValueError("starting in terminal state")
        else:
            self.initialState = initialState

        return

    def react(self, action):
        """
        Simulate response to an action performed by the agent.
        :param action:  action to respond to
        :return: reward and new state
        """

        # update magnet strengths according to chosen action
        self.strengths = self.strengths + action.changes

        # create lattice
        self.__createLattice(self.strengths)

        # run elegant simulation
        with open(os.devnull, "w") as f:
            sp.call(["elegant", "run.ele", "-rpnDefns=/etc/defns.rpn"], stdout=f, cwd=self.dir)

        # read elegant output
        os.chdir(self.dir)
        dataSet = SDDS(0)
        dataSet.load(self.dir + "/run.out")

        # calculate relative distance between beam focus and focus goal
        focus = torch.tensor((torch.tensor(dataSet.columnData[0]).mean(), torch.tensor(dataSet.columnData[2]).mean()))
        relCoord = focus - self.focusGoal

        # return terminal state if maximal amount of reactions exceeded
        if not self.reactCount < self.reactCountMax:
            print("forced abortion of episode, max steps exceeded")
            return State(self.strengths, focus, terminalState=True), -100
        else:
            self.reactCount += 1

        # return state and reward
        newDistanceToGoal = torch.sqrt(torch.sum(relCoord ** 2)).item()
        distanceChange = newDistanceToGoal - self.distanceToGoal
        self.distanceToGoal = newDistanceToGoal

        if newDistanceToGoal < self.acceptance:
            # goal reached
            return State(self.strengths, relCoord, terminalState=True), self.bounty
        elif torch.sqrt(torch.sum(focus ** 2)).item() >= self.targetRadius:
            # beam spot left target
            return State(self.strengths, relCoord, terminalState=True), self.penalty
        else:
            # reward according to distanceChange
            return State(self.strengths, relCoord), self.__reward(distanceChange, 10 ** 3)

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

    def __reward(self, distanceChange, scale):
        """
        Gives reward according to the change in distance to goal in the latest step.
        :param distanceChange: change in distance to goal
        :param scale: scalar value to adjust reward size
        :return: reward
        """
        return -scale * (10 ** 5 * distanceChange ** 3 + distanceChange)

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
    torch.set_default_tensor_type(torch.FloatTensor)

    env = Environment(0, 0)
    state = env.initialState
    action = Action(state, torch.tensor((0.01, 0.01)))

    eliParam = EligibleEnvironmentParameters(-0.05, 0.05, 0.01)
