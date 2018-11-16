import os
import subprocess as sp
import torch

from time import time
from shutil import copy, rmtree
from sdds import SDDS

from Struct import State, Action


class Environment(object):
    """represents the environment"""

    def __init__(self, strengthA, strengthB):
        # working directory
        workDir = "/dev/shm/"

        # create temporary storage
        while True:
            dir = str(int(time()))

            if not os.path.exists(workDir + dir):
                os.makedirs(workDir + dir)
                break

        self.dir = workDir + dir

        # copy elegant config file to working directory
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        copy("/home/conrad/RL/TempDiff/TargetFocus/res/run.ele", self.dir)

        # define focus goal
        self.focusGoal = torch.tensor((2e-2, 2e-2), dtype=torch.float)

        self.acceptance = 5e-3  # max distance between focus goal and beam focus of a state for the state to be considered terminal
        targetDiameter = 3e-2  # diameter of target
        self.targetRadius = targetDiameter / 2

        # initial strengths of magnets
        self.strengths = torch.tensor((strengthA, strengthB), dtype=torch.float)

        # get initial focus / initial state from them
        initialState = self.react(Action(State(torch.tensor((strengthA, strengthB), dtype=torch.float), torch.zeros(2, dtype=torch.float)), torch.zeros(2, dtype=torch.float)))[0]  # action is a dummy action

        if initialState.terminalState:
            raise ValueError("starting in terminal state")
        else:
            self.initialState = initialState

        return

    def react(self, action):
        """simulate response to chosen action"""
        # update magnet strengths according to chosen action
        self.strengths = self.strengths + action.changes

        # create lattice
        self.__createLattice(self.strengths)

        # run elegant simulation
        with open(os.devnull, "w") as f:
            sp.call(["elegant", "run.ele"], stdout=f, cwd=self.dir)

        # read elegant output
        os.chdir(self.dir)
        dataSet = SDDS(0)
        dataSet.load(self.dir + "/run.out")

        # calculate focus
        focus = torch.tensor((torch.tensor(dataSet.columnData[0]).mean(), torch.tensor(dataSet.columnData[2]).mean()))

        # return state and reward
        distanceToGoal = torch.sqrt(torch.sum((focus - self.focusGoal) ** 2)).item()

        if distanceToGoal < self.acceptance:
            return State(self.strengths, focus, terminalState=True), 1
        elif torch.sqrt(torch.sum(focus ** 2)).item() >= self.targetRadius:
            return State(self.strengths, focus, terminalState=True), -10
        else:
            return State(self.strengths, focus), -1

    def __createLattice(self, strengths):
        """creates the lattice.lte file"""
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


if __name__ == '__main__':
    torch.set_default_tensor_type(torch.FloatTensor)

    env = Environment(0, 0)
    state = env.initialState
    action = Action(state, torch.tensor((0.01, 0.01)))
