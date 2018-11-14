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
        copy("../res/run.ele", self.dir)

        # initial strengths of magnets
        self.strengthA, self.strengthB = strengthA, strengthB

        # get initial focus / initial state from them
        focus = self.react(Action(State(0, 0, 0), 0, 0))  # action is a dummy action
        self.initialState = State(self.strengthA, self.strengthB, focus)

        return

    def react(self, action):
        """simulate response to chosen action"""
        # update magnet strengths according to chosen action
        self.strengthA += action.changeA
        self.strengthB += action.changeB

        # create lattice
        self.createLattice(self.strengthA, self.strengthB)

        # run elegant simulation
        with open(os.devnull, "w") as f:
            sp.call(["elegant", "run.ele"], stdout=f, cwd=self.dir)

        # read elegant output
        os.chdir(self.dir)
        dataSet = SDDS(0)
        dataSet.load(self.dir + "/W2.SDDS")

        # calculate focus
        focus = torch.tensor((torch.tensor(dataSet.columnData[0]).mean(), torch.tensor(dataSet.columnData[2]).mean()))

        # to implement: calculate reward

        return State(self.strengthA, self.strengthB, focus)  # and reward

    def createLattice(self, strengthA, strengthB):
        """creates the lattice.lte file"""
        os.chdir(self.dir)
        with open("lattice.lte", 'w') as lattice:
            content = ("D1: Drift, L=0.5",
                       "SH1: HKICK, L=0.15, KICK={0}".format(strengthA),
                       "D2: Drift, L=0.15",
                       "SV1: VKICK, L=0.15, KICK={0}".format(strengthB),
                       "D3: Drift, L=0.5",
                       "W2: Watch, Mode=Coordinate, Filename=w2.sdds",
                       "beamline: line=(D1,SH1,D2,SV1,D3,W2)")

            content = "\n".join(content)  # create one string from content with newline symbols in between
            lattice.write(content)

            return

    def __del__(self):
        """clean up before destruction"""
        rmtree(self.dir)

        return


if __name__ == '__main__':
    env = Environment(0, 0)
    state = env.initialState
    action = Action(state, 0.1, 0.05)


