import os
import subprocess as sp
from itertools import product
from time import time
from shutil import copy, rmtree

import torch
from sdds import SDDS

from Struct import State, Action


class Environment(object):
    """represents the environment"""

    def __init__(self, strengthA, strengthB, resolutionTarget, resolutionOversize):
        """
        Constructs an environment
        :param strengthA: initial horizontal deflection
        :param strengthB: initial vertical deflection
        :param resolutionTarget: pixels used for target
        :param resolutionOversize: pixels used for space between target border and image border
        """
        self.dataSet = None
        # target image resolution
        self.resolutionTarget = resolutionTarget  # size of the image in pixel
        self.resolutionOversize = resolutionOversize  # black pixels between the target border and the image border

        # count how often the environment reacted
        self.reactCountMax = 50
        self.reactCount = 0

        # working directory
        workDir = "/dev/shm/"

        # create temporary storage
        while True:
            dir = str(int(1e7 * time()))

            if not os.path.exists(workDir + dir):
                os.makedirs(workDir + dir)
                break

        self.dir = workDir + dir

        # copy elegant config file to working directory
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        copy("../res/run.ele", self.dir)

        # define focus goal
        self.focusGoal = torch.tensor((8e-3, 8e-3), dtype=torch.float)

        self.acceptance = 5e-3  # max distance between focus goal and beam focus of a state for the state to be considered terminal
        targetDiameter = 3e-2  # diameter of target
        self.targetRadius = targetDiameter / 2

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

        self.dataSet = dataSet

        # calculate focus
        focus = torch.tensor((torch.tensor(dataSet.columnData[0]).mean(), torch.tensor(dataSet.columnData[2]).mean()))

        # calculate image
        image = self.__calcImage(dataSet)

        # return terminal state if maximal amount of reactions exceeded
        if not self.reactCount < self.reactCountMax:
            print("forced abortion of episode, max steps exceeded")
            return State(self.strengths, focus, image=image, terminalState=True), -100
        else:
            self.reactCount += 1

        # return state and reward
        distanceToGoal = torch.sqrt(torch.sum((focus - self.focusGoal) ** 2)).item()

        if distanceToGoal < self.acceptance:
            return State(self.strengths, focus, image=image, terminalState=True), 10
        elif torch.sqrt(torch.sum(focus ** 2)).item() >= self.targetRadius:
            return State(self.strengths, focus, image=image, terminalState=True), -100
        else:
            return State(self.strengths, focus, image=image), -1

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

    def __calcImage(self, elegantOutput):
        """calculates image on target from particle coordinates"""
        xCoordinate, yCoordinate = torch.tensor(elegantOutput.columnData[0], dtype=torch.float).view(-1), torch.tensor(
            elegantOutput.columnData[2], dtype=torch.float).view(-1)

        # rescale coordinates unit to [target diameter / number of pixels used to display target]
        xCoordinate *= (self.resolutionTarget - self.resolutionOversize) / (2 * self.targetRadius)
        yCoordinate *= (self.resolutionTarget - self.resolutionOversize) / (2 * self.targetRadius)

        image = torch.zeros(self.resolutionTarget + self.resolutionOversize,
                            self.resolutionTarget + self.resolutionOversize)

        countSkipped = 0

        # iterate over particles
        for particle in range(len(xCoordinate)):
            if torch.sqrt(xCoordinate[particle] ** 2 + yCoordinate[particle] ** 2) > (
                    self.resolutionTarget - self.resolutionOversize) / 2:
                # particle not on target
                countSkipped += 1

                continue

            xPixel = int(xCoordinate[particle]) + self.resolutionTarget // 2
            yPixel = int(yCoordinate[particle]) + self.resolutionTarget // 2

            image[xPixel][yPixel] += 1

        # print("number of particles {}".format(len(xCoordinate)))
        # print("skipped {} particles".format(countSkipped))

        # rescale image brightness
        image *= 255 / image.max()

        return image

    def __del__(self):
        """clean up before destruction"""
        rmtree(self.dir)

        return


def createEnvironmentParameters():
    """
    Generate a list of valid initialization parameters for Environment instances
    :return: list of valid environment parameter tuples
    """
    # deflection range to search within
    defRange = torch.arange(-0.05, 0.05, 0.01)

    # find eligible starting points
    eligibleEnvironmentParameters = []

    for x, y in product(defRange, defRange):
        try:
            env = Environment(x, y, 80, 20)  # pixel amounts are not of interest
            eligibleEnvironmentParameters.append((x, y))
        except ValueError:
            continue

    return tuple(eligibleEnvironmentParameters)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    torch.set_default_tensor_type(torch.FloatTensor)

    resolutionTarget, resolutionOversize = 80, 20

    env = Environment(0, 0, resolutionTarget, resolutionOversize)
    state = env.initialState
    action = Action(state, torch.tensor((0.046, -0.046)))
    state = env.react(action)

    # plot the image
    image = state[0].image
    fig, axes = plt.subplots(figsize=(12, 8))

    # make the axis denotation
    scaling = (env.resolutionTarget + env.resolutionOversize) / env.resolutionTarget
    notation = np.linspace(-env.targetRadius * scaling, env.targetRadius * scaling,
                           env.resolutionTarget + env.resolutionOversize, endpoint=True)

    im = axes.pcolormesh(notation, notation, image.transpose_(0, 1),
                         cmap='gray')

    plt.show()
