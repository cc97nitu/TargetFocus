# data class


class State(object):
    __slots__ = ['strengthA', 'strengthB', 'focus', 'terminalState']

    def __init__(self, strengthA, strengthB, focus, terminalState=False):
        self.strengthA = strengthA
        self.strengthB = strengthB
        self.focus = focus
        self.terminalState = terminalState

        return

    def __repr__(self):
        return "State (strengthA={}, strengthB={}, focus={})".format(self.strengthA, self.strengthB, self.focus)

    def __bool__(self):
        return self.terminalState


class Action(object):
    __slots__ = ['state', 'changeA', 'changeB', 'eligibility']

    def __init__(self, state, changeA, changeB):
        self.state = state
        self.changeA = changeA
        self.changeB = changeB
        self.eligibility = 1

        return

    def __repr__(self):
        return "Action ({}, changeA={}, changeB={})".format(self.state, self.changeA, self.changeB)
