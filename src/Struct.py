# data class


class State(object):
    __slots__ = ['strengths', 'focus', 'terminalState']

    def __init__(self, strengths, focus, terminalState=False):
        self.strengths = strengths
        self.focus = focus
        self.terminalState = terminalState

        return

    def __repr__(self):
        return "State(strengths={}, focus={})".format(self.strengths, self.focus)

    def __bool__(self):
        return self.terminalState


class Action(object):
    __slots__ = ['state', 'changes', 'eligibility']

    def __init__(self, state, changes):
        self.state = state
        self.changes = changes
        self.eligibility = 1

        return

    def __repr__(self):
        return "Action({}, changes={}".format(self.state, self.changes)


class Transition(object):
    __slots__ = ['action', 'reward', 'nextState']

    def __init__(self, action, reward, nextState):
        self.action = action
        self.reward = reward
        self.nextState = nextState

    def __repr__(self):
        return "Transition from {0} with reward {1} to {2}".format(self.action, self.reward, self.nextState)

    def __mul__(self, other):
        self.action.eligibility *= other
        return
