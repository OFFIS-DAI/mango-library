from enum import Enum


class InitiatingParty(Enum):
    Local = 0,
    Remote = 1


class PowerBalanceSolverStrategy:

    def solve(self, power_balance, initiator):
        pass


class Result:

    def __init__(self):
        self._solved = False
        self._solution_set = None

    def solved(self, bool=None):
        if bool is None:
            return self._solved
        else:
            self._solved = bool
