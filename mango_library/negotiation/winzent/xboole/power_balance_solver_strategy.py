from enum import Enum


class InitiatingParty(Enum):
    Local = 0,
    Remote = 1


class PowerBalanceSolverStrategy:

    def solve(self, power_balance, initiator):
        pass

    @staticmethod
    def find_initial_requirement(power_balance, initiator):
        r = None
        if initiator == InitiatingParty.Local:
            for i in power_balance:
                if i[1].from_target:
                    r = i[1]
                    break

        else:
            pb = []
            for i in power_balance:
                pb.append(i)
            r = pb[0]
            for i in range(1, len(pb) - 1):
                if (abs(r[1].forecast.second) <= abs(pb[i][1].forecast.second)) \
                        and ((abs(r[0]) * 1000)
                             <= (abs(pb[i][0] * 1000))):
                    r = pb[i]

        assert r is not None
        if isinstance(r, tuple):
            return r[1]
        return r


class Result:

    def __init__(self):
        self._solved = False
        self._solution_set = None

    def solved(self, bool=None):
        if bool is None:
            return self._solved
        else:
            self._solved = bool
