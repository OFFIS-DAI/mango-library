from enum import Enum


class InitiatingParty(Enum):
    Local = 0,
    Remote = 1


class PowerBalanceSolverStrategy:

    def solve(self, power_balance, initiator):
        pass

    @staticmethod
    def find_initial_requirement(power_balance, initiator):
        """
           Finds the initial requirement based on the given power balance and initiator.

           Parameters: power_balance (list): A list of tuples representing the power balance. Each tuple consists of
           two elements: - The first element is the power value. - The second element is an object containing
           attributes related to the power source. initiator (InitiatingParty): An enum representing the initiator of
           the process. It can be either InitiatingParty.Local or InitiatingParty.Remote.

           Returns: Union[Requirement, ForecastRequirement]: The initial requirement extracted from the power
           balance. If the initial requirement is represented as a tuple, it returns the second element of the tuple,
           which is an instance of Requirement or ForecastRequirement depending on the input.

           Raises:
               AssertionError: If the extracted initial requirement is None.

           Note: - For local initiators, the method searches for the first power balance item where the 'from_target'
           attribute is True. - For remote initiators, the method selects the initial requirement based on certain
           criteria from the power balance list.

           """
        print(power_balance)
        print(initiator)
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
