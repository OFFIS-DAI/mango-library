from collections import namedtuple

from mango_library.negotiation.winzent import xboole
from mango_library.negotiation.winzent.xboole import PowerBalanceSolverStrategy

class XbooleEthicalPowerBalanceSolverStrategy(PowerBalanceSolverStrategy):
    def __init__(self):
        self.power_balance_strategy = xboole.XboolePowerBalanceSolverStrategy()
        self.initial_requirement = None

    def sum_up_offers(self, power_balance):
        sum = 0
        for req in power_balance._ledger[0]:
            sum = sum + req.power
        return sum

    def get_ethics_score(self, requirement):
        return requirement.message.ethics_score


    def solve(self, power_balance, initiator):
        """
        Currently, this method sorts the available offers based on their ethics score
        and cuts off the unnecessary ones with the lowest score if enough are
        available to satisfy the initial requirement.
        Afterwards, the offers are handed to the solving algorithm.
        5.4.2023
        """
        self.initial_requirement = PowerBalanceSolverStrategy.find_initial_requirement(power_balance, initiator)
        # if all the available offers cannot satisfy the need for this timeslot,
        # no special solving strategy is needed
        sum = abs(self.sum_up_offers(power_balance))
        ini = abs(self.initial_requirement.power)
        if ini > sum:
            return self.power_balance_strategy.solve(power_balance, initiator)
        else:
            # in this case, there are more offers than needed to satisfy the initial requirement
            most_ethical_requirements = power_balance
            most_ethical_requirements._ledger[0].remove(self.initial_requirement)
            # requirements are sorted by their ethics score, so that the most ethical
            # offers are transferred to the solving algorithm
            most_ethical_requirements._ledger[0].sort(key=self.get_ethics_score, reverse=True)
            temp_sum = 0
            index = 0
            # cut off unnecessary offers with low ethics score
            for req in most_ethical_requirements._ledger[0]:
                if temp_sum + req.power >= ini:
                    index += 1
                    break
                temp_sum += req.power
                index += 1
            most_ethical_requirements._ledger[0] = most_ethical_requirements._ledger[0][0:index]
            # add initial requirement to ensure the functioning of the solving algorithm
            most_ethical_requirements._ledger[0].append(self.initial_requirement)
            return self.power_balance_strategy.solve(most_ethical_requirements, initiator)

