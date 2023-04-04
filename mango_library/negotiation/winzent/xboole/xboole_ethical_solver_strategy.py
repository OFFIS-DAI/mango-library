from collections import namedtuple

from mango_library.negotiation.winzent import xboole
from mango_library.negotiation.winzent.xboole import PowerBalanceSolverStrategy


class XbooleEthicalPowerBalanceSolverStrategy(PowerBalanceSolverStrategy):
    def __init__(self):
        self.power_balance_strategy = xboole.XboolePowerBalanceSolverStrategy()


    def sum_up_offers(self, power_balance):
        sum = 0
        for value in power_balance._ledger[0]:
            sum += value[0].message.value[0]
        return sum

    def get_ethics_score(self, requirement):
        return requirement.message.ethics_score


    def solve(self, power_balance, initiator):
        # if all the available offers cannot satisfy the need for this timeslot,
        # no special solving strategy is needed
        if curr_requirement_value > self.sum_up_offers(power_balance):
            return self.power_balance_strategy.solve(power_balance, initiator,)
        else:
            most_ethical_requirements = power_balance
            most_ethical_requirements._ledger[0].sort(keyword=self.get_ethics_score)

