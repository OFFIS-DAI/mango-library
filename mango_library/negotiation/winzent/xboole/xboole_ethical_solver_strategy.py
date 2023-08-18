import math
from collections import namedtuple
from datetime import datetime
from copy import deepcopy
from mango_library.negotiation.winzent import xboole
from mango_library.negotiation.winzent.xboole import PowerBalanceSolverStrategy


def get_ethics_score_from_requirement(requirement):
    return requirement.message.ethics_score


def normalize_ethics_score(value, req_list):
    min_value, max_value = get_ethics_score_min_and_max(req_list)
    return (value - min_value) / (max_value - min_value)


def get_ethics_score_min_and_max(req_list):
    lowest_ethics_score = 0
    highest_ethics_score = 0
    for req in req_list:
        if lowest_ethics_score == 0 and highest_ethics_score == 0:
            lowest_ethics_score = req.message.ethics_score
            highest_ethics_score = req.message.ethics_score
        else:
            if req.message.ethics_score < lowest_ethics_score:
                lowest_ethics_score = req.message.ethics_score
            if req.message.ethics_score > highest_ethics_score:
                highest_ethics_score = req.message.ethics_score
    return math.floor(lowest_ethics_score), math.ceil(highest_ethics_score)


def all_offers_in_same_time_span(req_list):
    time_span = [-1, 0]
    for req in req_list:
        if time_span == [-1, 0]:
            time_span = req[1].message.time_span
        elif time_span != req[1].message.time_span:
            return False
    return True


class XbooleEthicalPowerBalanceSolverStrategy(PowerBalanceSolverStrategy):
    def __init__(self, coverage_weight=0.9, ethics_score_weight=0.1):
        self.power_balance_strategy = xboole.XboolePowerBalanceSolverStrategy()
        self.initial_requirement = None
        self.start_time = 0
        self.coverage_weight = coverage_weight
        self.ethics_score_weight = ethics_score_weight

    def sum_up_offers(self, power_balance):
        sum = 0
        for start_time, req_list in power_balance._ledger.items():
            self.start_time = start_time
            for req in req_list:
                sum = sum + abs(req.message.value[0])
        return sum

    def calculate_solution_quality(self):
        print("noch nicht fertig")

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
        # no special solving strategy is needed since all will be accepted anyway
        offer_sum = abs(self.sum_up_offers(power_balance))
        initial_value = abs(self.initial_requirement.power)
        if initial_value > offer_sum:
            return self.power_balance_strategy.solve(power_balance, initiator)
        # in this case, there are more offers than needed to satisfy the initial requirement
        most_ethical_requirements = deepcopy(power_balance)
        most_ethical_requirements.ledger[self.start_time].remove(self.initial_requirement)
        if not all_offers_in_same_time_span(most_ethical_requirements):
            print("noch nicht fertig")
        else:
            # requirements are sorted by their ethics score, so that the most ethical
            # offers are transferred to the solving algorithm
            most_ethical_requirements.ledger[self.start_time].sort(key=get_ethics_score_from_requirement, reverse=True)
            temp_sum = 0
            index = 0
            # cut off unnecessary offers with low ethics score
            for req in most_ethical_requirements.ledger[self.start_time]:
                if temp_sum + abs(req.message.value[0]) >= initial_value:
                    index += 1
                    break
                temp_sum += abs(req.message.value[0])
                index += 1
            most_ethical_requirements.ledger[self.start_time] = most_ethical_requirements.ledger[self.start_time][
                                                                0:index]
            # add initial requirement to ensure the functioning of the solving algorithm
            most_ethical_requirements.ledger[self.start_time].append(self.initial_requirement)
            return self.power_balance_strategy.solve(most_ethical_requirements, initiator)
