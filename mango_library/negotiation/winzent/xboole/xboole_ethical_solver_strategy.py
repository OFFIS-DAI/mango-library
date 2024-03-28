import math
from copy import deepcopy
from itertools import combinations

from mango_library.negotiation.winzent import xboole
from mango_library.negotiation.winzent.xboole import PowerBalanceSolverStrategy


def get_ethics_score_from_req(requirement):
    return requirement.message.ethics_score


def find_groups(arr, group_size):
    result = []
    for combo in combinations(arr, group_size):
        result.append(list(combo))
    return result


def get_ethics_score_of_final_solution(final, req_list, start_time):
    req_list_absolute_power = 0
    afforded_values_per_agent = {}
    for key in final:
        afforded_values_per_agent[key] = 0
        for sub_key in final[key]:
            req_list_absolute_power += abs(final[key][sub_key][0])
            afforded_values_per_agent[key] += (abs(final[key][sub_key][0]))
    it = 0
    ethics_score_sum = 0
    for req in req_list.ledger[start_time]:
        if req.message.sender not in afforded_values_per_agent.keys():
            afforded_values_per_agent[req.message.sender] = 0
        ethics_score_sum += get_ethics_score_from_req(req) * \
                            (afforded_values_per_agent[req.message.sender] /
                             req_list_absolute_power)
        it += 1
    return ethics_score_sum


def get_normalized_ethics_score_of_req_list(final, req_list, full_req_list, start_time, initial_req):
    req_list_score = get_ethics_score_of_final_solution(final, req_list, start_time)
    min_value, max_value = get_ethics_score_min_and_max(full_req_list, initial_req)
    if min_value >= max_value:
        return 1.0
    return (req_list_score - min_value) / (max_value - min_value)


def get_ethics_score_min_and_max(req_list, initial_req):
    lowest_ethics_score = 0
    highest_ethics_score = 0
    for req in req_list:
        if req[1] != initial_req:
            if lowest_ethics_score == 0 and highest_ethics_score == 0:
                lowest_ethics_score = req[1].message.ethics_score
                highest_ethics_score = req[1].message.ethics_score
            else:
                if req[1].message.ethics_score < lowest_ethics_score:
                    lowest_ethics_score = req[1].message.ethics_score
                if req[1].message.ethics_score > highest_ethics_score:
                    highest_ethics_score = req[1].message.ethics_score
    return math.floor(lowest_ethics_score), math.ceil(highest_ethics_score)


def calc_solution_coverage(afforded_values, initial_values):
    it = 0
    coverage = 0
    for value in afforded_values.values():
        coverage += value / initial_values[it]
        it += 1
    return coverage / it


def are_all_offers_in_same_time_span(req_list):
    time_span = [-1, 0]
    for req in req_list:
        if time_span == [-1, 0]:
            time_span = req[1].message.time_span
        elif time_span != req[1].message.time_span or len(time_span) > 1:
            return False
    return True


class XbooleEthicalPowerBalanceSolverStrategy(PowerBalanceSolverStrategy):
    """
    The solving strategy for the ethics module. Contains the algorithm to create new solutions by
    deliberately cutting away requirements. It then compares these solutions to one another based
    on a weighted sum equation.
    :param min_coverage: The minimum coverage a solution must reach to be eligible
    :param coverage_weight: The weight of the coverage ratio in the weighted sum equation
    :param ethics_score_weight: The weight of the ethics score in the weighted sum equation
    """

    def __init__(self, min_coverage=1.0, coverage_weight=0.9, ethics_score_weight=0.1):
        self.power_balance_strategy = xboole.XboolePowerBalanceSolverStrategy()
        self.initial_requirement = None
        self.start_time = 0
        self.min_coverage = min_coverage
        self.req_removal_depth = 1
        self.coverage_weight = coverage_weight
        self.ethics_score_weight = ethics_score_weight

    def create_list_of_agents(self, req_list):
        agent_list = []
        for req in req_list.ledger[self.start_time]:
            if req != self.initial_requirement:
                agent_list.append(req)
        return agent_list

    def add_agents_to_final(self, req_list, final):
        new_final = {}
        for req in req_list.ledger[self.start_time]:
            for key in final.keys():
                if req.message.id == key:
                    new_final[req.message.sender] = final[key]
        return new_final

    def calc_solution_quality(self, final, afforded_values, initial_values, full_req_list, req_list):
        """
        This method calculates the quality score of a solution with a weighted sum equation.
        :param final: The composition of the final solution
        :param afforded_values: The values reached in the negotiation
        :param initial_values: The values set out to be reached at the start of the negotiation
        :param full_req_list: A list of all the requirements that came in.
        :param req_list: A list of all the requirements that were used in this particular solution.
        :return: the solution quality score
        """
        temp_req_list = deepcopy(req_list)
        temp_req_list.ledger[self.start_time].remove(self.initial_requirement)
        coverage = calc_solution_coverage(afforded_values, initial_values)
        temp_final = self.add_agents_to_final(full_req_list, final)
        normalized_ethics_score = get_normalized_ethics_score_of_req_list(temp_final, req_list, full_req_list,
                                                                          self.start_time,
                                                                          self.initial_requirement)

        if coverage >= self.min_coverage:
            return coverage * self.coverage_weight + normalized_ethics_score * self.ethics_score_weight
        else:
            return -1

    def ethical_solution_algorithm(self, req_list, initiator):
        """
        This method creates solutions by cutting away replies and using the solver contained in
        XboolePowerBalanceSolverStrategy with that new composition of replies.
        It then sorts the solution based on ethics score and coverage and picks the best one based
        on a weighted sum equation.
        :param req_list: The replies for the negotiation.
        :param initiator: The initiator for the boolean solver
        """
        print(f"ethical solution alg running!")
        final, afforded_values, initial_req = self.power_balance_strategy.solve(req_list, initiator)
        initial_values = initial_req.forecast.second
        initial_sol_score = self.calc_solution_quality(final, afforded_values, initial_values, req_list, req_list)
        if initial_sol_score == -1:
            print("Min. coverage undercut!")
            # returns the initial solution since the minimum coverage can already not be fulfilled
            return final, afforded_values, initial_req
        else:
            temp_sol_dict = {}
            full_req_list = deepcopy(req_list)
            reqs_to_be_removed = 1
            agent_list = self.create_list_of_agents(full_req_list)
            while reqs_to_be_removed <= self.req_removal_depth:
                # creates the possible combinations for the group size in reqs_to_be_removed
                possible_req_combinations = find_groups(agent_list, reqs_to_be_removed)
                for combination in possible_req_combinations:
                    removed_reqs = []
                    removed_reqs_ids = []
                    for req in combination:
                        req_index = req_list.ledger[self.start_time].index(req)
                        removed_reqs.append(req_list.ledger[self.start_time].pop(req_index))
                        removed_reqs_ids.append(req.message.id)
                    if len(req_list.ledger[self.start_time]) < 2:
                        temp_sol_dict[tuple(removed_reqs_ids)] = ([{}, None, None], -1)
                        continue
                    else:
                        temp_final, temp_afforded_values, temp_initial_req = self.power_balance_strategy.solve(
                            req_list, initiator)
                    # adds created solution solution dictionary
                    temp_sol_dict[tuple(removed_reqs_ids)] = (
                        [temp_final, temp_afforded_values, temp_initial_req],
                        self.calc_solution_quality(temp_final,
                                                   temp_afforded_values,
                                                   initial_values,
                                                   full_req_list,
                                                   req_list))
                    for removed_req in removed_reqs:
                        # adds the removed requirements back to the requirement list
                        req_list.ledger[self.start_time].append(removed_req)
                    removed_reqs.clear()
                    removed_reqs_ids.clear()
                # increases the requirements to be removed by one, starting a new cycle
                reqs_to_be_removed += 1
            highest_sol_score = initial_sol_score
            # solution scores are compared with one another
            for key in temp_sol_dict.keys():
                if temp_sol_dict[key][1] > highest_sol_score:
                    highest_sol_score = temp_sol_dict[key][1]
                    final = temp_sol_dict[key][0][0]
                    afforded_values = temp_sol_dict[key][0][1]
                    initial_req = temp_sol_dict[key][0][2]
            return final, afforded_values, initial_req

    def solve(self, power_balance, initiator):
        """
        This method either starts the solving process for multiple time slots or just
        sorts the requirements based on their ethics score and picks the top ones
        until the negotiation target is satisfied.
        :param power_balance: The replies for the negotiation.
        :param initiator: The initiator for the boolean solver
        """
        print("solve inside ethical solver running")
        # in this case, the agent did not receive any offers and the only requirement in the power balance is
        # his own. Consequently, no solution can be created.
        try:
            if len(power_balance.ledger[self.start_time]) < 2:
                print("no offers received. Aborting solution process")
                return {}, None, None
        except Exception as e:
            print(e)
        self.initial_requirement = PowerBalanceSolverStrategy.find_initial_requirement(power_balance, initiator)
        print(f"initial requirement done: {self.initial_requirement.message.value}")
        # if all the available offers cannot satisfy the need for this timeslot,
        # no special solving strategy is needed since all will be accepted anyway
        # in this case, there are more offers than needed to satisfy the initial requirement
        most_ethical_requirements = deepcopy(power_balance)
        try:
            most_ethical_requirements.ledger[self.start_time].remove(self.initial_requirement)
        except Exception as e:
            print("JUP; EXCEPTION TIME!")
            print(e)
        print("removed initial req")
        most_ethical_requirements.ledger[self.start_time].sort(key=get_ethics_score_from_req, reverse=True)
        print("reqs sorted")
        time_span = self.initial_requirement.time_span
        print("checking if all offers are of same time span")
        if not are_all_offers_in_same_time_span(most_ethical_requirements):
            most_ethical_requirements.ledger[self.start_time].append(self.initial_requirement)
            return self.ethical_solution_algorithm(most_ethical_requirements, initiator)
        else:
            initial_values = []
            # find the relevant values in the initial requirement that fit the time frames of the offer requirements
            for time_slot in self.initial_requirement.message.time_span:
                if time_slot in time_span:
                    value_index = list(self.initial_requirement.message.time_span).index(time_slot)
                    initial_values.append(abs(self.initial_requirement.message.value[value_index]))
            # requirements are sorted by their ethics score, so that the most ethical
            # offers are transferred to the solving algorithm
            it = 0
            # cut off unnecessary offers with low ethics score
            for req in most_ethical_requirements.ledger[self.start_time]:
                if all(initial_values) <= 0:
                    break
                value_index = 0
                for value in req.message.value:
                    initial_values[value_index] = initial_values[value_index] - value
                    value_index += 1
                it += 1
            most_ethical_requirements.ledger[self.start_time] = most_ethical_requirements.ledger[self.start_time][
                                                                0:it]
            # add initial requirement to ensure the functioning of the solving algorithm
            most_ethical_requirements.ledger[self.start_time].append(self.initial_requirement)
            return self.power_balance_strategy.solve(most_ethical_requirements, initiator)
