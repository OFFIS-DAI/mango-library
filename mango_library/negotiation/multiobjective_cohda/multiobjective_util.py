import random
import warnings
from copy import deepcopy
from typing import List, Tuple

from evoalgos.selection import HyperVolumeContributionSelection

from mango_library.negotiation.multiobjective_cohda.sms_emoa.selection import get_worst_individual


def get_hypervolume(performances: List[Tuple[float, ...]],
                    reference_point: Tuple[float, ...]) -> float:
    """
    This function calculates the hypervolume from a list of performance tuples with a given reference point
    :param performances:
    :param reference_point:
    :return:
    """
    #  ----- START DUMMY ALLERT -----

    total_performance = 0
    for candidate_performance in performances:
        for objective_performance in candidate_performance:
            total_performance += objective_performance

    #  ----- END DUMMY ALLERT -----

    return total_performance


def get_hypervolume_sms_emoa(performances: List[Tuple[float, ...]],
                             reference_point: Tuple[float, ...]) -> float:
    """
    Calculate hypervolume of a population.
    """

    selection = HyperVolumeContributionSelection()

    if reference_point is None:
        warnings.warn(
            "There is no global reference point given to calculate the "
            "hypervolume! This should not be the case!")
    selection.sorting_component.reference_point = reference_point

    hv = selection.sorting_component.hypervolume_indicator.assess_non_dom_front(
        performances)
    return hv


def get_index_of_worst(performances: List[Tuple[float, ...]]) -> int:
    """
    This function returns the index of the performance tuple with the lowest impact on hypervolume.
    :param performances:
    :return:
    """
    #  ----- START DUMMY ALLERT -----

    worst_contribution = None
    worst_index = None
    for i, candidate_performance in enumerate(performances):
        contribution = sum(candidate_performance)
        if worst_contribution is None or contribution < worst_contribution:
            worst_contribution = contribution
            worst_index = i

    return worst_index

    #  ----- END DUMMY ALLERT -----


def get_index_of_worst_sms_emoa(performances,
                                reference_point: float) -> int:
    old_population = deepcopy(performances)
    new_population = deepcopy(performances)
    # create new candidate with all objective values
    # other parameters as agent_id and candidate are default values, since
    # only the objective values are nedeed, but the data structure is relevant
    population_set = set(new_population)

    random.shuffle(new_population)

    deleted = \
        get_worst_individual(population_set, new_population, reference_point)[0]
    idx = get_position_in_population(deleted, old_population)

    return idx


def get_position_in_population(individual, population):
    for idx, ind in enumerate(population):
        if ind.id == individual.id:
            return idx
    raise ValueError('Indiviual was never in population! Position can not'
                     'be determined!')
