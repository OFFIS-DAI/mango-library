import random
import warnings

from evoalgos.selection import TruncationSelection, INFINITY, \
    HyperVolumeContributionSelection
from evoalgos.sorting import LexicographicSorting


def get_worst_individual(population_set, population, reference_point=None):
    selection = HyperVolumeContributionSelection(prefer_boundary_points=False)
    number = len(population) - 1

    all_individuals = population

    selection_variant = selection.selection_variant
    if selection_variant == "auto" or selection_variant is None:
        if len(population) > 2 * number:
            selection_variant = "forward-greedy"
        else:
            selection_variant = "backward-greedy"
    hv_sorting = selection.sorting_component
    compute_fronts = hv_sorting.non_dom_sorting.identify_groups
    # the reference point is obtained from the population
    if reference_point is None:
        warnings.warn(
            "There is no global reference point given to calculate the "
            "worst individual! This should not be the case!")

    hv_sorting.reference_point = reference_point
    # check for easier special case
    is_2d = True
    for individual in all_individuals:
        is_2d &= len(individual.objective_values) == 2
    if is_2d:
        sort_front = hv_sorting.sort_front_2d
        calc_contributions = hv_sorting.hypervolume_indicator.calc_contributions_2d
    else:
        sort_front = hv_sorting.sort_front
        calc_contributions = hv_sorting.hypervolume_indicator.calc_contributions
    fronts = compute_fronts(all_individuals)
    selected = []
    rejected = []
    if selection_variant == "forward-greedy":
        for front in fronts:
            if len(selected) + len(front) <= number:
                # accept all in this front
                for ind in front:
                    if ind in population_set:
                        selected.append(ind)
            elif len(selected) >= number:
                # reject all in this front
                for ind in front:
                    if ind in population_set:
                        rejected.append(ind)
            else:
                selected_this_front = []
                prefer_boundary_points = hv_sorting.prefer_boundary_points and is_2d
                if prefer_boundary_points:
                    LexicographicSorting().sort(front)
                remaining_indices = list(range(len(front)))
                max_index = remaining_indices[-1]
                while len(selected) < number and remaining_indices:
                    hv_values = []
                    random.shuffle(remaining_indices)
                    for i in remaining_indices:
                        if prefer_boundary_points and i in (0, max_index):
                            hv_values.append(INFINITY)
                        else:
                            contribs = calc_contributions([front[i]],
                                                          others=selected_this_front)
                            hv_values.append(contribs[front[i]])
                    best_index = max(range(len(remaining_indices)),
                                     key=hv_values.__getitem__)
                    best_ind = front[remaining_indices[best_index]]
                    remaining_indices.pop(best_index)
                    selected_this_front.append(best_ind)
                    if best_ind in population_set:
                        selected.append(best_ind)
                for i in remaining_indices:
                    if front[i] in population_set:
                        rejected.append(front[i])
        population[:] = selected
        return rejected
    elif selection_variant == "backward-greedy":
        while len(rejected) < len(population) - number:
            last_front = fronts[-1]
            # sort the last front by hypervolume contribution
            if len(last_front) > 1:
                random.shuffle(last_front)
                sort_front(last_front)
            # remove worst and update fronts to avoid recalculation
            removed = last_front.pop()
            if removed in population_set:
                rejected.append(removed)
            if len(last_front) == 0:
                fronts.pop()
        for front in fronts:
            for ind in front:
                if ind in population_set:
                    selected.append(ind)
        population[:] = selected
        return rejected
    elif selection_variant == "super-greedy":
        return TruncationSelection.reduce_to(selection, population, number,
                                             already_chosen=None)
    else:
        raise ValueError(
            "unknown selection variant: " + str(selection_variant))
