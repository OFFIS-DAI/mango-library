from copy import deepcopy

from evoalgos.selection import HyperVolumeContributionSelection


def run(population, objective_function, reduced_population_size,
        max_iterations=1,
        stopping_criterion=None):
    selection = HyperVolumeContributionSelection()
    iter = 0

    def default_stopping_criterion():
        return iter == max_iterations

    if stopping_criterion is None:
        stopping_criterion = default_stopping_criterion

    while not stopping_criterion():
        iter += 1
        # step
        # index of deleted version
        population, positions_of_removed_ind = reduce_population(selection,
                                                                 population,
                                                                 reduced_population_size)
    return population


def reduce_population(selection, population, number):
    old_population = deepcopy(population)
    positions = []
    deleted = selection.reduce_to(population=population,
                                  number=number)
    for removed_ind in deleted:
        position_of_deleted = get_idx_per_id(population, removed_ind.id)
        del old_population[position_of_deleted]
        positions.append(position_of_deleted)

    return old_population, positions


def get_idx_per_id(population, id):
    for idx, entry in enumerate(population):
        if entry.id == id:
            return idx
    raise ValueError(f'Individual with id {id} is not in population!')
