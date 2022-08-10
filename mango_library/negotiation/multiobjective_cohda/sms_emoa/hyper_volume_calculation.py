from evoalgos.selection import HyperVolumeContributionSelection


def calculate_hv_total(population, global_reference_point=None):
    """
    Calculate hypervolume of a population.
    """
    selection = HyperVolumeContributionSelection()

    if global_reference_point is not None:
        reference_point = global_reference_point
    else:
        reference_point = selection.construct_ref_point(population,
                                                        selection.offsets)
    selection.sorting_component.reference_point = reference_point

    hv = selection.sorting_component.hypervolume_indicator.assess_non_dom_front(
        population)
    return hv


def calculate_hv_COHDA(performances, reference_point=None):
    """
    Calculate hypervolume of a population.
    """

    print('performances', performances)
    selection = HyperVolumeContributionSelection()

    if reference_point is None:
        reference_point = selection.construct_ref_point(performances,
                                                        selection.offsets)
    selection.sorting_component.reference_point = reference_point

    hv = selection.sorting_component.hypervolume_indicator.assess_non_dom_front(
        performances)
    return hv
