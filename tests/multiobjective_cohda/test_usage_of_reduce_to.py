from copy import deepcopy
from typing import List

import numpy as np

from mango_library.negotiation.multiobjective_cohda.data_classes import SystemConfig, SolutionCandidate, \
    ScheduleSelections, WorkingMemory, SolutionPoint
from mango_library.negotiation.multiobjective_cohda.multiobjective_cohda import MoCohdaNegotiation


def min_perf_func(cluster_schedules, target_params):
    performances = []
    for cs in cluster_schedules:
        performances.append(tuple(np.mean(cs, axis=0)))
    return performances


def mock_pick(solution_points: List[SolutionPoint]) -> List[SolutionPoint]:
    return []


def mock_get_hypervolume(performances, population=None):
    return -float('inf')


def test_unique_points_equals_number_solutions_points(monkeypatch):
    """
    Reduce_to is called in decide and reduces population to a given number.
    In this case, the number of unique solution points is equal to candidate.num_solution_points
    """
    global_ref_point = (1.1, 1.1)
    possible_schedules = [[0.1, 0.9], [0.1, 0.1]]
    cohda = MoCohdaNegotiation(
        schedule_provider=lambda: possible_schedules,
        is_local_acceptable=lambda s: True,
        perf_func=min_perf_func,
        reference_point=global_ref_point,
        part_id='1',
        num_iterations=1,
    )
    sysconf = SystemConfig(
        num_solution_points=1,
        schedule_choices={'1': ScheduleSelections(np.array([[1, 1], [1, 1], [1, 1]]), counter=1),
                          },
    )
    candidate = SolutionCandidate(
        num_solution_points=1,
        agent_id='1',
        schedules={'1': np.array([[1, 1], [1, 1], [1, 1]])},
        perf=[(1., 1.), (1., 1.), (1., 1.,), (1., 1.)],
        hypervolume=float('-inf'),
    )
    cohda._memory = WorkingMemory(system_config=sysconf, solution_candidate=candidate, target_params=None)
    monkeypatch.setattr(cohda, '_pick_func', mock_pick)
    monkeypatch.setattr(cohda, 'get_hypervolume', mock_get_hypervolume)
    new_sys, new_candidate = cohda._decide(
        sysconfig=deepcopy(cohda._memory.system_config), candidate=deepcopy(cohda._memory.solution_candidate))
    assert len(new_candidate.solution_points) == candidate.num_solution_points


def test_more_unique_points_than_num_points_selection_variant_adaption():
    """
    Reduce_to is called in decide and reduces population to a given number.
    In this case, the number of unique solution points is larger than candidate.num_solution_points
    and the length of all_solution_points - num_unique_solution_points is smaller than candidate.num_solution_points.
    Therefore, the way to sort (sorting variant) is manually adapted
    """
    global_ref_point = (1.1, 1.1)
    possible_schedules = [[0.1, 0.9], [0.1, 0.1]]
    cohda = MoCohdaNegotiation(
        schedule_provider=lambda: possible_schedules,
        is_local_acceptable=lambda s: True,
        perf_func=min_perf_func,
        reference_point=global_ref_point,
        part_id='1',
        num_iterations=1,
    )
    sysconf = SystemConfig(
        num_solution_points=2,
        schedule_choices={'1': ScheduleSelections(np.array([[1, 1], [1, 1]]), counter=1),
                          '2': ScheduleSelections(np.array([[0.5, 0.5], [1, 0]]), counter=1),
                          },
    )
    candidate = SolutionCandidate(
        num_solution_points=2,
        agent_id='1',
        schedules={'1': np.array([[1, 1], [1, 1]]), '2': np.array([[1, 1], [1, 1]])},
        perf=[(1., 1.), (1., 1.)],
        hypervolume=float('-inf'),
    )
    cohda._memory = WorkingMemory(system_config=sysconf, solution_candidate=candidate, target_params=None)

    new_sys, new_candidate = cohda._decide(
        sysconfig=deepcopy(cohda._memory.system_config), candidate=deepcopy(cohda._memory.solution_candidate))
    assert len(new_candidate.solution_points) == candidate.num_solution_points
    # selection variant was reset to "auto"
    assert cohda._selection.selection_variant == 'auto'


def test_more_unique_points_than_num_points_no_adaption(monkeypatch):
    """
    Reduce_to is called in decide and reduces population to a given number.
    In the third test, the number of unique solution points is larger than candidate.num_solution_points.
    """
    # the length of all_solution_points - num_unique_solution_points is larger than candidate.num_solution_points.
    global_ref_point = (1.1, 1.1)
    possible_schedules = [[0.1, 0.9], [0.1, 0.1]]
    cohda = MoCohdaNegotiation(
        schedule_provider=lambda: possible_schedules,
        is_local_acceptable=lambda s: True,
        perf_func=min_perf_func,
        reference_point=global_ref_point,
        part_id='1',
        num_iterations=1,
    )
    sysconf = SystemConfig(
        num_solution_points=2,
        schedule_choices={'1': ScheduleSelections(np.array([[1, 1], [1, 1]]), counter=1),
                          '2': ScheduleSelections(np.array([[1, 1], [1, 1]]), counter=1),
                          },
    )
    candidate = SolutionCandidate(
        num_solution_points=2,
        agent_id='1',
        schedules={'1': np.array([[1, 1], [1, 1]]), '2': np.array([[1, 1], [1, 1]])},
        perf=[(1., 1.), (1., 1.)],
        hypervolume=float('-inf'),
    )
    cohda._memory = WorkingMemory(system_config=sysconf, solution_candidate=candidate, target_params=None)

    new_sys, new_candidate = cohda._decide(
        sysconfig=deepcopy(cohda._memory.system_config), candidate=deepcopy(cohda._memory.solution_candidate))
    assert len(new_candidate.solution_points) == candidate.num_solution_points

    # the length of all_solution_points - num_unique_solution_points is equal to candidate.num_solution_points.
    global_ref_point = (1.1, 1.1)
    possible_schedules = [[0.1, 0.9], [0.1, 0.1]]
    cohda = MoCohdaNegotiation(
        schedule_provider=lambda: possible_schedules,
        is_local_acceptable=lambda s: True,
        perf_func=min_perf_func,
        reference_point=global_ref_point,
        part_id='1',
        num_iterations=1,
    )
    sysconf = SystemConfig(
        num_solution_points=2,
        schedule_choices={'1': ScheduleSelections(np.array([[1, 1], [1.1, 1]]), counter=1),
                          '2': ScheduleSelections(np.array([[1, 1], [1, 1]]), counter=1),
                          },
    )
    candidate = SolutionCandidate(
        num_solution_points=2,
        agent_id='1',
        schedules={'1': np.array([[1, 1], [1, 1]]), '2': np.array([[1, 1], [1, 1]])},
        perf=[(1., 1.), (1., 1.)],
        hypervolume=float('-inf'),
    )
    cohda._memory = WorkingMemory(system_config=sysconf, solution_candidate=candidate, target_params=None)

    new_sys, new_candidate = cohda._decide(
        sysconfig=deepcopy(cohda._memory.system_config), candidate=deepcopy(cohda._memory.solution_candidate))
    assert len(new_candidate.solution_points) == candidate.num_solution_points
