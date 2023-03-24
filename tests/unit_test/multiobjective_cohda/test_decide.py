from copy import deepcopy

import numpy as np
import pytest

from mango_library.negotiation.multiobjective_cohda.data_classes import SystemConfig, SolutionCandidate, \
    ScheduleSelections, WorkingMemory
from mango_library.negotiation.multiobjective_cohda.multiobjective_cohda import MoCohdaNegotiation


def min_perf_func(cluster_schedules, target_params):
    performances = []
    for cs in cluster_schedules:
        performances.append(tuple(np.mean(cs, axis=0)))
    return performances


@pytest.fixture
def initial_working_memory():
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

    return WorkingMemory(system_config=sysconf, solution_candidate=candidate, target_params=None)


@pytest.fixture
def cohda(initial_working_memory):
    global_ref_point = (1.1, 1.1)
    possible_schedules = [[0.1, 0.9], [0.1, 0.1]]
    my_cohda = MoCohdaNegotiation(
        schedule_provider=lambda: possible_schedules,
        is_local_acceptable=lambda s: True,
        perf_func=min_perf_func,
        reference_point=global_ref_point,
        part_id='1',
        num_iterations=1,
    )
    my_cohda._memory = initial_working_memory
    return my_cohda


def test_decide(cohda):
    new_sys, new_candidate = cohda._decide(
        sysconfig=deepcopy(cohda._memory.system_config), candidate=deepcopy(cohda._memory.solution_candidate))

    assert new_sys != cohda._memory.system_config
    assert np.array_equal(np.array([[0.1, 0.1], [0.1, 0.1]]), new_candidate.schedules['1']) or np.array_equal(
        np.array([[0.1, 0.1], [0.1, 0.1]])
    )
