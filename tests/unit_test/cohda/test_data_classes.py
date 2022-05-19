from mango_library.negotiation.cohda.data_classes import SystemConfig, SolutionCandidate, ScheduleSelection
import numpy as np
from typing import Dict, List, Tuple
import pytest


def test_sysconf_init():
    selections_1 = ScheduleSelection(schedule=np.array([1, 2, 3]), counter=42)
    selections_2 = ScheduleSelection(schedule=np.array([4, 2, 1]), counter=4)
    sysconf = SystemConfig(schedule_choices={1: selections_1, 2: selections_2})
    assert sysconf.schedule_choices == {1: selections_1, 2: selections_2}


def test_candidate_init():
    schedules = {1: np.array([1, 2, 3]), 2: np.array([4, 5, 6])}
    candidate = SolutionCandidate(agent_id=1, schedules=schedules, perf=2)
    assert candidate.agent_id == 1
    assert candidate.schedules == schedules
    assert candidate.perf == 2


@pytest.mark.parametrize(
    "schedules_i, schedules_j, expected",
    [
        ({1: ([1, 2], 42), 2: ([4, 2], 4)}, {1: ([10, 20], 45), 2: ([40, 20], 5)}, False),
        ({1: ([1, 2], 42), 2: ([4, 2], 4)}, {1: ([1, 2], 42), 2: ([4, 2], 4)}, True),
        ({1: ([1, 2], 42)}, {1: ([1, 2], 42), 2: ([4, 2], 4)}, False),
        ({1: ([1, 2], 42), 2: ([4, 2], 4)}, {1: ([1, 2], 42), 2: ([4, 2], 5)}, False),
        ({1: ([1, 2], 42), 2: ([4, 2], 4)}, {1: ([1, 2], 42), 3: ([4, 2], 4)}, False),
    ]
)
def test_sysconf_equal(schedules_i: Dict[int, Tuple[List, int]], schedules_j: Dict[int, Tuple[List, int]], expected: bool):
    schedule_selections_i = {}
    schedule_selections_j = {}
    for part_id, (schedule, counter) in schedules_i.items():
        schedule_selections_i[part_id] = ScheduleSelection(schedule=np.array(schedule), counter=counter)

    for part_id, (schedule, counter) in schedules_j.items():
        schedule_selections_j[part_id] = ScheduleSelection(schedule=np.array(schedule), counter=counter)

    sysconfig_i = SystemConfig(schedule_choices=schedule_selections_i)
    sysconfig_j = SystemConfig(schedule_choices=schedule_selections_j)

    assert (sysconfig_i == sysconfig_j) == expected

@pytest.mark.parametrize(
    "schedules_i, schedules_j, expected_schedules",
    [
        ({1: ([1, 2], 42), 2: ([4, 2], 4)}, {1: ([10, 20], 45), 2: ([40, 20], 5)},
         {1: ([10, 20], 45), 2: ([40, 20], 5)}),
        ({1: ([1, 2], 42)}, {1: ([10, 20], 40), 2: ([40, 20], 5)}, {1: ([1, 2], 42), 2: ([40, 20], 5)}),
        ({1: ([1, 2], 42)}, {2: ([40, 20], 5)}, {1: ([1, 2], 42), 2: ([40, 20], 5)}),
        ({1: ([1, 2], 42)}, {1: ([40, 20], 5)}, {1: ([1, 2], 42)}),
        ({1: ([1, 2], 42), 2: ([40, 20], 5)}, {1: ([1, 2], 42), 2: ([40, 20], 5)},
         {1: ([1, 2], 42), 2: ([40, 20], 5)}),
    ]
)
def test_sysconf_merge(schedules_i: Dict[int, Tuple[List, int]], schedules_j: Dict[int, Tuple[List, int]], expected_schedules):
    schedule_selections_i = {}
    schedule_selections_j = {}
    expected_selections = {}
    for part_id, (schedule, counter) in schedules_i.items():
        schedule_selections_i[part_id] = ScheduleSelection(schedule=np.array(schedule), counter=counter)

    for part_id, (schedule, counter) in schedules_j.items():
        schedule_selections_j[part_id] = ScheduleSelection(schedule=np.array(schedule), counter=counter)

    for part_id, (schedule, counter) in expected_schedules.items():
        expected_selections[part_id] = ScheduleSelection(schedule=np.array(schedule), counter=counter)

    sysconfig_i = SystemConfig(schedule_choices=schedule_selections_i)
    sysconfig_j = SystemConfig(schedule_choices=schedule_selections_j)
    expected_sysconfig = SystemConfig(schedule_choices=expected_selections)

    merged_sysconfig = SystemConfig.merge(sysconfig_i, sysconfig_j)
    assert merged_sysconfig == expected_sysconfig
    assert (sysconfig_i == merged_sysconfig) == (sysconfig_i is merged_sysconfig)


def test_sysconf_cluster_schedule():
    selections_1 = ScheduleSelection(schedule=np.array([1, 2, 3]), counter=42)
    selections_2 = ScheduleSelection(schedule=np.array([4, 2, 1]), counter=4)
    sysconf = SystemConfig({2: selections_2, 1: selections_1})
    assert np.array_equal(sysconf.cluster_schedule, np.array([[4, 2, 1], [1, 2, 3]]))
