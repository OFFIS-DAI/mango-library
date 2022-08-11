import fractions

from mango_library.negotiation.cohda.data_classes import SystemConfig,\
    SolutionCandidate, ScheduleSelection, WorkingMemory
from mango_library.negotiation.cohda.cohda import CohdaMessage, COHDA
from mango_library.negotiation.util import cohda_serializers
from mango_library.negotiation.core import *
from mango.messages.codecs import *
import numpy as np
from typing import Dict, List, Tuple
import pytest


def test_sysconf_init():
    selections_1 = ScheduleSelection(schedule=np.array([1, 2, 3]), counter=42)
    selections_2 = ScheduleSelection(schedule=np.array([4, 2, 1]), counter=4)
    sysconf = SystemConfig(schedule_choices={'1': selections_1, '2': selections_2})
    assert sysconf.schedule_choices == {'1': selections_1, '2': selections_2}


def test_serialization():
    codec = JSON()
    for serializer in cohda_serializers:
        codec.add_serializer(*serializer())

    selections_1 = ScheduleSelection(schedule=np.array([1, 2, 3]), counter=42)
    selections_2 = ScheduleSelection(schedule=np.array([4, 2, 1]), counter=4)
    encoded_selection = (codec.encode(selections_1), codec.encode(selections_2))
    assert codec.decode(encoded_selection[0]) == selections_1
    assert codec.decode(encoded_selection[1]) == selections_2

    sysconf = SystemConfig(schedule_choices={'1': selections_1, '2': selections_2})
    encoded = codec.encode(sysconf)
    decoded = codec.decode(encoded)
    assert sysconf == decoded

    candidate = SolutionCandidate(agent_id='1', perf=float('-inf'),
                                  schedules={'1': np.array([5, 6, 7]),
                                             '2': np.array([8, 9, 10])})
    encoded = codec.encode(candidate)
    decoded = codec.decode(encoded)
    assert candidate == decoded

    working_memory = WorkingMemory(target_params=[[1, 2, 3], [1, 1, 1]],
                                   system_config=sysconf,
                                   solution_candidate=candidate)
    encoded = codec.encode(working_memory)
    decoded = codec.decode(encoded)
    assert working_memory == decoded

    msg = CohdaMessage(working_memory=working_memory)
    encoded = codec.encode(msg)
    decoded = codec.decode(encoded)
    assert msg.working_memory == decoded.working_memory

    negotiation_msg = NegotiationMessage(coalition_id=uuid.uuid1(), negotiation_id=uuid.uuid4(), message=msg)
    negotiation_msg.message_weight = fractions.Fraction(2, 5)
    encoded = codec.encode(negotiation_msg)
    decoded = codec.decode(encoded)
    assert negotiation_msg.coalition_id == decoded.coalition_id
    assert negotiation_msg.negotiation_id == decoded.negotiation_id
    assert negotiation_msg.message.working_memory == decoded.message.working_memory
    assert negotiation_msg.message_weight == decoded.message_weight


def test_candidate_init():
    schedules = {'1': np.array([1, 2, 3]), '2': np.array([4, 5, 6])}
    candidate = SolutionCandidate(agent_id='1', schedules=schedules, perf=2)
    assert candidate.agent_id == '1'
    assert candidate.schedules == schedules
    assert candidate.perf == 2


@pytest.mark.parametrize(
    "schedules_i, schedules_j, expected",
    [
        ({'1': ([1, 2], 42), '2': ([4, 2], 4)}, {'1': ([10, 20], 45), '2': ([40, 20], 5)}, False),
        ({'1': ([1, 2], 42), '2': ([4, 2], 4)}, {'1': ([1, 2], 42), '2': ([4, 2], 4)}, True),
        ({'1': ([1, 2], 42)}, {'1': ([1, 2], 42), '2': ([4, 2], 4)}, False),
        ({'1': ([1, 2], 42), '2': ([4, 2], 4)}, {'1': ([1, 2], 42), '2': ([4, 2], 5)}, False),
        ({'1': ([1, 2], 42), '2': ([4, 2], 4)}, {'1': ([1, 2], 42), '3': ([4, 2], 4)}, False),
    ]
)
def test_sysconf_equal(schedules_i: Dict[str, Tuple[List, int]], schedules_j: Dict[str, Tuple[List, int]],
                       expected: bool):
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
    "schedules_i, agent_id_i, perf_i, schedules_j, agent_id_j, perf_j, expected",
    [
        ({'1': [1, 2], '2': [4, 2]}, '1', 0.5, {'1': [10, 20], '2': [40, 20]}, '1', 0.5, False),
        ({'1': [1, 2], '2': [4, 2]}, '1', 0.5, {'1': [1, 2], '2': [4, 2]}, '1', 0.5, True),
        ({'2': [4, 2]}, '1', 0.5, {'1': [1, 2], '2': [4, 2]}, '1', 0.5, False),
        ({'1': [1, 2], '2': [4, 2]}, '1', 0.5, {'1': [1, 2]}, '1', 0.5, False),
        ({'1': [1, 3], '2': [4, 2]}, '1', 0.5, {'1': [1, 2], '2':[4, 2]}, '1', 0.5, False),
        ({'1': [1, 2], '2': [4, 2]}, '2', 0.5, {'1': [1, 2], '2': [4, 2]}, '1', 0.5, False),
        ({'1': [1, 2], '2': [4, 2]}, '1', 0.4, {'1': [1, 2], '2': [4, 2]}, '1', 0.5, False),
    ]
)
def test_candidate_equal(schedules_i: Dict[str, List], agent_id_i: str, perf_i: float,
                         schedules_j: Dict[str, List], agent_id_j: str, perf_j: float,
                         expected: bool):
    candidate_i = SolutionCandidate(schedules={k: np.array(v) for k, v in schedules_i.items()}, agent_id=agent_id_i,
                                    perf=perf_i)
    candidate_j = SolutionCandidate(schedules={k: np.array(v) for k, v in schedules_j.items()}, agent_id=agent_id_j,
                                    perf=perf_j)
    assert (candidate_i == candidate_j) == expected


@pytest.mark.parametrize(
    "schedules_i, schedules_j, expected_schedules",
    [
        ({'1': ([1, 2], 42), '2': ([4, 2], 4)}, {'1': ([10, 20], 45), '2': ([40, 20], 5)},
         {'1': ([10, 20], 45), '2': ([40, 20], 5)}),
        ({'1': ([1, 2], 42)}, {'1': ([10, 20], 40), '2': ([40, 20], 5)}, {'1': ([1, 2], 42), '2': ([40, 20], 5)}),
        ({'1': ([1, 2], 42)}, {'2': ([40, 20], 5)}, {'1': ([1, 2], 42), '2': ([40, 20], 5)}),
        ({'1': ([1, 2], 42)}, {'1': ([40, 20], 5)}, {'1': ([1, 2], 42)}),
        ({'1': ([1, 2], 42), '2': ([40, 20], 5)}, {'1': ([1, 2], 42), '2': ([40, 20], 5)},
         {'1': ([1, 2], 42), '2': ([40, 20], 5)}),
    ]
)
def test_sysconf_merge(schedules_i: Dict[str, Tuple[List, int]], schedules_j: Dict[str, Tuple[List, int]],
                       expected_schedules):
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

    merged_sysconfig = COHDA._merge_sysconfigs(sysconfig_i, sysconfig_j)
    assert merged_sysconfig == expected_sysconfig
    assert (sysconfig_i == merged_sysconfig) == (sysconfig_i is merged_sysconfig)


@pytest.mark.parametrize(
    "schedules_i, agent_id_i, perf_i, schedules_j, agent_id_j, perf_j, own_agent_id, expected_schedules, "
    "expected_agent_id, expected_perf",
    [
        ({'1': [1, 2], '2': [4, 2]}, '1', 0.5, {'1': [10, 20], '2': [40, 20]}, '2', 0.5, '3',
         {'1': [1, 2], '2': [4, 2]}, '1', 0.5),
        ({'1': [1, 2], '2': [4, 2]}, '1', 0.4, {'1': [10, 20], '2': [40, 20]}, '2', 0.5, '3',
         {'1': [10, 20], '2': [40, 20]}, '2', 0.5),
        ({'1': [1, 2], '2': [4, 2]}, '1', 0.4, {'2': [40, 20]}, '2', 0.5, '3',
         {'1': [1, 2], '2': [4, 2]}, '1', 0.4),
        ({'1': [1, 2]}, '1', 0.4, {'1': [10, 20], '2': [40, 20]}, '2', 0.5, '3',
         {'1': [10, 20], '2': [40, 20]}, '2', 0.5),
        ({'1': [1, 2]}, '1', 0.4, {'2': [40, 20]}, '2', 0.5, '3', {'1': [1, 2], '2': [40, 20]}, '3', 63),
    ]
)
def test_candidate_merge(schedules_i: Dict[str, List], agent_id_i: str, perf_i: float,
                         schedules_j: Dict[str, List], agent_id_j: str, perf_j: float,
                         own_agent_id: str,
                         expected_schedules: Dict[int, List], expected_agent_id: str, expected_perf: float):
    candidate_i = SolutionCandidate(schedules={k: np.array(v) for k, v in schedules_i.items()}, agent_id=agent_id_i,
                                    perf=perf_i)
    candidate_j = SolutionCandidate(schedules={k: np.array(v) for k, v in schedules_j.items()}, agent_id=agent_id_j,
                                    perf=perf_j)
    expected_candidate = SolutionCandidate(schedules={k: np.array(v) for k, v in expected_schedules.items()},
                                           agent_id=expected_agent_id, perf=expected_perf)

    def sum_schedule(cluster_schedule, _):
        return cluster_schedule.sum()

    assert COHDA._merge_candidates(candidate_i=candidate_i, candidate_j=candidate_j, agent_id=own_agent_id,
                                   perf_func=sum_schedule, target_params=None) == expected_candidate


def test_sysconf_cluster_schedule():
    selections_1 = ScheduleSelection(schedule=np.array([1, 2, 3]), counter=42)
    selections_2 = ScheduleSelection(schedule=np.array([4, 2, 1]), counter=4)
    sysconf = SystemConfig({'2': selections_2, '1': selections_1})
    assert np.array_equal(sysconf.cluster_schedule, np.array([[4, 2, 1], [1, 2, 3]]))


def test_candidate_cluster_schedule():
    candidate = SolutionCandidate(agent_id='1', schedules={'1': np.array([1, 2]), '2': np.array([2, 3])}, perf=0)
    assert np.array_equal(candidate.cluster_schedule, np.array([[1, 2], [2, 3]]))
