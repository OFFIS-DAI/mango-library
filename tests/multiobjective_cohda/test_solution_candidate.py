from typing import List, Tuple
from mango_library.negotiation.multiobjective_cohda.data_classes import SolutionCandidate, SolutionPoint
from mango_library.negotiation.multiobjective_cohda.multiobjective_cohda import COHDA
import numpy as np


def get_hypervolume(performances: List[Tuple[float, ...]]) -> float:
    """
    This function calculates the hypervolume from a list of performance tuples with a given reference point
    :param performances:
    :return:
    """
    #  ----- START DUMMY ALLERT -----

    total_performance = 0
    for candidate_performance in performances:
        for objective_performance in candidate_performance:
            total_performance += objective_performance

    #  ----- END DUMMY ALLERT -----

    return total_performance


def test_init():
    schedules_1 = np.array([[1, 2, 3], [2, 3, 4]], np.int32)
    schedules_2 = np.array([[4, 2, 1], [4, 5, 4]], np.int32)
    candidate = SolutionCandidate(agent_id='1', schedules={'1': schedules_1, '2': schedules_2}, num_solution_points=2,
                                  perf=None, hypervolume=None)
    assert candidate.schedules == {'1': schedules_1,
                                   '2': schedules_2} and candidate.agent_id == '1'


def perf_fkt(cluster_schedules: List[np.array], target_params=None) -> List[Tuple[float, ...]]:
    # test with dummy perf fkt
    perf_list = []
    for cs in cluster_schedules:
        mean_perf = np.mean(cs)
        max_perf = np.max(cs)
        perf_list.append((float(mean_perf), float(max_perf)))
    return perf_list


def perf_fkt_min(solution_points: List[SolutionPoint], target_params=None) -> List[Tuple[float, ...]]:
    perf_list = []
    for solution_point in solution_points:
        print('cluster_schedule', solution_point.cluster_schedule)
        minimas = np.min(solution_point.cluster_schedule, axis=0)
        current_perf = []
        for minimum in minimas:
            current_perf.append(float(minimum))
        perf_list.append(tuple(current_perf))
    return perf_list


def test_merge_dummy():
    possible_schedules = []
    ref_point = (1.1, 1.1, 1.1)

    cohda = COHDA(
        schedule_provider=lambda: possible_schedules,
        is_local_acceptable=lambda s: True,
        perf_func=perf_fkt,
        reference_point=ref_point,
        part_id='1',
        num_iterations=1,
    )
    # use dummy function for hypervolume calculation for testing
    cohda.get_hypervolume = get_hypervolume

    schedules_1_1 = np.array([[1, 2, 3], [2, 3, 4]], np.int32)
    schedules_2_1 = np.array([[1, 1, 1], [1, 1, 1]], np.int32)

    schedules_1_2 = np.array([[2, 3, 4], [2, 4, 5]], np.int32)
    schedules_2_2 = np.array([[2, 2, 2], [2, 2, 2]], np.int32)

    schedules_1_3 = np.array([[1, 2, 3], [2, 3, 4]], np.int32)
    schedules_2_3 = np.array([[4, 2, 1], [4, 5, 4]], np.int32)
    schedule_3_3 = np.array([[4, 2, 1], [4, 5, 4]], np.int32)

    candidate_1 = SolutionCandidate(agent_id='1',
                                    schedules={'1': schedules_1_1, '2': schedules_2_1},
                                    num_solution_points=2)
    candidate_1.perf = perf_fkt(candidate_1.cluster_schedules)
    candidate_1.hypervolume = get_hypervolume(candidate_1.perf)
    candidate_2 = SolutionCandidate(agent_id='1', schedules={'1': schedules_1_2,
                                                             '2': schedules_2_2}, num_solution_points=2)
    candidate_2.perf = perf_fkt(candidate_2.cluster_schedules)
    candidate_2.hypervolume = get_hypervolume(candidate_2.perf)
    candidate_3 = SolutionCandidate(agent_id='2', schedules={'2': schedules_2_3,
                                                             '3': schedule_3_3}, num_solution_points=2)
    candidate_3.perf = perf_fkt(candidate_3.cluster_schedules)
    candidate_3.hypervolume = get_hypervolume(candidate_3.perf)

    print(candidate_1.perf)
    merge_result = cohda._merge_candidates(
        candidate_i=candidate_1, candidate_j=candidate_2, agent_id='1',
        perf_func=perf_fkt)

    assert merge_result != candidate_1 and merge_result is not candidate_1

    merge_result = cohda._merge_candidates(
        candidate_i=candidate_1, candidate_j=candidate_1, agent_id='1',
        perf_func=perf_fkt)
    assert merge_result == candidate_1 and merge_result is candidate_1

    merge_result = cohda._merge_candidates(
        candidate_i=candidate_3, candidate_j=candidate_1, agent_id='3',
        perf_func=perf_fkt)

    assert merge_result != candidate_3 and merge_result is not candidate_3
    assert merge_result.agent_id == '3'
    candidate = merge_result.schedules
    assert set(candidate.keys()) == {'1', '2', '3'}
    assert np.array_equal(candidate['1'], schedules_1_1)
    assert np.array_equal(candidate['2'], schedules_2_3)
    assert np.array_equal(candidate['3'], schedule_3_3)


def test_merge_sms_emoa():
    ref_point = (1.1, 1.1, 1.1)
    possible_schedules = [[0.1, 0.9], [0.3, 0.6]]

    cohda = COHDA(
        schedule_provider=lambda: possible_schedules,
        is_local_acceptable=lambda s: True,
        perf_func=perf_fkt,
        reference_point=ref_point,
        part_id='1',
        num_iterations=1,
    )

    schedules_1_1 = np.array([[1, 2, 3], [2, 3, 4]], np.int32)
    schedules_2_1 = np.array([[1, 1, 1], [1, 1, 1]], np.int32)

    schedules_1_2 = np.array([[2, 3, 4], [2, 4, 5]], np.int32)
    schedules_2_2 = np.array([[2, 2, 2], [2, 2, 2]], np.int32)

    schedules_1_3 = np.array([[1, 2, 3], [2, 3, 4]], np.int32)
    schedules_2_3 = np.array([[4, 2, 1], [4, 5, 4]], np.int32)
    schedule_3_3 = np.array([[4, 2, 1], [4, 5, 4]], np.int32)

    candidate_1 = SolutionCandidate(agent_id='1', schedules={'1': schedules_1_1, '2': schedules_2_1},
                                    num_solution_points=2)
    candidate_1.perf = perf_fkt_min(candidate_1.solution_points)
    candidate_1.hypervolume = cohda.get_hypervolume(candidate_1.perf)

    candidate_2 = SolutionCandidate(agent_id='1', schedules={'1': schedules_1_2,
                                                             '2': schedules_2_2}, num_solution_points=2)
    candidate_2.perf = perf_fkt_min(candidate_2.solution_points)
    candidate_2.hypervolume = cohda.get_hypervolume(candidate_2.perf)

    candidate_3 = SolutionCandidate(agent_id='2', schedules={'2': schedules_2_3,
                                                             '3': schedule_3_3}, num_solution_points=2)
    candidate_3.perf = perf_fkt_min(candidate_3.solution_points)
    candidate_3.hypervolume = cohda.get_hypervolume(candidate_3.perf)

    merge_result = cohda._merge_candidates(
        candidate_i=candidate_1, candidate_j=candidate_2, agent_id='1',
        perf_func=perf_fkt_min)

    assert merge_result is candidate_1

    merge_result = cohda._merge_candidates(
        candidate_i=candidate_1, candidate_j=candidate_1, agent_id='1',
        perf_func=perf_fkt_min)
    assert merge_result is candidate_1

    merge_result = cohda._merge_candidates(
        candidate_i=candidate_3, candidate_j=candidate_1, agent_id='3',
        perf_func=perf_fkt)

    assert merge_result is not candidate_3 and merge_result != candidate_3
    assert merge_result.agent_id == '3'
    schedules = merge_result.schedules
    assert set(schedules.keys()) == {'1', '2', '3'}
    assert np.array_equal(schedules['1'], schedules_1_1)
    assert np.array_equal(schedules['2'], schedules_2_3)
    assert np.array_equal(schedules['3'], schedule_3_3)


def test_cluster_schedule():
    schedules_1 = np.array([[1, 2, 3], [2, 3, 4]])
    schedules_2 = np.array([[4, 2, 1], [4, 5, 4]])
    candidate = SolutionCandidate(schedules={'2': schedules_2, '1': schedules_1},
                                  agent_id='1', num_solution_points=2)
    assert candidate.solution_points[0].idx == {'1': 0, '2': 1}
    assert np.array_equal(candidate.solution_points[0].cluster_schedule, np.array([[1, 2, 3], [4, 2, 1]]))
    assert np.array_equal(candidate.solution_points[1].cluster_schedule, np.array([[2, 3, 4], [4, 5, 4]]))
