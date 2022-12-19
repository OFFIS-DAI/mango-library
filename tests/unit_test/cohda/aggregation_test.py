import numpy as np
import pytest

from mango_library.negotiation.cohda.cohda_solution_aggregation import CohdaSolutionAggregationRole
from mango_library.negotiation.cohda.data_classes import SolutionCandidate


candidate_1 = SolutionCandidate(agent_id='1',
                                schedules={
                                    '1': np.array([1, 2, 3]),
                                    '2': np.array([2, 3, 4]),
                                },
                                perf=10)
candidate_2 = SolutionCandidate(agent_id='2',
                                schedules={
                                    '1': np.array([1, 2, 3]),
                                    '2': np.array([3, 4, 5]),
                                },
                                perf=11)

candidate_3 = SolutionCandidate(agent_id='2',
                                schedules={
                                    '2': np.array([3, 4, 5]),
                                },
                                perf=9)

candidate_4 = SolutionCandidate(agent_id='1',
                                schedules={
                                    '1': np.array([3, 4, 5]),
                                },
                                perf=8)

agg_candidate = SolutionCandidate(agent_id='Aggregation',
                                  schedules={
                                      '1': np.array([3, 4, 5]),
                                      '2': np.array([3, 4, 5]),
                                  },
                                  perf=float('-inf'))


@pytest.mark.parametrize('candidates, expected',
                         [
                             ([candidate_1, candidate_2], candidate_2),
                             ([candidate_2, candidate_2], candidate_2),
                             ([candidate_1, candidate_1], candidate_1),
                             ([candidate_1, candidate_2, candidate_3], candidate_2),
                             ([candidate_3, candidate_4, candidate_2], candidate_2),
                             ([candidate_3, candidate_4, candidate_1], candidate_1),
                             ([candidate_3, candidate_4], agg_candidate),
                         ])
def test_aggregate_solution(candidates, expected):
    solution = CohdaSolutionAggregationRole.aggregate_solution(candidates)
    assert solution == expected
