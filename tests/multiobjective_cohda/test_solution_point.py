from mango_library.negotiation.multiobjective_cohda.data_classes import SolutionPoint
import numpy as np


def test_set():
    point_one = SolutionPoint(cluster_schedule=np.array([[1, 2], [3, 4]]), performance=(1, 2), idx={'1': 2})
    point_two = SolutionPoint(cluster_schedule=np.array([[5, 6], [7, 8]]), performance=(2, 1), idx={'1': 2})
    example_points = [point_one, point_one, point_two]
    point_set = set(example_points)
    assert len(point_set) == len(example_points) - 1


def test_sort():
    point_one = SolutionPoint(cluster_schedule=np.array([[1, 2], [3, 4]]), performance=(1, 2), idx={'1': 2})
    point_two = SolutionPoint(cluster_schedule=np.array([[5, 6], [7, 8]]), performance=(2, 1), idx={'1': 2})
    assert sorted([point_one, point_two], reverse=True) == [point_two, point_one]
