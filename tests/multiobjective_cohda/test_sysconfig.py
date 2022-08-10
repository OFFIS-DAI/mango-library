from mango_library.negotiation.multiobjective_cohda.data_classes import SystemConfig, ScheduleSelections
from mango_library.negotiation.multiobjective_cohda.multiobjective_cohda import COHDA
import numpy as np


def test_init():
    selections_1 = ScheduleSelections(schedules=np.array([[1, 2, 3], [2, 3, 4]], np.int32), counter=42)
    selections_2 = ScheduleSelections(schedules=np.array([[4, 2, 1], [4, 5, 4]], np.int32), counter=4)
    sysconf = SystemConfig({'1': selections_1, '2': selections_2}, num_solution_points=2)
    assert sysconf.schedule_choices == {'1': selections_1, '2': selections_2}


def test_merge():
    old_1 = ScheduleSelections(schedules=np.array([[1, 2, 3], [2, 3, 4]], np.int32), counter=42)
    old_2 = ScheduleSelections(schedules=np.array([[4, 2, 1], [4, 5, 4]], np.int32), counter=4)
    new_1 = ScheduleSelections(schedules=np.array([[10, 20, 30], [23, 31, 41]], np.int32), counter=45)
    new_2 = ScheduleSelections(schedules=np.array([[32, 4, 2], [12, 3, 44]], np.int32), counter=5)
    new_3 = ScheduleSelections(schedules=np.array([[32, 7, 6], [6, 5, 44]], np.int32), counter=9)

    sysconf_i = SystemConfig({'2': new_2}, num_solution_points=2)
    sysconf_j = SystemConfig({'1': new_1, '2': old_2, '3': new_3}, num_solution_points=2)
    expected = SystemConfig({'3': new_3, '2': new_2, '1': new_1}, num_solution_points=2)
    merge_result = COHDA._merge_sysconfigs(sysconf_i, sysconf_j)
    assert expected == merge_result
    assert all(elem in merge_result.schedule_choices.keys() for elem in ['1', '2', '3'])

    second_merge = COHDA._merge_sysconfigs(merge_result, sysconf_j)
    assert second_merge == merge_result
    assert second_merge is merge_result

    third_merge = COHDA._merge_sysconfigs(expected, merge_result)

    assert third_merge == merge_result
    assert third_merge is not merge_result


def test_equal():
    selections_1_1 = ScheduleSelections(schedules=np.array([[1, 2, 3], [2, 3, 4]], np.int32), counter=42)
    selections_2_1 = ScheduleSelections(schedules=np.array([[4, 2, 1], [4, 5, 4]], np.int32), counter=4)
    selections_1_2 = ScheduleSelections(schedules=np.array([[1, 2, 3], [2, 3, 4]], np.int32), counter=42)
    selections_2_2 = ScheduleSelections(schedules=np.array([[4, 2, 1], [4, 5, 4]], np.int32), counter=4)
    selections_1_3 = ScheduleSelections(schedules=np.array([[0, 2, 3], [2, 3, 4]], np.int32), counter=42)
    selections_2_3 = ScheduleSelections(schedules=np.array([[0, 2, 1], [4, 5, 4]], np.int32), counter=4)
    selections_1_4 = ScheduleSelections(schedules=np.array([[1, 2, 3], [2, 3, 4]], np.int32), counter=99)
    selections_2_4 = ScheduleSelections(schedules=np.array([[4, 2, 1], [4, 5, 4]], np.int32), counter=99)

    sysconf_1 = SystemConfig({'1': selections_1_1, '2': selections_2_1}, num_solution_points=2)
    sysconf_2 = SystemConfig({'1': selections_1_2, '2': selections_2_2}, num_solution_points=2)
    sysconf_3 = SystemConfig({'1': selections_1_3, '2': selections_2_3}, num_solution_points=2)
    sysconf_4 = SystemConfig({'1': selections_1_4, '2': selections_2_4}, num_solution_points=2)

    assert sysconf_1 == sysconf_2
    assert sysconf_2 != sysconf_3
    assert sysconf_4 != sysconf_2


def test_cluster_schedule():
    selections_1 = ScheduleSelections(schedules=np.array([[1, 2, 3], [2, 3, 4]]), counter=42)
    selections_2 = ScheduleSelections(schedules=np.array([[4, 2, 1], [4, 5, 4]]), counter=4)
    sysconf = SystemConfig({'2': selections_2, '1': selections_1}, num_solution_points=2)
    assert np.array_equal(sysconf.cluster_schedules,
                          [np.array([[4, 2, 1], [1, 2, 3]]), np.array([[4, 5, 4], [2, 3, 4]])])
