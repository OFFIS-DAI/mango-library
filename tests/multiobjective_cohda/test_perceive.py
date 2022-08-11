import numpy as np

from mango_library.negotiation.multiobjective_cohda.data_classes import SolutionCandidate, ScheduleSelections, \
    SystemConfig, WorkingMemory
from mango_library.negotiation.multiobjective_cohda.multiobjective_cohda import CohdaMessage, COHDA


def target_func(cluster_schedules, target_params=None):
    performances = []
    for cs in cluster_schedules:
        mean = cs.mean(axis=0)
        performances.append(tuple(mean))
    return performances


def test_perceive():
    possible_schedules = [[0, 1], [0.2, 0.7], [0.3, 0.3], [0.6, 0.5]]

    cohda = COHDA(schedule_provider=lambda: possible_schedules,
                  is_local_acceptable=lambda s: True,
                  perf_func=target_func,
                  part_id='1',
                  num_iterations=1,
                  reference_point=(1, 1)
                  )

    candidate_1 = SolutionCandidate(agent_id='1', schedules={
        '2': np.array([[0.5, 0.5], [0.3, 0.7]])
    }, num_solution_points=2)

    candidate_1.perf = target_func(candidate_1.cluster_schedules)
    candidate_1.hypervolume = cohda.get_hypervolume(performances=candidate_1.perf)

    systemconfig_1 = SystemConfig(schedule_choices={
        '2': ScheduleSelections(counter=2,
                                schedules=np.array([[0.1, 0.1], [0.2, 0.2]])),
    }, num_solution_points=2
    )

    new_wm = WorkingMemory(target_params=None, system_config=systemconfig_1, solution_candidate=candidate_1)

    msg = CohdaMessage(working_memory=new_wm
                       )

    new_sysconf, new_candidate = cohda._perceive(messages=[msg])
    assert cohda._memory.system_config != new_sysconf
    assert cohda._memory.solution_candidate != new_candidate
    assert '1' in new_sysconf.schedule_choices.keys() and '2' in new_sysconf.schedule_choices.keys()
    # assert new_wm.system_config == systemconfig_1
    # assert new_wm.solution_candidate == candidate_1

    cohda._memory = WorkingMemory(system_config=new_sysconf, solution_candidate=new_candidate, target_params=None)
    new_sysconf_2, new_candidate_2 = cohda._perceive(messages=[msg])

    assert new_sysconf_2 == cohda._memory.system_config
    assert new_candidate_2 == cohda._memory.solution_candidate
    assert new_sysconf_2 is cohda._memory.system_config
    assert new_candidate_2 is cohda._memory.solution_candidate
