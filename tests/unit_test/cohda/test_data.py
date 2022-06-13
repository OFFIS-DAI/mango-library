from mango_library.negotiation.cohda.cohda import *


def create_cohda_object(schedule_provider, part_id, memory_target_params, is_local_acceptable=lambda x: True, counter=0,
                        perf_func=None):

    cohda_object = COHDA(schedule_provider=schedule_provider,
                         is_local_acceptable=is_local_acceptable,
                         part_id=part_id, perf_func=perf_func)
    cohda_object._memory.target_params = memory_target_params
    cohda_object._counter = counter
    return cohda_object


# 'old_sysconfig, old_candidate, cohda_object, expected_sysconfig, expected_candidate',
test_decide_params = [
    (
        SystemConfig(schedule_choices={'1': ScheduleSelection([0, 0], 1)}),
        SolutionCandidate(agent_id='1', schedules={'1': [0, 0]}, perf=-2),
        create_cohda_object(schedule_provider=lambda: [[0, 0], [1, 1]],
                            part_id='1', counter=1,
                            memory_target_params=([1, 1], [1, 1]),),
        SystemConfig(schedule_choices={'1': ScheduleSelection([1, 1], 2)}),
        SolutionCandidate(agent_id='1', schedules={'1': [1, 1]}, perf=0),
    ),
    (
        SystemConfig(schedule_choices={'1': ScheduleSelection([0, 0], 1)}),
        SolutionCandidate(agent_id='1', schedules={'1': [0, 0]}, perf=-2),
        create_cohda_object(schedule_provider=lambda: [[0, 0], [1, 1]],
                            is_local_acceptable=lambda x: True if x[0] <= 0 else False, part_id='1', counter=1,
                            memory_target_params=([1, 1], [1, 1]), ),
        SystemConfig(schedule_choices={'1': ScheduleSelection([0, 0], 1)}),
        SolutionCandidate(agent_id='1', schedules={'1': [0, 0]}, perf=-2),
    ),
    (
        SystemConfig(schedule_choices={'1': ScheduleSelection([0, 0], 1)}),
        SolutionCandidate(agent_id='1', schedules={'1': [0, 0]}, perf=-2),
        create_cohda_object(schedule_provider=lambda: [[0, 0], [0, 1], [1, 1]],
                            is_local_acceptable=lambda x: True if x[0] <= 0 else False, part_id='1', counter=1,
                            memory_target_params=([1, 1], [1, 1]), ),
        SystemConfig(schedule_choices={'1': ScheduleSelection([0, 1], 2)}),
        SolutionCandidate(agent_id='1', schedules={'1': [0, 1]}, perf=-1),
    ),
    (
        SystemConfig(schedule_choices={'1': ScheduleSelection([0, 0], 1), '2': ScheduleSelection([3, 4], 2)}),
        SolutionCandidate(agent_id='1', schedules={'1': [0, 0], '2': [1, 1]}, perf=0),
        create_cohda_object(schedule_provider=lambda: [[0, 0], [1, 1]],
                            part_id='1', counter=1,
                            memory_target_params=([1, 1], [1, 1]), ),
        SystemConfig(schedule_choices={'1': ScheduleSelection([0, 0], 1), '2': ScheduleSelection([3, 4], 2)}),
        SolutionCandidate(agent_id='1', schedules={'1': [0, 0], '2': [1, 1]}, perf=0),
    ),
    (
        SystemConfig(schedule_choices={'1': ScheduleSelection([0, 0], 1), '2': ScheduleSelection([2, 0], 2)}),
        SolutionCandidate(agent_id='1', schedules={'1': [0, 0], '2': [0, -1]}, perf=-3),
        create_cohda_object(schedule_provider=lambda: [[0, 0], [0, 1], [1, 1]],
                            part_id='1', counter=1,
                            memory_target_params=([1, 1], [1, 1]), ),
        SystemConfig(schedule_choices={'1': ScheduleSelection([0, 1], 2), '2': ScheduleSelection([2, 0], 2)}),
        SolutionCandidate(agent_id='1', schedules={'1': [0, 1], '2': [2, 0]}, perf=-1),
    ),
]

test_perceive_params = [
                             (SystemConfig({}), SolutionCandidate(agent_id='1', perf=None, schedules={}),
                              [CohdaMessage(working_memory=WorkingMemory(
                                  target_params=([1, 1, 1], [1, 1, 1]),
                                  system_config=SystemConfig({'2': ScheduleSelection([0, 0, 2], 1)}),
                                  solution_candidate=SolutionCandidate(schedules={'2': [0, 0, 2]},
                                                                       agent_id='2', perf=-3)
                              ))],
                              SystemConfig({'2': ScheduleSelection([0, 0, 2], 1),
                                            '1': ScheduleSelection([0, 1, 2], 1)}),
                              SolutionCandidate(schedules={'2': [0, 0, 2], '1': [0, 1, 2]}, agent_id='1', perf=-4)
                              ),

                             (SystemConfig({}), SolutionCandidate(agent_id='1', perf=None, schedules={}),
                              [CohdaMessage(working_memory=WorkingMemory(
                                  target_params=([1, 1, 1], [1, 1, 1]),
                                  system_config=SystemConfig({'2': ScheduleSelection([0, 0, 2], 1)}),
                                  solution_candidate=SolutionCandidate(schedules={'2': [0, 0, 2]},
                                                                       agent_id='2', perf=-3)
                              )),
                                  CohdaMessage(working_memory=WorkingMemory(
                                      target_params=([1, 1, 1], [1, 1, 1]),
                                      system_config=SystemConfig({'3': ScheduleSelection([1, 0, 2], 1)}),
                                      solution_candidate=SolutionCandidate(schedules={'3': [1, 0, 2]}, agent_id='3',
                                                                           perf=-2)
                                  ))],
                              SystemConfig({'3': ScheduleSelection([1, 0, 2], 1),
                                            '2': ScheduleSelection([0, 0, 2], 1),
                                            '1': ScheduleSelection([0, 1, 2], 1)}),
                              SolutionCandidate(schedules={'2': [0, 0, 2], '1': [0, 1, 2], '3': [1, 0, 2]},
                                                agent_id='1', perf=-5)
                              ),

                             (SystemConfig({}), SolutionCandidate(agent_id='1', perf=None, schedules={}),
                              [CohdaMessage(working_memory=WorkingMemory(
                                  target_params=([1, 1, 1], [1, 1, 1]),
                                  system_config=SystemConfig({'2': ScheduleSelection([0, 0, 2], 1)}),
                                  solution_candidate=SolutionCandidate(schedules={'2': [0, 0, 2]},
                                                                       agent_id='2', perf=-3)
                              )),
                                  CohdaMessage(working_memory=WorkingMemory(
                                      target_params=([1, 1, 1], [1, 1, 1]),
                                      system_config=SystemConfig({'2': ScheduleSelection([1, 1, 1], 2),
                                                                  '3': ScheduleSelection([1, 0, 2], 1)}),
                                      solution_candidate=SolutionCandidate(schedules={'3': [1, 0, 2], '2': [1, 1, 1]},
                                                                           agent_id='3', perf=-3)
                                  ))],
                              SystemConfig({'3': ScheduleSelection([1, 0, 2], 1),
                                            '2': ScheduleSelection([1, 1, 1], 2),
                                            '1': ScheduleSelection([0, 1, 2], 1)}),
                              SolutionCandidate(schedules={'2': [0, 0, 2], '1': [0, 1, 2], '3': [1, 0, 2]},
                                                agent_id='1', perf=-5)
                              ),

                             (SystemConfig(schedule_choices={'1': ScheduleSelection([2, 2, 2], 1)}),
                              SolutionCandidate(agent_id='1', perf=-3, schedules={'1': [2, 2, 2]}),
                              [CohdaMessage(working_memory=WorkingMemory(
                                  target_params=([1, 1, 1], [1, 1, 1]),
                                  system_config=SystemConfig({'2': ScheduleSelection([0, 0, 2], 1)}),
                                  solution_candidate=SolutionCandidate(schedules={'2': [0, 0, 2]},
                                                                       agent_id='2', perf=-3)
                              )),
                                  CohdaMessage(working_memory=WorkingMemory(
                                      target_params=([1, 1, 1], [1, 1, 1]),
                                      system_config=SystemConfig({'2': ScheduleSelection([1, 1, 1], 2),
                                                                  '3': ScheduleSelection([1, 0, 2], 1)}),
                                      solution_candidate=SolutionCandidate(schedules={'3': [1, 0, 2], '2': [1, 1, 1]},
                                                                           agent_id='3', perf=-3)
                                  ))],
                              SystemConfig({'3': ScheduleSelection([1, 0, 2], 1),
                                            '2': ScheduleSelection([1, 1, 1], 2),
                                            '1': ScheduleSelection([2, 2, 2], 1)}),
                              SolutionCandidate(schedules={'2': [0, 0, 2], '1': [2, 2, 2], '3': [1, 0, 2]},
                                                agent_id='1', perf=-8)
                              ),
]
