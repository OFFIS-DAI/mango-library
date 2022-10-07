import asyncio

import numpy as np
import pytest
from mango.core.container import Container
from mango.messages.codecs import JSON

from mango_library.negotiation.multiobjective_cohda.data_classes import Target
from mango_library.negotiation.multiobjective_cohda.multiobjective_cohda import COHDA
from mango_library.negotiation.util import multi_objective_serializers
from util import MINIMIZE_TARGETS, MAXIMIZE_TARGETS, \
    create_agents, get_solution

NUM_ITERATIONS = 1
NUM_AGENTS = 3
NUM_CANDIDATES = 1
CHECK_MSG_QUEUE_INTERVAL = 1
SCHEDULES_FOR_AGENTS_SIMPEL = [
    [
        [0.1, 0.7], [0.1, 0.1],
    ],
    [
        [0.1, 0.9], [0.2, 0.2],
    ],
    [
        [0.2, 0.7], [0.4, 0.4],
    ],
]

SCHEDULES_FOR_AGENTS_COMPLEX = [[1, 1], [0.6, 0.6]]
for i in range(11):
    SCHEDULES_FOR_AGENTS_COMPLEX.append([i * 0.1, 1 - i * 0.1])
SCHEDULES_FOR_AGENTS_COMPLEX = [SCHEDULES_FOR_AGENTS_COMPLEX]


@pytest.mark.asyncio
async def test_minimize_scenario():
    """
    Method to test a scenario of multi-objective COHDA with two objectives to
    minimize: to minimize the deviation within a schedule and to minimize
    the sum of the schedule. In this example, only one candidate is taken
    and each agent only stores two possible schedules. The schedules are
    very conflicting and one is good for both objectives, the other one not.
    Since there is only one candidate given, the agent mutates its chosen
    schedule in the given candidate and because of the mock mutation function,
    both schedules are tried at least once. By that it is made sure that the
    solution of the negotiation is actually the best possible because all
    options were considered.
    """
    c = await Container.factory(addr=('127.0.0.2', 5555))

    agents, addrs, controller_agent = await create_agents(container=c,
                                                          targets=MINIMIZE_TARGETS,
                                                          possible_schedules=SCHEDULES_FOR_AGENTS_SIMPEL,
                                                          num_iterations=NUM_ITERATIONS,
                                                          num_candidates=NUM_CANDIDATES,
                                                          check_msg_queue_interval=CHECK_MSG_QUEUE_INTERVAL,
                                                          num_agents=NUM_AGENTS)

    await asyncio.wait_for(wait_for_term(controller_agent), timeout=15)
    solution_dict = get_solution(agents).schedules
    print('solution:', solution_dict, '\n')
    for aid, chosen_schedules in solution_dict.items():
        # for minimizing, every second schedule is the better because
        # sum and deviations are minimized
        chosen_schedule = chosen_schedules[0]
        print(f'[{aid}] chosen schedule: {chosen_schedule}.')
        idx = int(aid[-1]) - 1
        assert np.array_equal(chosen_schedule, SCHEDULES_FOR_AGENTS_SIMPEL[idx][1])

    await c.shutdown()


@pytest.mark.asyncio
async def test_maximize_scenario():
    """
    This method follows the same principle as the other test, but the
    goal is to maximize the objectives.
    """
    c = await Container.factory(addr=('127.0.0.2', 5555))

    agents, addrs, controller_agent = await create_agents(container=c,
                                                          targets=MAXIMIZE_TARGETS,
                                                          possible_schedules=SCHEDULES_FOR_AGENTS_SIMPEL,
                                                          num_iterations=NUM_ITERATIONS,
                                                          num_candidates=NUM_CANDIDATES,
                                                          check_msg_queue_interval=CHECK_MSG_QUEUE_INTERVAL,
                                                          num_agents=NUM_AGENTS)

    await asyncio.wait_for(wait_for_term(controller_agent), timeout=15)

    solution_dict = get_solution(agents).schedules
    print('solution:', solution_dict, '\n')
    for aid, chosen_schedules in solution_dict.items():
        # for minimizing, every second schedule is the better because
        # sum and deviations are minimized
        chosen_schedule = chosen_schedules[0]
        print(f'[{aid}] chosen schedule: {chosen_schedule}.')
        idx = int(aid[-1]) - 1
        assert np.array_equal(chosen_schedule, SCHEDULES_FOR_AGENTS_SIMPEL[idx][0])

    # gracefully shutdown
    await c.shutdown()


@pytest.mark.asyncio
async def test_maximize_scenario_without_fixed_reference_point():
    """
    This method follows the same principle as the other test, but the
    goal is to maximize the objectives.
    """
    c = await Container.factory(addr=('127.0.0.2', 5555))

    agents, addrs, controller_agent = await create_agents(container=c,
                                                          targets=MAXIMIZE_TARGETS,
                                                          possible_schedules=SCHEDULES_FOR_AGENTS_SIMPEL,
                                                          num_iterations=NUM_ITERATIONS,
                                                          num_candidates=NUM_CANDIDATES,
                                                          check_msg_queue_interval=CHECK_MSG_QUEUE_INTERVAL,
                                                          num_agents=NUM_AGENTS,
                                                          use_fixed_ref_point=False,
                                                          offsets=None)

    await asyncio.wait_for(wait_for_term(controller_agent), timeout=15)

    solution_dict = get_solution(agents).schedules
    print('solution:', solution_dict, '\n')
    for aid, chosen_schedules in solution_dict.items():
        # for minimizing, every second schedule is the better because
        # sum and deviations are minimized
        chosen_schedule = chosen_schedules[0]
        print(f'[{aid}] chosen schedule: {chosen_schedule}.')
        idx = int(aid[-1]) - 1
        assert np.array_equal(chosen_schedule, SCHEDULES_FOR_AGENTS_SIMPEL[idx][0])

    for agent in agents:
        # use_fixed_ref_point was set to False, therefore, no reference point was given. After the negotiation is done,
        # there has to be a reference point, since there were calculations of it during the negotiation. This
        # reference point has to be different than the default ones given in the targets to make sure there has been
        # a calculation
        assert list(agent.roles[0]._cohda.values())[
                   0]._selection.sorting_component.hypervolume_indicator.reference_point is not None
        assert list(agent.roles[0]._cohda.values())[
            0]._selection.sorting_component.hypervolume_indicator.reference_point != MAXIMIZE_TARGETS[0].ref_point
        assert list(agent.roles[0]._cohda.values())[
            0]._selection.sorting_component.hypervolume_indicator.reference_point != MAXIMIZE_TARGETS[1].ref_point

    # gracefully shutdown
    await c.shutdown()


@pytest.mark.asyncio
async def test_maximize_scenario_without_fixed_reference_point_and_with_offsets():
    """
    This method follows the same principle as the other test, but the
    goal is to maximize the objectives.
    """
    c = await Container.factory(addr=('127.0.0.2', 5555))
    offsets = [2.0, 2.0]

    agents, addrs, controller_agent = await create_agents(container=c,
                                                          targets=MAXIMIZE_TARGETS,
                                                          possible_schedules=SCHEDULES_FOR_AGENTS_SIMPEL,
                                                          num_iterations=NUM_ITERATIONS,
                                                          num_candidates=NUM_CANDIDATES,
                                                          check_msg_queue_interval=CHECK_MSG_QUEUE_INTERVAL,
                                                          num_agents=NUM_AGENTS,
                                                          use_fixed_ref_point=False,
                                                          offsets=offsets)

    await asyncio.wait_for(wait_for_term(controller_agent), timeout=15)

    solution_dict = get_solution(agents).schedules
    print('solution:', solution_dict, '\n')
    for aid, chosen_schedules in solution_dict.items():
        # for minimizing, every second schedule is the better because
        # sum and deviations are minimized
        chosen_schedule = chosen_schedules[0]
        print(f'[{aid}] chosen schedule: {chosen_schedule}.')
        idx = int(aid[-1]) - 1
        assert np.array_equal(chosen_schedule, SCHEDULES_FOR_AGENTS_SIMPEL[idx][0])

    for agent in agents:
        # use_fixed_ref_point was set to False, therefore, no reference point was given. After the negotiation is done,
        # there has to be a reference point, since there were calculations of it during the negotiation. This
        # reference point has to be different than the default ones given in the targets to make sure there has been
        # a calculation
        assert list(agent.roles[0]._cohda.values())[
                   0]._selection.sorting_component.hypervolume_indicator.reference_point is not None
        assert list(agent.roles[0]._cohda.values())[
            0]._selection.sorting_component.hypervolume_indicator.reference_point != MAXIMIZE_TARGETS[0].ref_point
        assert list(agent.roles[0]._cohda.values())[
            0]._selection.sorting_component.hypervolume_indicator.reference_point != MAXIMIZE_TARGETS[1].ref_point

    # gracefully shutdown
    await c.shutdown()


@pytest.mark.asyncio
async def _test_maximize_different_container():
    """
    This method follows the same principle as the other test, but the
    goal is to maximize the objectives.
    """
    codec = JSON()
    codec2 = JSON()
    codec3 = JSON()
    for serializer in multi_objective_serializers:
        codec.add_serializer(*serializer())
        codec2.add_serializer(*serializer())
        codec3.add_serializer(*serializer())

    c_1 = await Container.factory(addr=('127.0.0.2', 5555), codec=codec)
    c_2 = await Container.factory(addr=('127.0.0.2', 5556), codec=codec2)
    c_3 = await Container.factory(addr=('127.0.0.2', 5557), codec=codec3)

    agents, addrs, controller_agent = await create_agents(container=[c_1, c_2, c_3],
                                                          targets=MAXIMIZE_TARGETS,
                                                          possible_schedules=SCHEDULES_FOR_AGENTS_SIMPEL,
                                                          num_iterations=NUM_ITERATIONS,
                                                          num_candidates=NUM_CANDIDATES,
                                                          check_msg_queue_interval=CHECK_MSG_QUEUE_INTERVAL,
                                                          num_agents=NUM_AGENTS)

    await asyncio.wait_for(wait_for_term(controller_agent), timeout=30)

    solution_dict = get_solution(agents).schedules
    print('solution:', solution_dict, '\n')
    for aid, chosen_schedules in solution_dict.items():
        # for minimizing, every second schedule is the better because
        # sum and deviations are minimized
        chosen_schedule = chosen_schedules[0]
        print(f'[{aid}] chosen schedule: {chosen_schedule}.')
        idx = int(aid[-1]) - 1
        assert np.array_equal(chosen_schedule, SCHEDULES_FOR_AGENTS_SIMPEL[idx][0])

    # gracefully shutdown
    await c_1.shutdown()
    await c_2.shutdown()


@pytest.mark.asyncio
async def test_complex_scenario():
    """
        Now we are going to test more complex scenarios
        """

    c_1 = await Container.factory(addr=('127.0.0.2', 5555))

    def minimize_first(cs):
        return float(np.mean(cs, axis=0)[0])

    def minimize_second(cs):
        return float(np.mean(cs, axis=0)[1])

    target_first = Target(target_function=minimize_first, ref_point=1.1, maximize=False)
    target_second = Target(target_function=minimize_second, ref_point=1.1, maximize=False)
    pick_fkt = COHDA.pick_all_points
    # pick_fkt = COHDA.pick_random_point
    mutate_fkt = COHDA.mutate_with_all_possible
    # mutate_fkt = COHDA.mutate_with_one_random

    agents, addrs, controller_agent = await create_agents(container=c_1,
                                                          targets=[target_first, target_second],
                                                          possible_schedules=SCHEDULES_FOR_AGENTS_COMPLEX,
                                                          num_iterations=NUM_ITERATIONS,
                                                          num_candidates=5,
                                                          check_msg_queue_interval=CHECK_MSG_QUEUE_INTERVAL,
                                                          num_agents=5,
                                                          pick_fkt=pick_fkt,
                                                          mutate_fkt=mutate_fkt,
                                                          )

    for a in agents:
        if a._check_inbox_task.done():
            if a._check_inbox_task.exception() is not None:
                raise a._check_inbox_task.exception()
            else:
                assert False, f'check_inbox terminated unexpectedly.'

    await asyncio.wait_for(wait_for_term(controller_agent), timeout=60)

    solution = get_solution(agents)
    print('cluster schedules:', solution.cluster_schedules)
    print('performances:', [(round(s[0], 2), round(s[1], 2)) for s in solution.perf])
    print('hypervolume:', round(solution.hypervolume, 4))
    for aid, chosen_schedules in solution.schedules.items():
        # for minimizing, every second schedule is the better because
        # sum and deviations are minimized
        for schedule in chosen_schedules:
            if np.sum(schedule) != 1:
                assert pick_fkt == COHDA.pick_random_point or mutate_fkt == COHDA.mutate_with_one_random

        # gracefully shutdown
        await c_1.shutdown()


async def wait_for_term(controller_agent):
    while len(controller_agent.roles[0]._weight_map.values()) != 1 or \
            list(controller_agent.roles[0]._weight_map.values())[0] != 1:
        await asyncio.sleep(0.1)
    print('Terminated!')
