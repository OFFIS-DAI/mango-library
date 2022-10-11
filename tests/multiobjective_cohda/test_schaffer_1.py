import asyncio
import numpy as np
import pytest
from mango_library.negotiation.multiobjective_cohda.data_classes import Target
from mango_library.negotiation.multiobjective_cohda.multiobjective_cohda import COHDA
from util import create_agents, get_solution
from mango.core.container import Container

A = 10
NUM_AGENTS = 30
NUM_SCHEDULES = 20
NUM_SOLUTION_POINTS = 5
TIMEOUT = 100

PICK_FKT = COHDA.pick_all_points
# PICK_FKT = COHDA.pick_random_point
MUTATE_FKT = COHDA.mutate_with_all_possible


# MUTATE_FKT = COHDA.mutate_with_one_random


@pytest.mark.asyncio
async def test_schaffer_1():
    targets = [
        Target(target_function=lambda cs: cs.sum() ** 2, ref_point=A ** 2 * 1.1),
        Target(target_function=lambda cs: (cs.sum() - 2) ** 2, ref_point=(A + 2) ** 2 * 1.1)
    ]
    schedule_threshold = A / NUM_AGENTS
    schedule_step_size = (schedule_threshold * 2) / (NUM_SCHEDULES - 1)
    possible_schedules = []
    for i in range(NUM_SCHEDULES):
        possible_schedules.append(np.array([-schedule_threshold + i * schedule_step_size]))

    c_1 = await Container.factory(addr=('127.0.0.2', 5555))

    agents, addrs, controller_agent = await create_agents(
        container=c_1, targets=targets, possible_schedules=possible_schedules, num_iterations=1,
        num_candidates=NUM_SOLUTION_POINTS, check_msg_queue_interval=0.1, num_agents=NUM_AGENTS, pick_fkt=PICK_FKT,
        mutate_fkt=MUTATE_FKT, schedules_all_equal=True)

    await asyncio.wait_for(wait_for_term(controller_agent), timeout=TIMEOUT)

    solution = get_solution(agents)
    rounded_perfs = []
    for sp in sorted(solution.solution_points):
        rounded_perfs.append([round(p, 2) for p in sp.performance])

    print('performances:', rounded_perfs)
    print('hypervolume', round(solution.hypervolume, 2))
    await c_1.shutdown()


async def wait_for_term(controller_agent):
    while len(controller_agent.roles[0]._weight_map.values()) != 1 or \
            list(controller_agent.roles[0]._weight_map.values())[0] != 1:
        await asyncio.sleep(0.1)
    print('Terminated!')
