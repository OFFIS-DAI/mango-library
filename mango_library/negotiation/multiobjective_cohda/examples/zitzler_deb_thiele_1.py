import asyncio
import math

import numpy as np

from mango_library.negotiation.multiobjective_cohda.data_classes import Target
from mango_library.negotiation.multiobjective_cohda.examples.simulation_util import simulate_mo_cohda
from mango_library.negotiation.multiobjective_cohda.multiobjective_cohda import MoCohdaNegotiation

SIM_NAME = 'Zitzler_1'

NUM_AGENTS = 30
NUM_SCHEDULES = 10
NUM_SOLUTION_POINTS = 5
NUM_ITERATIONS = 1
CHECK_INBOX_INTERVAL = 0.05

PICK_FKT = MoCohdaNegotiation.pick_all_points
# PICK_FKT = MoCohdaNegotiation.pick_random_point
MUTATE_FKT = MoCohdaNegotiation.mutate_with_all_possible
# MUTATE_FKT = MoCohdaNegotiation.mutate_with_one_random

NUM_SIMULATIONS = 1


def g(cs):
    return 1 + 9 / 29 * cs.sum(axis=0)[1]


def h(cs):
    return 1 - math.sqrt(target_func_1(cs) / g(cs))


def target_func_1(cs):
    """
    cs.sum(axis=0)[0]
    """
    return cs.sum(axis=0)[0]


def target_func_2(cs):
    """
    (1 + 9/29 * cs.sum(axis=0)[1]) * (1 - math.sqrt(cs.sum(axis=0)[0]/g(cs)))
    """
    return g(cs) * h(cs)


TARGET_1 = Target(target_function=target_func_1, ref_point=1.1)
TARGET_2 = Target(target_function=target_func_2, ref_point=1.1)
TARGETS = [TARGET_1, TARGET_2]

SCHEDULE_STEP_SIZE = 1 / (NUM_SCHEDULES - 1)
SINGLE_POINT_SCHEDULES = [SCHEDULE_STEP_SIZE * i for i in range(NUM_SCHEDULES)]
POSSIBLE_SCHEDULES = []
for i in range(NUM_AGENTS):
    if i == 0:
        POSSIBLE_SCHEDULES.append([np.array([p, 0]) for p in SINGLE_POINT_SCHEDULES])
    else:
        POSSIBLE_SCHEDULES.append([np.array([0, p]) for p in SINGLE_POINT_SCHEDULES])


async def simulate_zitzler_1(name):
    await simulate_mo_cohda(
        num_simulations=NUM_SIMULATIONS,
        num_agents=NUM_AGENTS,
        possible_schedules=POSSIBLE_SCHEDULES, schedules_all_equal=False,
        targets=TARGETS, num_solution_points=NUM_SOLUTION_POINTS, num_iterations=NUM_ITERATIONS,
        check_inbox_interval=CHECK_INBOX_INTERVAL, pick_func=PICK_FKT, mutate_func=MUTATE_FKT,
        sim_name=name
    )


if __name__ == '__main__':
    asyncio.run(simulate_zitzler_1(SIM_NAME))
