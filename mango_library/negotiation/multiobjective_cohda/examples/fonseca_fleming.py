import asyncio
import math
import time

import numpy as np

from mango_library.negotiation.multiobjective_cohda.data_classes import Target
from mango_library.negotiation.multiobjective_cohda.examples.simulation_util import simulate_mo_cohda, store_in_db
from mango_library.negotiation.multiobjective_cohda.multiobjective_cohda import MoCohdaNegotiation

SIM_NAME = 'Fonseca_Fleming'

NUM_AGENTS = 10
NUM_SCHEDULES = 25
NUM_SOLUTION_POINTS = 10
NUM_ITERATIONS = 1
CHECK_INBOX_INTERVAL = 0.05

PICK_FKT = MoCohdaNegotiation.pick_all_points
# PICK_FKT = MoCohdaNegotiation.pick_random_point
MUTATE_FKT = MoCohdaNegotiation.mutate_with_all_possible
# MUTATE_FKT = MoCohdaNegotiation.mutate_with_one_random

NUM_SIMULATIONS = 2
DENOMINATOR = math.sqrt(NUM_AGENTS)


def target_func_1(cs: np.array):
    """
    exponent = 0
    for x_i in cs:
        exponent += (float(x_i) - 1/math.sqrt(NUM_AGENTS)) ** 2
    return 1 - math.exp(-exponent)
    """
    exponent = 0

    for x_i in cs:
        exponent += (float(x_i) - 1 / DENOMINATOR) ** 2
    return 1 - math.exp(-exponent)


def target_func_2(cs):
    """
    exponent = 0
    for x_i in cs:
        exponent += (float(x_i) + 1/math.sqrt(NUM_AGENTS)) ** 2
    return 1 - math.exp(-exponent)

    """
    exponent = 0
    for x_i in cs:
        exponent += (float(x_i) + 1 / DENOMINATOR) ** 2
    return 1 - math.exp(-exponent)


# minimize
TARGET_1 = Target(target_function=target_func_1, ref_point=1.1, maximize=False)
TARGET_2 = Target(target_function=target_func_2, ref_point=1.1, maximize=False)
TARGETS = [TARGET_1, TARGET_2]

SCHEDULE_STEP_SIZE = 8 / (NUM_SCHEDULES - 1)
# each agent has a range from -4 until 4
SINGLE_POINT_SCHEDULES = [np.array([-4 + SCHEDULE_STEP_SIZE * i]) for i in range(NUM_SCHEDULES)]

POSSIBLE_SCHEDULES = SINGLE_POINT_SCHEDULES


async def simulate_fonseca(name):
    await simulate_mo_cohda(
        num_simulations=NUM_SIMULATIONS,
        num_agents=NUM_AGENTS,
        possible_schedules=POSSIBLE_SCHEDULES, schedules_all_equal=True,
        targets=TARGETS, num_solution_points=NUM_SOLUTION_POINTS, num_iterations=NUM_ITERATIONS,
        check_inbox_interval=CHECK_INBOX_INTERVAL, pick_func=PICK_FKT, mutate_func=MUTATE_FKT,
        sim_name=name, store_updates_to_db=True
    )


if __name__ == '__main__':
    asyncio.run(simulate_fonseca(SIM_NAME))
