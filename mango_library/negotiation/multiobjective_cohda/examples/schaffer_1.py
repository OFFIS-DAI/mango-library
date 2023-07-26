import asyncio
import time

import numpy as np

from mango_library.negotiation.multiobjective_cohda.data_classes import Target
from mango_library.negotiation.multiobjective_cohda.examples.simulation_util import simulate_mo_cohda, store_in_db
from mango_library.negotiation.multiobjective_cohda.multiobjective_cohda import MoCohdaNegotiation

FILE = 'Schaffer_1.hdf5'
SIM_NAME = 'Schaffer_1'
A = 10
NUM_AGENTS = 10
NUM_SCHEDULES = 20
NUM_SOLUTION_POINTS = 5
NUM_ITERATIONS = 1
CHECK_INBOX_INTERVAL = 0.05

PICK_FKT = MoCohdaNegotiation.pick_all_points
# PICK_FKT = MoCohdaNegotiation.pick_random_point
MUTATE_FKT = MoCohdaNegotiation.mutate_with_all_possible
# MUTATE_FKT = MoCohdaNegotiation.mutate_with_one_random

NUM_SIMULATIONS = 1


def target_func_1(cs):
    """
    x ** 2
    """
    return cs.sum() ** 2


def target_func_2(cs):
    """
    (x - 2) ** 2
    """
    return (cs.sum() - 2) ** 2


TARGET_1 = Target(target_function=target_func_1, ref_point=A ** 2 * 1.1)
TARGET_2 = Target(target_function=target_func_2, ref_point=(A + 2) ** 2 * 1.1)
TARGETS = [TARGET_1, TARGET_2]

SCHEDULE_THRESHOLD = A / NUM_AGENTS
SCHEDULE_STEP_SIZE = (SCHEDULE_THRESHOLD * 2) / (NUM_SCHEDULES - 1)
POSSIBLE_SCHEDULES = []
# each agent receives schedules with values from -1 until 1, since A / num_agents equals this possible
# interval per agent
for schedule_no in range(NUM_SCHEDULES):
    POSSIBLE_SCHEDULES.append(np.array([-SCHEDULE_THRESHOLD + schedule_no * SCHEDULE_STEP_SIZE]))


async def simulate_schaffer(name, db_file):
    await simulate_mo_cohda(
        num_simulations=NUM_SIMULATIONS,
        num_agents=NUM_AGENTS,
        possible_schedules=POSSIBLE_SCHEDULES, schedules_all_equal=True,
        targets=TARGETS, num_solution_points=NUM_SOLUTION_POINTS, num_iterations=NUM_ITERATIONS,
        check_inbox_interval=CHECK_INBOX_INTERVAL, pick_func=PICK_FKT, mutate_func=MUTATE_FKT, db_file=db_file,
        sim_name=name
    )


if __name__ == '__main__':
    asyncio.run(simulate_schaffer(SIM_NAME, FILE))
