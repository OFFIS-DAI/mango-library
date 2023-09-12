import asyncio
import math
import random

from mango_library.negotiation.multiobjective_cohda.data_classes import Target
from mango_library.negotiation.multiobjective_cohda.examples.simulation_util import simulate_mo_cohda
from mango_library.negotiation.multiobjective_cohda.multiobjective_cohda import MoCohdaNegotiation

# In this implementation of the problem Zitzler 3, every agent can control each variable for
# a certain level. For the version in which each agent controls one variable (thus 30 agents are taken into account),
# have a look at: Zitzler_3_version_2.py.
SIM_NAME = "Zitzler_3"

NUM_AGENTS = 5
NUM_SCHEDULES = 30
NUM_SOLUTION_POINTS = 10
NUM_ITERATIONS = 1
CHECK_INBOX_INTERVAL = 0.05

PICK_FKT = MoCohdaNegotiation.pick_all_points
# PICK_FKT = MoCohdaNegotiation.pick_random_point
MUTATE_FKT = MoCohdaNegotiation.mutate_with_all_possible
# MUTATE_FKT = MoCohdaNegotiation.mutate_with_one_random_schedule

NUM_SIMULATIONS = 2


def g(cs):
    return 1 + 9 / 29 * cs.sum(axis=0)[1]


def h(cs):
    return 1 - math.sqrt(target_func_1(cs) / g(cs)) - (target_func_1(cs) / g(cs)) * math.sin(
        10 * math.pi * target_func_1(cs))


def target_func_1(cs):
    """
    cs.sum(axis=0)[0]
    """
    return cs.sum(axis=0)[0]


def target_func_2(cs):
    """
    1 + 9/29 * cs.sum(axis=0)[1] *
    (1 - math.sqrt(target_func_1(cs)/g(cs)) - (target_func_1(cs)/g(cs)) * math.sin(10 * math.pi * target_func_1(cs)))

    """
    return g(cs) * h(cs)


# minimize, x between 0 and 1
TARGET_1 = Target(target_function=target_func_1, ref_point=1.1, maximize=False)
TARGET_2 = Target(target_function=target_func_2, ref_point=1.1, maximize=False)
TARGETS = [TARGET_1, TARGET_2]

# each agent can control all variables
# determine possible interval per agent,
possible_interval = 1 / NUM_AGENTS
SCHEDULE_LENGTH = 30
# create schedules with this interval, each schedule has 30 entries, different variants per variable
POSSIBLE_SCHEDULES = []
schedules_per_agent = []

for _ in range(NUM_AGENTS):
    # totally random
    for _ in range(int(NUM_SCHEDULES / 2)):
        schedules_per_agent.append([random.uniform(0, possible_interval) for _ in range(SCHEDULE_LENGTH)])

    # every interval for each entry
    possible_range = possible_interval / int(NUM_SCHEDULES / 2)
    current_range = 0
    for _ in range(int(NUM_SCHEDULES / 2)):
        schedules_per_agent.append([current_range for _ in range(SCHEDULE_LENGTH)])
        current_range += possible_range
        if current_range > possible_interval:
            break
    POSSIBLE_SCHEDULES.append(schedules_per_agent)
    schedules_per_agent = []


async def simulate_zitzler_3(name):
    await simulate_mo_cohda(
        num_simulations=NUM_SIMULATIONS,
        num_agents=NUM_AGENTS,
        possible_schedules=POSSIBLE_SCHEDULES, schedules_all_equal=False,
        targets=TARGETS, num_solution_points=NUM_SOLUTION_POINTS, num_iterations=NUM_ITERATIONS,
        check_inbox_interval=CHECK_INBOX_INTERVAL, pick_func=PICK_FKT, mutate_func=MUTATE_FKT,
        store_updates_to_db=False, sim_name=name
    )


if __name__ == '__main__':
    asyncio.run(simulate_zitzler_3(SIM_NAME))
