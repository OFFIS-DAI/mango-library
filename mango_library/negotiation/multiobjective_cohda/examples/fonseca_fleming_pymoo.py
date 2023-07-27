import asyncio
import math
import time

import numpy as np
from pymoo.core.problem import Problem

from mango_library.negotiation.multiobjective_cohda.data_classes import Target
from mango_library.negotiation.multiobjective_cohda.examples.simulation_util import store_in_db, \
    simulate_mo_cohda_NSGA2
from mango_library.negotiation.multiobjective_cohda.multiobjective_cohda import MoCohdaNegotiation

SIM_NAME = 'Fonseca_Fleming'

NUM_AGENTS = 10
NUM_SOLUTION_POINTS = 1
NUM_ITERATIONS = 1
CHECK_INBOX_INTERVAL = 0.05

PICK_FKT = MoCohdaNegotiation.pick_all_points
# PICK_FKT = MoCohdaNegotiation.pick_random_point
MUTATE_FKT = MoCohdaNegotiation.mutate_NSGA2

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


class FonsecaFleming(Problem):

    def __init__(self):
        super().__init__(n_var=1,
                         n_obj=2,
                         xl=-4,
                         xu=4)

    def _evaluate(self, x, out, *args, **kwargs):
        x = x.sum(axis=0)
        f1 = [target_func_1(x)]
        f2 = [target_func_2(x)]
        out["F"] = np.column_stack([f1, f2])


TARGET_1 = Target(target_function=target_func_1, ref_point=4.1, maximize=False)
TARGET_2 = Target(target_function=target_func_2, ref_point=4.1, maximize=False)
TARGETS = [TARGET_1, TARGET_2]

possible_interval = 8 / NUM_AGENTS


async def simulate_fonseca(name):
    p = FonsecaFleming()
    await simulate_mo_cohda_NSGA2(
        num_simulations=NUM_SIMULATIONS,
        num_agents=NUM_AGENTS,
        targets=TARGETS, num_solution_points=NUM_SOLUTION_POINTS, num_iterations=NUM_ITERATIONS,
        check_inbox_interval=CHECK_INBOX_INTERVAL, pick_func=PICK_FKT, mutate_func=MUTATE_FKT, problem=p,
        lower_limit=-4, upper_limit=4, possible_interval=possible_interval, sim_name=name
    )


if __name__ == '__main__':
    asyncio.run(simulate_fonseca(SIM_NAME))
