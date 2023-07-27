import asyncio

import numpy as np
from pymoo.core.problem import Problem

from mango_library.negotiation.multiobjective_cohda.data_classes import Target
from mango_library.negotiation.multiobjective_cohda.examples.simulation_util import simulate_mo_cohda_NSGA2
from mango_library.negotiation.multiobjective_cohda.multiobjective_cohda import MoCohdaNegotiation

SIM_NAME = 'Schaffer_1'
A = 10
NUM_AGENTS = 1
NUM_SOLUTION_POINTS = 5
NUM_ITERATIONS = 1
CHECK_INBOX_INTERVAL = 0.05

PICK_FKT = MoCohdaNegotiation.pick_all_points
# PICK_FKT = MoCohdaNegotiation.pick_random_point
MUTATE_FKT = MoCohdaNegotiation.mutate_NSGA2

NUM_SIMULATIONS = 2


class Schaffer1_pymoo(Problem):

    def __init__(self):
        super().__init__(n_var=1,
                         n_obj=2,
                         xl=-A / 2,
                         xu=A / 2)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = [target_func_1(x)]
        f2 = [target_func_2(x)]
        out["F"] = np.column_stack([f1, f2])


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

possible_interval = A / NUM_AGENTS


async def simulate_schaffer(name):
    p = Schaffer1_pymoo()
    await simulate_mo_cohda_NSGA2(
        num_simulations=NUM_SIMULATIONS,
        num_agents=NUM_AGENTS,
        targets=TARGETS, num_solution_points=NUM_SOLUTION_POINTS, num_iterations=NUM_ITERATIONS,
        check_inbox_interval=CHECK_INBOX_INTERVAL, pick_func=PICK_FKT, mutate_func=MUTATE_FKT, problem=p,
        lower_limit=-A, upper_limit=A, possible_interval=possible_interval, sim_name=name
    )


if __name__ == '__main__':
    asyncio.run(simulate_schaffer(SIM_NAME))
