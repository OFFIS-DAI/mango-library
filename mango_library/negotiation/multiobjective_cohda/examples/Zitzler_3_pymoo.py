import asyncio
import math

from mango_library.negotiation.multiobjective_cohda.data_classes import Target
from mango_library.negotiation.multiobjective_cohda.examples.simulation_util import store_in_db, \
    simulate_mo_cohda_NSGA2
from mango_library.negotiation.multiobjective_cohda.multiobjective_cohda import COHDA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.result import Result
from pymoo.optimize import minimize
from pymoo.problems import get_problem

FILE = 'Zitzler_3_pymoo.hdf5'
SIM_NAME = 'Zitzler_3'

NUM_AGENTS = 5
NUM_SCHEDULES = 30
NUM_SOLUTION_POINTS = 10
NUM_ITERATIONS = 1
CHECK_INBOX_INTERVAL = 0.05

PICK_FKT = COHDA.pick_random_point
MUTATE_FKT = COHDA.mutate_NSGA2

NUM_SIMULATIONS = 1
p = get_problem('zdt3')
ALGORITHM = NSGA2(pop_size=NUM_SOLUTION_POINTS)


def target_func_1(cs):
    """
    """

    output = p.evaluate(cs, ALGORITHM)
    solution_tuple = output[0]
    result_target_1 = solution_tuple[0][0]

    return result_target_1


def target_func_2(cs):
    """
    """
    output = p.evaluate(cs, ALGORITHM)
    solution_tuple = output[0]
    result_target_2 = solution_tuple[0][1]

    return result_target_2


TARGET_1 = Target(target_function=target_func_1, ref_point=1.1)
TARGET_2 = Target(target_function=target_func_2, ref_point=1.1)
TARGETS = [TARGET_1, TARGET_2]

possible_interval = 1 / NUM_AGENTS


async def simulate_zitzler(name, db_file):
    results = await simulate_mo_cohda_NSGA2(
        num_simulations=NUM_SIMULATIONS,
        num_agents=NUM_AGENTS,
        targets=TARGETS, num_solution_points=NUM_SOLUTION_POINTS, num_iterations=NUM_ITERATIONS,
        check_inbox_interval=CHECK_INBOX_INTERVAL, pick_func=PICK_FKT, mutate_func=MUTATE_FKT,
    )

    store_in_db(
        db_file=db_file, sim_name=name, n_agents=NUM_AGENTS, targets=TARGETS,
        n_solution_points=NUM_SOLUTION_POINTS, n_iterations=NUM_ITERATIONS, check_inbox_interval=CHECK_INBOX_INTERVAL,
        mutate_func=MUTATE_FKT, pick_func=PICK_FKT, results=results
    )


if __name__ == '__main__':
    asyncio.run(simulate_zitzler(SIM_NAME, FILE))
