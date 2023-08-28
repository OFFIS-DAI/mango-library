import asyncio

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem

from mango_library.negotiation.multiobjective_cohda.data_classes import Target
from mango_library.negotiation.multiobjective_cohda.examples.simulation_util import simulate_mo_cohda_NSGA2
from mango_library.negotiation.multiobjective_cohda.multiobjective_cohda import MoCohdaNegotiation

SIM_NAME = 'Zitzler_1_nsga2_'
PROBLEM = 'zdt1'

NUM_AGENTS = 30
NUM_SOLUTION_POINTS = 1
POPULATION_SIZE = 1
NUM_ITERATIONS = 1
CHECK_INBOX_INTERVAL = 0.05

PICK_FKT = MoCohdaNegotiation.pick_all_points
# PICK_FKT = MoCohdaNegotiation.pick_random_point
MUTATE_FKT = MoCohdaNegotiation.mutate_NSGA2

NUM_SIMULATIONS = 2
p = get_problem(PROBLEM)
ALGORITHM = NSGA2(pop_size=POPULATION_SIZE)


def target_func_1(cs):
    """
    """
    output = p.evaluate(cs, ALGORITHM)
    result_target_1 = output[0][0]
    return result_target_1


def target_func_2(cs):
    """
    """
    output = p.evaluate(cs, ALGORITHM)
    result_target_2 = output[0][1]
    return result_target_2


# minimize function, range between 0 and 1
TARGET_1 = Target(target_function=target_func_1, ref_point=1.1, maximize=False)
TARGET_2 = Target(target_function=target_func_2, ref_point=1.1, maximize=False)
TARGETS = [TARGET_1, TARGET_2]

possible_interval = 1 / NUM_AGENTS


async def simulate_zitzler_1_NSGA2(name):
    await simulate_mo_cohda_NSGA2(
        num_simulations=NUM_SIMULATIONS,
        num_agents=NUM_AGENTS,
        targets=TARGETS, num_solution_points=NUM_SOLUTION_POINTS, num_iterations=NUM_ITERATIONS,
        check_inbox_interval=CHECK_INBOX_INTERVAL, pick_func=PICK_FKT, mutate_func=MUTATE_FKT, problem=PROBLEM,
        possible_interval=possible_interval, population_size=POPULATION_SIZE, sim_name=name
    )


if __name__ == '__main__':
    asyncio.run(simulate_zitzler_1_NSGA2(SIM_NAME))
