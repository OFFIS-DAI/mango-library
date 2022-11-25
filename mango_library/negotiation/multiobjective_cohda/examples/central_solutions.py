from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.result import Result
from pymoo.optimize import minimize
from pymoo.problems import get_problem

NUM_SOLUTION_POINTS = 20

PROBLEM = 'Zitzler_3'
ALGORITHM = NSGA2(pop_size=NUM_SOLUTION_POINTS)
NUM_AGENTS = 30


def get_solution(problem):
    if problem == 'Zitzler_3':
        p = get_problem('zdt3')
    elif problem == 'Zitzler_1':
        p = get_problem('zdt1')
    else:
        return f'no Problem Found for {PROBLEM}'

    result: Result = minimize(p, ALGORITHM)
    return result.F


def get_solution_certain_range(problem):
    NUM_AGENTS = 10
    possible_range = 1 / NUM_AGENTS

    if problem == 'Zitzler_3':
        p = get_problem('zdt3')
    elif problem == 'Zitzler_1':
        p = get_problem('zdt1')
    else:
        return f'no Problem Found for {PROBLEM}'

    example_schedule = [possible_range for _ in range(30)]
    example_cluster_schedule = [example_schedule for _ in range(NUM_AGENTS)]
    example_sum = [sum(l) for l in zip(*example_cluster_schedule)]

    agent_id = 1
    old_schedule = example_cluster_schedule[agent_id]
    diff_to_upper = [possible_range - old_schedule[i] for i in range(len(old_schedule))]
    diff_to_lower = old_schedule

    new_xl = [example_sum[i] - diff_to_lower[i] if example_sum[i] - diff_to_lower[i] >= 0 else 0 for i in
              range(len(old_schedule))]
    # TODO maximal 1?
    new_xu = [example_sum[i] + diff_to_upper[i] if example_sum[i] + diff_to_upper[i] <= 1 else 1 for i in
              range(len(old_schedule))]

    for idx in range(len(new_xl)):
        p.xl[idx] = new_xl[idx]
        p.xu[idx] = new_xu[idx]

    result: Result = minimize(p, ALGORITHM)
    return result.F


if __name__ == '__main__':
    print(get_solution_certain_range(PROBLEM))
