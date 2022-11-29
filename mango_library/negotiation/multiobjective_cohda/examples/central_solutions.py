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


if __name__ == '__main__':
    print(get_solution(PROBLEM))
