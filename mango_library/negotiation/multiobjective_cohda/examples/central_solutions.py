import time

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.result import Result
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from evoalgos.selection import HyperVolumeContributionSelection
import pandas as pd

NUM_SOLUTION_POINTS = 25

PROBLEM = 'Zitzler_2'
ALGORITHM = NSGA2(pop_size=NUM_SOLUTION_POINTS)
NUM_AGENTS = 30
number_of_runs = 25


def get_solution(problem):
    if problem == 'Zitzler_3':
        p = get_problem('zdt3')
    elif problem == 'Zitzler_2':
        p = get_problem('zdt2')
    elif problem == 'Zitzler_1':
        p = get_problem('zdt1')
    else:
        try:
            p = get_problem(problem)
        except Exception:
            return f'no Problem Found for {problem}'
    result: Result = minimize(p, ALGORITHM)
    return result.F


if __name__ == '__main__':
    x = 0
    all_results = {'fronts': [], 'hv': [], 'duration': []}
    while x < number_of_runs:
        start_time = time.time()
        front = get_solution(PROBLEM)
        end_time = time.time()
        duration = end_time - start_time
        selection = HyperVolumeContributionSelection(prefer_boundary_points=False)
        selection.sorting_component.hypervolume_indicator.reference_point = (1.1, 6.9)
        hv = selection.sorting_component.hypervolume_indicator.assess_non_dom_front(front)
        all_results['fronts'].append(front.tolist())
        all_results['hv'].append(hv)
        all_results['duration'].append(duration)
        x += 1
    pd.DataFrame(all_results).to_csv(f"central_solution_{PROBLEM}.csv", index=False)
