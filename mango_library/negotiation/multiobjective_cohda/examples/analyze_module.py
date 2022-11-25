import math
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from mango_library.negotiation.multiobjective_cohda.examples.central_solutions import get_solution, \
    get_solution_certain_range
from evoalgos.selection import HyperVolumeContributionSelection

def get_performance_metrics(approximated_front, reference_front, reference_point, p, inside_exponent, minimize):
    performance_metrics = {}
    performance_metrics["HV"] = calculate_hypervolume(approximated_front, reference_point)
    performance_metrics["GD"] = calculate_generational_distance(approximated_front, reference_front, p, inside_exponent)
    performance_metrics["IGD"] = calculate_inverted_generational_distance(approximated_front, reference_front, p, inside_exponent)
    performance_metrics["Delta_p"] = calculate_averaged_hausdorff_distance(approximated_front, reference_front, p, inside_exponent)
    performance_metrics["C(A,B)"] = calculate_two_set_coverage(approximated_front, reference_front, minimize)
    performance_metrics["C(B,A)"] = calculate_two_set_coverage(reference_front, approximated_front, minimize)
    performance_metrics["Delta"] = calculate_delta_indicator(approximated_front, reference_front)
    performance_metrics["S"] = calculate_spacing(approximated_front, reference_front)

    return performance_metrics


"""Hypervolume (HV) is an indicator of both the convergence and diversity of an approximation front. Given a set X 
of solutions and their image A in the objective space, the hypervolume of A is the volume generated by the relation 
of the points of the Pareto front obtained with a given reference point, called the nadir point. The latter is 
usually chosen to be the worst values reached for each objective of the problem, thus guaranteeing that all the 
solutions of the obtained front will not be dominated by that corresponding to the nadir point. 
Since it uses the nadir point as a reference, the HV calculation does not depend on the availability of an optimal 
Pareto front. One of the main advantages of hypervolume is that it is able to capture in a single number both the 
closeness of the solutions to the optimal set and, to some extent, the spread of the solutions across objective 
space."""
def calculate_hypervolume(approximated_front, reference_point):
    selection = HyperVolumeContributionSelection(prefer_boundary_points=False)
    selection.sorting_component.hypervolume_indicator.reference_point = reference_point

    return selection.sorting_component.hypervolume_indicator.assess_non_dom_front(approximated_front)


"""Generational distance (GD) is used to measure the proximity of the approximate front A found by the algorithm 
from a reference front R, which is either the true Pareto front or a very good approximation to it. The distances
between each objective vector a in A and the closest objective vector r in R are averaged over the size of A. The 
GD metric is fast to compute and correlates with convergence to the reference front, but is very sensitive to the 
number of points found by a given algorithm. Thus, large approximation fronts of poor quality may be ranked highly
by GD.

- p-value is considered as p=1 in some papers and p=2 in some others
- some versions consider division by number of points inside the exponent 1/p, some outside
- see https://mlopez-ibanez.github.io/eaf/reference/igd.html
"""
def calculate_generational_distance(approximated_front, reference_front, p, inside_exponent):
    af_size, rf_size = len(approximated_front), len(reference_front)
    gd = 0
    for i in range(af_size):
        dist_min = min([np.linalg.norm(approximated_front[i] - reference_front[j]) for j in range(0, rf_size)])
        gd += dist_min ** p
    if inside_exponent:
        return (gd / af_size) ** (1/p)
    else:
        return (gd ** (1/p)) / af_size


"""The inverted generational distance (IGD) was proposed as an improvement over the GD based on the very simple idea 
of reversing the order of the fronts considered as input by the GD. In other words, the IGD equals the GD metric 
when computing the distance between each objective vector in the reference front and its closest objective vector in 
the approximation front, averaged over the size of the reference front. The main advantages of the IGD measure are 
twofold. One is its computational efficiency, the other is its capability to show the overall quality of an
obtained approximation front A (i.e., convergence to the Pareto front and diversity over the Pareto front).

- same as for gd
"""
def calculate_inverted_generational_distance(approximated_front, reference_front, p, inside_exponent):
    af_size, rf_size = len(approximated_front), len(reference_front)
    igd = 0
    for i in range(rf_size):
        dist_min = min([np.linalg.norm(reference_front[i] - approximated_front[j]) for j in range(0, af_size)])
        igd += dist_min ** p
    if inside_exponent:
        return (igd / rf_size) ** (1/p)
    else:
        return (igd ** (1/p)) / rf_size


"""The averaged Hausdorff distance (∆p) was proposed as an attempt to address potential drawbacks of the IGD. It is 
defined as an averaged Hausdorff distance metric, controlled by the parameter p. In particular, larger values of p
mean stronger penalties for outliers."""
def calculate_averaged_hausdorff_distance(approximated_front, reference_front, p, inside_exponent):
    delta_p = max(calculate_generational_distance(approximated_front, reference_front, p, inside_exponent),
                  calculate_inverted_generational_distance(approximated_front, reference_front, p, inside_exponent))
    return delta_p


"""Spread metric (Δ) examines how evenly the solutions are distributed among the approximation fronts in objective 
space. First, it calculates the Euclidean distance between the consecutive solutions in the obtained nondominated 
set of solutions. Then it calculates the average of these distances. After that, from the obtained set of 
non-dominated solutions the extreme solutions are calculated. Finally, it calculates the nonuniformity in the 
distribution. A low value of Δ metric indicates wide and uniform spread out of the solutions across the Pareto 
front. Thus, Δ = 0 indicates that the approximation front is as uniformly distributed as possible.

d_f = Euklidean distance between first point of approximated front and first point of reference front
d_l = Euklidean distance between last point of approximated front and last point of reference front
d_bar = average of all Euklidean distances between consecutive solutions
"""
def calculate_delta_indicator(approximated_front, reference_front):
    af_size = len(approximated_front)
    d_f = np.linalg.norm(approximated_front[0] - reference_front[0])
    d_l = np.linalg.norm(approximated_front[-1] - reference_front[-1])
    d_bar = (sum([np.linalg.norm(approximated_front[i] - approximated_front[i+1]) for i in range(0, af_size-1)])) / af_size
    d_sum = sum([abs(np.linalg.norm(approximated_front[i] - approximated_front[i+1]) - d_bar) for i in range(0, af_size-1)])
    delta = (d_f + d_l + d_sum) / (d_f + d_l + ((af_size - 1) * d_bar))
    return delta


"""The spacing metric (S) measures the dispersion of the obtained approximate front in comparison with the optimal 
Pareto front. The spacing must be as small as possible for the solution set to be of superior quality. A value of 
zero for this metric indicates that the solutions in the approximate front are equidistantly spaced.

d_i_list = list of Euklidean distances between point i in approximated front and closest point j in reference front
d_bar = mean of all d_i
"""
def calculate_spacing(approximated_front, reference_front):
    af_size = len(approximated_front)
    rf_size = len(reference_front)
    d_i_list = []
    for i in range(af_size):
        dist_min = min([np.linalg.norm(approximated_front[i] - reference_front[j]) for j in range(0, rf_size)])
        d_i_list.append(dist_min)
    d_bar = sum(d_i_list) / af_size
    d_sum = sum([(d_i_list[i] - d_bar) ** 2 for i in range(0, af_size)])
    s = math.sqrt(d_sum / (af_size - 1))
    return s


"""Coverage of two sets (C) metric compares the quality of two non-dominated sets. If C(A, B) = 1, all the candidate 
solutions in B are dominated by or equal to at least one solution in A. If C(A, B) = 0, no candidate solutions in B is 
covered by any solution in A."""
def calculate_two_set_coverage(front_A, front_B, minimize):
    covered_solutions = 0
    for point_B in front_B:
        for point_A in front_A:
            if dominates(point_A, point_B, minimize):
                covered_solutions += 1
                # print("Solution ", point_B, "is covered by", point_A)
                break
    c = covered_solutions / len(front_B)
    return c


def dominates(point_A, point_B, minimize):
    if minimize:
        if np.any(point_A == point_B) and np.any(point_A < point_B):
            return True
        elif np.all(point_A < point_B):
            return True
        elif np.all(point_A == point_B):
            return True
        else:
            return False
    else:
        if np.any(point_B > point_A) and np.all(point_B >= point_A):
            return True
        else:
            return False


""" Create reference fronts for different optimization problems"""
def create_reference_front(problem, number_of_points):
    performances = []
    values = []
    if problem == "Zitzler_3":
        points_per_part = number_of_points / 5
        # for boundary values: https://pymoo.org/problems/multi/zdt.html
        all_boundaries = [(0.0, 0.0830), (0.1822, 0.2577), (0.4093, 0.4538), (0.6183, 0.6525), (0.8233, 0.8518)]

        for part in all_boundaries:
            lower_boundary = part[0]
            upper_boundary = part[1]
            step_size = (upper_boundary - lower_boundary) / points_per_part
            n = lower_boundary
            while n <= upper_boundary:
                values.append(n)
                n = n + step_size

        for x in values:
            f1 = x
            f2 = 1 - math.sqrt(x/1) - (x/1) * math.sin(10 * math.pi * x)
            performances.append([f1, f2])

    elif problem == "Zitzler_1":
        lower_boundary = 0
        upper_boundary = 1
        step_size = upper_boundary / number_of_points
        n = lower_boundary
        while n <= upper_boundary:
            values.append(n)
            n = n + step_size

        for x in values:
            f1 = x
            f2 = 1 - math.sqrt(f1)
            performances.append([f1, f2])

    return np.array(performances)


if __name__ == '__main__':

    PROBLEM = "Zitzler_1"
    PROBLEM = "Zitzler_3"
    REFERENCE_POINT = (1.1, 1.1)
    P = 2
    INSIDE_EXPONENT = False
    MINIMIZE = True

    path = os.path.dirname(__file__)

    results = h5py.File(path+'/'+PROBLEM+'.hdf5', 'r')

    # get approximated front from database
    performances = np.array(results.get('Results').get('Results_0').get('performances'))
    approximated_front = []
    for performance_tuple in performances:
        f1 = float(performance_tuple[0])
        f2 = float(performance_tuple[1])
        approximated_front.append([f1, f2])

    # create reference front
    reference_front = create_reference_front(PROBLEM, 500)

    # create front with central approach
    central_front = get_solution_certain_range(PROBLEM)

    # calculate and print metrics
    metrics_approximated_front = get_performance_metrics(np.array(approximated_front), np.array(reference_front),
                                                         REFERENCE_POINT, P, INSIDE_EXPONENT, MINIMIZE)
    metrics_central_front = get_performance_metrics(np.array(central_front), np.array(reference_front),
                                                         REFERENCE_POINT, P, INSIDE_EXPONENT, MINIMIZE)
    metrics_reference_front = get_performance_metrics(np.array(reference_front), np.array(reference_front),
                                                         REFERENCE_POINT, P, INSIDE_EXPONENT, MINIMIZE)

    print("MO-COHDA", metrics_approximated_front)
    print("Central", metrics_central_front)
    print("Reference Front", metrics_reference_front)

    # Create Plots
    figure, axis = plt.subplots(2, 2)
    axis[0, 0].scatter(np.array(approximated_front)[:, 0], np.array(approximated_front)[:, 1], s=3, c='#1f77b4')
    axis[0, 0].set_title("MO-COHDA")
    axis[0, 1].scatter(central_front[:, 0], central_front[:, 1], s=3, c='#ff7f0e')
    axis[0, 1].set_title("Central")
    axis[1, 0].scatter(reference_front[:, 0], reference_front[:, 1], s=3, c='#2ca02c')
    axis[1, 0].set_title("Reference Front")
    axis[1, 1].scatter(np.array(approximated_front)[:, 0], np.array(approximated_front)[:, 1], s=3)
    axis[1, 1].scatter(central_front[:, 0], central_front[:, 1], s=3)
    axis[1, 1].scatter(reference_front[:, 0], reference_front[:, 1], s=3)
    axis[1, 1].set_title("All")
    plt.show()