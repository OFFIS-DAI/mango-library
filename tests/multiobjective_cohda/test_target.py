from mango_library.negotiation.multiobjective_cohda.data_classes import Target


def sum_fct(cs, target_params=None):
    return sum([sum(i) for i in zip(*cs)])


def test_target():
    # ref point is -1, maximize function
    ref_point = -1
    maximize_target = Target(target_function=sum_fct, ref_point=ref_point,
                             maximize=True)

    example_low = [[0.1, 0.1], [0.1, 0.1]]
    example_low_sum = sum_fct(example_low)
    target_low = maximize_target.performance(example_low)

    example_high = [[0.2, 0.2], [0.2, 0.2]]
    example_high_sum = sum_fct(example_high)
    target_high = maximize_target.performance(example_high)

    # In this case, the target function should maximize. The values of
    # the example_high are lower and with the used target function,
    # the sum should be higher. For maximizing, these would be the results
    assert example_low_sum < example_high_sum

    # the first (worse) example with the lower number is closer to the ref
    # point
    assert abs(ref_point - example_low_sum) < abs(ref_point - example_high_sum)

    # But since the Target object internally always
    # minimizes, the target_high should be smaller than the first one
    assert target_high < target_low

    # the internal ref point should also be inverted
    assert maximize_target.ref_point == - ref_point

    # the second example is better (higher), therefore the distance from
    # it to the ref point is larger than from first example to ref point
    assert abs(maximize_target.ref_point - target_low) < abs(
        maximize_target.ref_point - target_high)

    # create target with minimize function
    # ref point is 1, maximize function
    ref_point = 1
    minimize_target = Target(target_function=sum_fct, ref_point=ref_point,
                             maximize=False)
    target_low = sum_fct(example_low)
    target_high = sum_fct(example_high)

    # this time its minimizing, that is why the larger result (target high) is
    # worse and closer to ref point
    assert abs(minimize_target.ref_point - target_high) < abs(
        minimize_target.ref_point - target_low)
