import asyncio
from copy import deepcopy

import numpy as np
import pytest
from mango import create_container
from mango import RoleAgent

from mango_library.coalition.core import (
    CoalitionParticipantRole,
    CoalitionInitiatorRole,
)
from mango_library.negotiation.multiobjective_cohda.cohda_messages import (
    MoCohdaNegotiationMessage,
)
from mango_library.negotiation.multiobjective_cohda.data_classes import (
    Target,
    SolutionCandidate,
    SolutionPoint,
)
from mango_library.negotiation.multiobjective_cohda.mocohda_solution_aggregation import (
    MoCohdaSolutionAggregationRole,
)
from mango_library.negotiation.multiobjective_cohda.mocohda_starting import (
    MoCohdaNegotiationDirectStarterRole,
)
from mango_library.negotiation.multiobjective_cohda.multiobjective_cohda import (
    MultiObjectiveCOHDARole,
    MoCohdaNegotiationModel,
    MoCohdaSolutionModel,
)
from mango_library.negotiation.termination import (
    NegotiationTerminationParticipantRole,
    NegotiationTerminationDetectorRole,
)

NUM_ITERATIONS = 1
NUM_AGENTS = 10
NUM_CANDIDATES = 2
CHECK_MSG_QUEUE_INTERVAL = 1
SCHEDULES_FOR_AGENTS_SIMPEL = [
    [
        [0.1, 0.7],
        [0.1, 0.1],
    ],
    [
        [0.1, 0.9],
        [0.2, 0.2],
    ],
    [
        [0.2, 0.7],
        [0.4, 0.4],
    ],
]


def determine_deviations(cs, target_params):
    dev_per_time_step = []
    sum_cs = sum(map(abs, cs))
    for idx, entry in enumerate(sum_cs):
        if idx + 1 < len(sum_cs):
            dev_per_time_step.append(abs(entry - sum_cs[idx + 1]))
    return float(np.mean(dev_per_time_step))


def determine_sums(cs, target_params):
    sum_cs = sum(map(abs, cs))
    return np.max(sum_cs)


MAXIMIZE_TARGETS = [
    Target(target_function=determine_deviations, ref_point=-1, maximize=True),
    Target(target_function=determine_sums, ref_point=-1, maximize=True),
]


@pytest.mark.asyncio
async def test_coalition_to_mocohda_with_termination():
    # create container
    c = await create_container(addr=("127.0.0.3", 5555))

    # create cohda_agents
    cohda_agents = []
    addrs = []
    controller_agent = RoleAgent(c)
    termination_detector_role = NegotiationTerminationDetectorRole(
        aggregator_addr=c.addr, aggregator_id=controller_agent.aid
    )
    controller_agent.add_role(termination_detector_role)
    aggregation_role = MoCohdaSolutionAggregationRole(
        solution_point_choosing_function=choose_first_solution_point
    )
    controller_agent.add_role(aggregation_role)

    schedules_all_equal = False

    def provide_schedules(index):
        if not schedules_all_equal:
            return deepcopy(lambda: SCHEDULES_FOR_AGENTS_SIMPEL[index])
        else:
            return lambda: SCHEDULES_FOR_AGENTS_SIMPEL

    for i in range(NUM_AGENTS):
        a = RoleAgent(c)
        cohda_role = MultiObjectiveCOHDARole(
            schedule_provider=provide_schedules(i % len(SCHEDULES_FOR_AGENTS_SIMPEL)),
            targets=MAXIMIZE_TARGETS,
            local_acceptable_func=lambda s: True,
            num_solution_points=NUM_CANDIDATES,
            num_iterations=NUM_ITERATIONS,
            check_inbox_interval=CHECK_MSG_QUEUE_INTERVAL,
            pick_func=None,
            mutate_func=None,
        )
        a.add_role(cohda_role)
        a.add_role(CoalitionParticipantRole())
        a.add_role(
            NegotiationTerminationParticipantRole(
                negotiation_model_class=MoCohdaNegotiationModel,
                negotiation_message_class=MoCohdaNegotiationMessage,
            )
        )
        addrs.append((c.addr, a.aid))
        cohda_agents.append(a)

    coalition_initiator_role = CoalitionInitiatorRole(
        addrs, "mocohda", "mocohda-negotiation"
    )
    controller_agent.add_role(coalition_initiator_role)

    await wait_for_assignments_sent(coalition_initiator_role)
    print("Starts negotiations")
    await asyncio.sleep(0.5)

    cohda_agents[0].add_role(
        MoCohdaNegotiationDirectStarterRole(
            target_params=None, num_solution_points=NUM_CANDIDATES
        )
    )

    for a in cohda_agents + [controller_agent]:
        if a._check_inbox_task.done():
            if a._check_inbox_task.exception() is not None:
                raise a._check_inbox_task.exception()
            else:
                assert False, f"check_inbox terminated unexpectedly."

    await asyncio.wait_for(wait_for_solution_confirmed(aggregation_role), timeout=10)

    # gracefully shutdown
    for a in cohda_agents + [controller_agent]:
        await a.shutdown()
    await c.shutdown()

    # TODO: in windows this ==2
    # assert len(asyncio.all_tasks()) == 1, f'Too many Tasks are running{asyncio.all_tasks()}'
    cohda_negotiation = list(
        cohda_agents[0]
        ._role_context.get_or_create_model(MoCohdaNegotiationModel)
        ._negotiations.values()
    )[0]
    cohda_negotiation_id = list(
        cohda_agents[0]
        ._role_context.get_or_create_model(MoCohdaNegotiationModel)
        ._negotiations.keys()
    )[0]
    cluster_schedule = cohda_negotiation._memory.solution_candidate.solution_points[
        0
    ].cluster_schedule

    solution_in_aggregator = aggregation_role.cohda_solutions[cohda_negotiation_id]

    for index, agent in enumerate(cohda_agents):
        # Assert first schedule was chosen
        assert np.array_equal(
            get_final_schedule(agent),
            SCHEDULES_FOR_AGENTS_SIMPEL[index % len(SCHEDULES_FOR_AGENTS_SIMPEL)][0],
        )
        # Assert the unit agents have the same schedule as the aggregator
        assert np.array_equal(
            get_final_schedule(agent),
            solution_in_aggregator.cluster_schedule[
                solution_in_aggregator.idx[agent.aid[5:]]
            ],
        )

    assert np.array_equal(cluster_schedule[0], [0.1, 0.7])
    assert next(iter(controller_agent.roles[0]._weight_map.values())) == 1


async def wait_for_solution_confirmed(aggregation_role):
    while len(aggregation_role._confirmed_cohda_solutions) == 0:
        await asyncio.sleep(0.05)
    print("Solution confirmed")


async def wait_for_assignments_sent(coalition_initiator_role):
    while not coalition_initiator_role._assignments_sent:
        await asyncio.sleep(0.05)


def get_final_schedule(cohda_agent):
    return list(
        cohda_agent._role_context.get_or_create_model(
            MoCohdaSolutionModel
        )._final_schedules.values()
    )[0]


def choose_first_solution_point(solution_front: SolutionCandidate) -> SolutionPoint:
    """
    Chooses a SolutionPoint from the pareto front
    :param solution_front: MOCOHDA SolutionCandidate with solution front
    :return: the chosen SolutionPoint
    """
    return solution_front.solution_points[0]
