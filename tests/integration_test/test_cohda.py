import asyncio
import pytest
import numpy as np
from mango import create_tcp_container, AgentAddress
from mango import RoleAgent
import mango.messages.codecs
from mango import activate

from mango_library.negotiation.cohda.cohda_negotiation import (
    COHDANegotiationRole,
    CohdaNegotiationModel,
    CohdaSolutionModel,
)
from mango_library.negotiation.cohda.cohda_solution_aggregation import (
    CohdaSolutionAggregationRole,
)
from mango_library.negotiation.cohda.cohda_starting import (
    CohdaNegotiationDirectStarterRole,
)
from mango_library.negotiation.termination import (
    NegotiationTerminationParticipantRole,
    NegotiationTerminationDetectorRole,
)
from mango_library.coalition.core import (
    CoalitionParticipantRole,
    CoalitionInitiatorRole,
)
import mango_library.negotiation.util as util
from tests.unit_test.cohda.coalition_test import wait_for_coalition_built


@pytest.mark.asyncio
async def test_coalition_to_cohda_with_termination():
    # create container
    c = create_tcp_container(addr=("127.0.0.3", 5555))
    s_array = [
        [
            [1, 1, 1, 1, 1],
            [4, 3, 3, 3, 3],
            [6, 6, 6, 6, 6],
            [9, 8, 8, 8, 8],
            [11, 11, 11, 11, 11],
        ]
    ]

    # create cohda_agents
    cohda_agents = []
    addrs = []
    controller_agent = c.register(RoleAgent())
    controller_agent.add_role(
        NegotiationTerminationDetectorRole(
            aggregator_addr=AgentAddress(c.addr, controller_agent.aid)
        )
    )
    aggregation_role = CohdaSolutionAggregationRole()
    controller_agent.add_role(aggregation_role)

    for i in range(10):
        a = c.register(RoleAgent())

        def schedules_provider(candidate):
            return s_array[0]

        cohda_role = COHDANegotiationRole(
            schedules_provider=schedules_provider, local_acceptable_func=lambda s: True
        )
        a.add_role(cohda_role)
        a.add_role(CoalitionParticipantRole())
        a.add_role(NegotiationTerminationParticipantRole())

        if i == 0:
            a.add_role(
                CohdaNegotiationDirectStarterRole(
                    (
                        [110, 110, 110, 110, 110],
                        [
                            1,
                            1,
                            1,
                            1,
                            1,
                        ],
                    )
                )
            )
        addrs.append(AgentAddress(c.addr, a.aid))
        cohda_agents.append(a)

    async with activate(c):
        controller_agent.add_role(
            CoalitionInitiatorRole(addrs, "cohda", "cohda-negotiation")
        )

        for a in cohda_agents + [controller_agent]:
            if a._check_inbox_task.done():
                if a._check_inbox_task.exception() is not None:
                    raise a._check_inbox_task.exception()
                else:
                    assert False, f"check_inbox terminated unexpectedly."
        await asyncio.wait_for(wait_for_coalition_built(cohda_agents), timeout=10)
        await asyncio.wait_for(wait_for_solution_confirmed(aggregation_role), timeout=5)

    assert (
            len(asyncio.all_tasks()) == 1
    ), f"Too many Tasks are running{asyncio.all_tasks()}"
    cohda_negotiation = list(
        cohda_agents[0]
        ._role_context.get_or_create_model(CohdaNegotiationModel)
        ._negotiations.values()
    )[0]
    cluster_schedule = cohda_negotiation._memory.solution_candidate.cluster_schedule
    for a in cohda_agents:
        assert np.array_equal(get_final_schedule(a), [11, 11, 11, 11, 11])
    assert np.array_equal(cluster_schedule[0], [11, 11, 11, 11, 11])
    assert next(iter(controller_agent.roles[0]._weight_map.values())) == 1


@pytest.mark.asyncio
async def test_coalition_to_cohda_with_termination_different_container():
    # create containers
    codec = mango.messages.codecs.JSON()
    codec2 = mango.messages.codecs.JSON()
    for serializer in util.cohda_serializers:
        codec.add_serializer(*serializer())
        codec2.add_serializer(*serializer())
    c_1 = create_tcp_container(addr=("127.0.0.3", 5555), codec=codec)
    c_2 = create_tcp_container(addr=("127.0.0.3", 5556), codec=codec2)

    s_array = [
        [
            [1, 1, 1, 1, 1],
            [4, 3, 3, 3, 3],
            [6, 6, 6, 6, 6],
            [9, 8, 8, 8, 8],
            [11, 11, 11, 11, 11],
        ]
    ]

    # create cohda_agents
    cohda_agents = []
    addrs = []
    controller_agent = c_1.register(RoleAgent())
    controller_agent.add_role(NegotiationTerminationDetectorRole())
    aggregation_role = CohdaSolutionAggregationRole()
    controller_agent.add_role(aggregation_role)

    for i in range(5):
        c = c_2 if i % 2 == 0 else c_1
        a = c.register(RoleAgent())
        cohda_role = COHDANegotiationRole(lambda: s_array[0], lambda s: True)
        a.add_role(cohda_role)
        a.add_role(CoalitionParticipantRole())
        a.add_role(NegotiationTerminationParticipantRole())
        if i == 0:
            a.add_role(
                CohdaNegotiationDirectStarterRole(
                    (
                        [110, 110, 110, 110, 110],
                        [
                            1,
                            1,
                            1,
                            1,
                            1,
                        ],
                    )
                )
            )
        addrs.append(AgentAddress(c.addr, a.aid))
        cohda_agents.append(a)

    async with activate([c_1, c_2]):
        controller_agent.add_role(
            CoalitionInitiatorRole(addrs, "cohda", "cohda-negotiation")
        )

        for a in cohda_agents + [controller_agent]:
            if a._check_inbox_task.done():
                if a._check_inbox_task.exception() is not None:
                    raise a._check_inbox_task.exception()
                else:
                    assert False, f"check_inbox terminated unexpectedly."

        await asyncio.wait_for(wait_for_coalition_built(cohda_agents), timeout=10)
        await asyncio.wait_for(wait_for_solution_confirmed(aggregation_role), timeout=10)

    assert (
            len(asyncio.all_tasks()) == 1
    ), f"Too many Tasks are running{asyncio.all_tasks()}"
    cohda_negotiation = list(
        cohda_agents[1]
        ._role_context.get_or_create_model(CohdaNegotiationModel)
        ._negotiations.values()
    )[0]
    cluster_schedule = cohda_negotiation._memory.solution_candidate.cluster_schedule
    assert np.array_equal(cluster_schedule[0], [11, 11, 11, 11, 11])
    for a in cohda_agents:
        assert np.array_equal(get_final_schedule(a), [11, 11, 11, 11, 11])
    assert next(iter(controller_agent.roles[0]._weight_map.values())) == 1


@pytest.mark.asyncio
async def test_coalition_to_cohda_with_termination_long_scenario():
    # create containers
    c = create_tcp_container(addr=("127.0.0.2", 5555))
    controller_agent = c.register(RoleAgent())
    controller_agent.add_role(NegotiationTerminationDetectorRole())
    aggregation_role = CohdaSolutionAggregationRole()
    controller_agent.add_role(aggregation_role)

    s_array = [[1], [0]]
    n_agents = 40
    cohda_agents = []
    addrs = []

    # create cohda_agents
    for i in range(n_agents):
        a = c.register(RoleAgent())
        cohda_role = COHDANegotiationRole(lambda: s_array)
        a.add_role(cohda_role)
        a.add_role(CoalitionParticipantRole())
        a.add_role(NegotiationTerminationParticipantRole())
        cohda_agents.append(a)
        addrs.append(AgentAddress(c.addr, a.aid))
    async with activate(c):
        controller_agent.add_role(
            CoalitionInitiatorRole(addrs, "cohda", "cohda-negotiation")
        )
        cohda_agents[0].add_role(CohdaNegotiationDirectStarterRole(([n_agents // 2], [1])))

        for a in cohda_agents:
            if a._check_inbox_task.done():
                if a._check_inbox_task.exception() is not None:
                    raise a._check_inbox_task.exception()
                else:
                    assert False, f"check_inbox terminated unexpectedly."

        await asyncio.wait_for(wait_for_solution_confirmed(aggregation_role), timeout=30)

    for agent in cohda_agents:
        if list(agent.roles[2]._weight_map.values())[0] != 0:
            print("Final weight:", agent.roles[2]._weight_map)

    assert len(asyncio.all_tasks()) == 1
    cohda_negotiation = list(
        cohda_agents[0]
        ._role_context.get_or_create_model(CohdaNegotiationModel)
        ._negotiations.values()
    )[0]
    final_candidate = cohda_negotiation._memory.solution_candidate

    assert np.array_equal(final_candidate.cluster_schedule.sum(axis=0), [n_agents // 2])
    for a in cohda_agents:
        # get part_id
        part_id = list(
            a._role_context.get_or_create_model(
                CohdaNegotiationModel
            )._negotiations.values()
        )[0]._part_id
        assert np.array_equal(get_final_schedule(a), final_candidate.schedules[part_id])
    assert next(iter(controller_agent.roles[0]._weight_map.values())) == 1


async def wait_for_solution_confirmed(aggregation_role):
    while len(aggregation_role._confirmed_cohda_solutions) == 0:
        await asyncio.sleep(0.05)


def get_final_schedule(cohda_agent):
    return list(
        cohda_agent._role_context.get_or_create_model(
            CohdaSolutionModel
        )._final_schedules.values()
    )[0]
