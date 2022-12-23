import asyncio
import pytest
import numpy as np
from mango import Container
from mango import RoleAgent
import mango.messages.codecs
from mango_library.negotiation.cohda.cohda_negotiation import COHDANegotiationRole, CohdaNegotiationModel, \
    CohdaSolutionModel
from mango_library.negotiation.cohda.cohda_solution_aggregation import CohdaSolutionAggregationRole
from mango_library.negotiation.cohda.cohda_starting import CohdaNegotiationStarterRole
from mango_library.negotiation.termination import NegotiationTerminationParticipantRole,\
    NegotiationTerminationDetectorRole
from mango_library.coalition.core import CoalitionParticipantRole, CoalitionInitiatorRole
import mango_library.negotiation.util as util


@pytest.mark.asyncio
async def test_coalition_to_cohda_with_termination():
    # create container
    c = await Container.factory(addr=('127.0.0.3', 5555))
    s_array = [[[1, 1, 1, 1, 1], [4, 3, 3, 3, 3], [6, 6, 6, 6, 6], [9, 8, 8, 8, 8], [11, 11, 11, 11, 11]]]

    # create cohda_agents
    cohda_agents = []
    addrs = []
    controller_agent = RoleAgent(c)
    controller_agent.add_role(NegotiationTerminationDetectorRole(aggregator_addr=c.addr,
                                                                 aggregator_id=controller_agent.aid))
    aggregation_role = CohdaSolutionAggregationRole()
    controller_agent.add_role(aggregation_role)

    for i in range(10):
        a = RoleAgent(c)
        def schedules_provider(candidate):
            print('This is the candidate', candidate)
            return s_array[0]
        cohda_role = COHDANegotiationRole(schedules_provider=schedules_provider,
                                          local_acceptable_func=lambda s: True)
        a.add_role(cohda_role)
        a.add_role(CoalitionParticipantRole())
        a.add_role(NegotiationTerminationParticipantRole())
        if i == 0:
            a.add_role(CohdaNegotiationStarterRole(([110, 110, 110, 110, 110], [1, 1, 1, 1, 1, ])))
        addrs.append((c.addr, a._aid))
        cohda_agents.append(a)

    controller_agent.add_role(CoalitionInitiatorRole(addrs, 'cohda', 'cohda-negotiation'))

    for a in cohda_agents + [controller_agent]:
        if a._check_inbox_task.done():
            if a._check_inbox_task.exception() is not None:
                raise a._check_inbox_task.exception()
            else:
                assert False, f'check_inbox terminated unexpectedly.'

    await asyncio.wait_for(wait_for_solution_confirmed(aggregation_role), timeout=5)

    # gracefully shutdown
    for a in cohda_agents + [controller_agent]:
        await a.shutdown()
    await c.shutdown()

    assert len(asyncio.all_tasks()) == 1, f'Too many Tasks are running{asyncio.all_tasks()}'
    cohda_negotiation = \
        list(cohda_agents[0]._agent_context.get_or_create_model(CohdaNegotiationModel)._negotiations.values())[0]
    cluster_schedule = cohda_negotiation._memory.solution_candidate.cluster_schedule
    for a in cohda_agents:
        assert np.array_equal(get_final_schedule(a), [11, 11, 11, 11, 11])
    assert np.array_equal(cluster_schedule[0],
                          [11, 11, 11, 11, 11])
    assert next(iter(controller_agent.roles[0]._weight_map.values())) == 1


@pytest.mark.asyncio
async def test_coalition_to_cohda_with_termination_different_container():
    # create containers
    codec = mango.messages.codecs.JSON()
    codec2 = mango.messages.codecs.JSON()
    for serializer in util.cohda_serializers:
        codec.add_serializer(*serializer())
        codec2.add_serializer(*serializer())
    c_1 = await Container.factory(addr=('127.0.0.3', 5555), codec=codec)
    c_2 = await Container.factory(addr=('127.0.0.3', 5556), codec=codec2)

    s_array = [[[1, 1, 1, 1, 1], [4, 3, 3, 3, 3], [6, 6, 6, 6, 6], [9, 8, 8, 8, 8], [11, 11, 11, 11, 11]]]

    # create cohda_agents
    cohda_agents = []
    addrs = []
    controller_agent = RoleAgent(c_1)
    controller_agent.add_role(NegotiationTerminationDetectorRole())
    aggregation_role = CohdaSolutionAggregationRole()
    controller_agent.add_role(aggregation_role)

    for i in range(5):
        c = c_2 if i % 2 == 0 else c_1
        a = RoleAgent(c)
        cohda_role = COHDANegotiationRole(lambda: s_array[0], lambda s: True)
        a.add_role(cohda_role)
        a.add_role(CoalitionParticipantRole())
        a.add_role(NegotiationTerminationParticipantRole())
        if i == 0:
            a.add_role(CohdaNegotiationStarterRole(([110, 110, 110, 110, 110], [1, 1, 1, 1, 1, ])))
        addrs.append((c.addr, a._aid))
        cohda_agents.append(a)

    controller_agent.add_role(CoalitionInitiatorRole(addrs, 'cohda', 'cohda-negotiation'))

    for a in cohda_agents + [controller_agent]:
        if a._check_inbox_task.done():
            if a._check_inbox_task.exception() is not None:
                raise a._check_inbox_task.exception()
            else:
                assert False, f'check_inbox terminated unexpectedly.'

    await asyncio.wait_for(wait_for_solution_confirmed(aggregation_role), timeout=5)

    # gracefully shutdown
    for a in cohda_agents + [controller_agent]:
        await a.shutdown()
    await c_1.shutdown()
    await c_2.shutdown()

    assert len(asyncio.all_tasks()) == 1, f'Too many Tasks are running{asyncio.all_tasks()}'
    cohda_negotiation = \
        list(cohda_agents[1]._agent_context.get_or_create_model(CohdaNegotiationModel)._negotiations.values())[0]
    cluster_schedule = cohda_negotiation._memory.solution_candidate.cluster_schedule
    assert np.array_equal(cluster_schedule[0],
                          [11, 11, 11, 11, 11])
    for a in cohda_agents:
        assert np.array_equal(get_final_schedule(a), [11, 11, 11, 11, 11])
    assert next(iter(controller_agent.roles[0]._weight_map.values())) == 1


@pytest.mark.asyncio
async def test_coalition_to_cohda_with_termination_long_scenario():
    # create containers
    c = await Container.factory(addr=('127.0.0.2', 5555))
    controller_agent = RoleAgent(c)
    controller_agent.add_role(NegotiationTerminationDetectorRole())
    aggregation_role = CohdaSolutionAggregationRole()
    controller_agent.add_role(aggregation_role)

    s_array = [[1], [0]]
    n_agents = 40
    cohda_agents = []
    addrs = []

    # create cohda_agents
    for i in range(n_agents):
        a = RoleAgent(c)
        cohda_role = COHDANegotiationRole(lambda: s_array)
        a.add_role(cohda_role)
        a.add_role(CoalitionParticipantRole())
        a.add_role(NegotiationTerminationParticipantRole())
        cohda_agents.append(a)
        addrs.append((c.addr, a._aid))

    controller_agent.add_role(CoalitionInitiatorRole(addrs, 'cohda', 'cohda-negotiation'))
    cohda_agents[0].add_role(CohdaNegotiationStarterRole(([n_agents//2], [1])))

    for a in cohda_agents:
        if a._check_inbox_task.done():
            if a._check_inbox_task.exception() is not None:
                raise a._check_inbox_task.exception()
            else:
                assert False, f'check_inbox terminated unexpectedly.'

    await asyncio.wait_for(wait_for_solution_confirmed(aggregation_role), timeout=25)

    for agent in cohda_agents:
        if list(agent.roles[2]._weight_map.values())[0] != 0:
            print('Final weight:', agent.roles[2]._weight_map)

    # gracefully shutdown
    for a in cohda_agents:
        await a.shutdown()
    await c.shutdown()

    assert len(asyncio.all_tasks()) == 1
    cohda_negotiation = \
        list(cohda_agents[0]._agent_context.get_or_create_model(CohdaNegotiationModel)._negotiations.values())[0]
    final_candidate = cohda_negotiation._memory.solution_candidate

    assert np.array_equal(
        final_candidate.cluster_schedule.sum(axis=0), [n_agents//2])
    for a in cohda_agents:
        # get part_id
        part_id = list(a._agent_context.get_or_create_model(CohdaNegotiationModel)._negotiations.values())[0]._part_id
        assert np.array_equal(get_final_schedule(a), final_candidate.schedules[part_id])
    assert next(iter(controller_agent.roles[0]._weight_map.values())) == 1


async def wait_for_solution_confirmed(aggregation_role):
    while len(aggregation_role._confirmed_cohda_solutions) == 0:
        await asyncio.sleep(0.05)


def get_final_schedule(cohda_agent):
    return list(cohda_agent._agent_context.get_or_create_model(CohdaSolutionModel)._final_schedules.values())[0]
