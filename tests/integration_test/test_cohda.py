
import pytest
from mango.core.container import Container
from mango.role.core import RoleAgent
import mango.messages.codecs
from mango_library.negotiation.cohda.cohda import *
from mango_library.negotiation.termination import NegotiationTerminationParticipantRole,\
    NegotiationTerminationDetectorRole
from mango_library.coalition.core import *
import mango_library.negotiation.util as util
import asyncio


@pytest.mark.asyncio
async def test_coalition_to_cohda_with_termination():
    # create container
    c = await Container.factory(addr=('127.0.0.3', 5555))
    s_array = [[[1, 1, 1, 1, 1], [4, 3, 3, 3, 3], [6, 6, 6, 6, 6], [9, 8, 8, 8, 8], [11, 11, 11, 11, 11]]]

    # create agents
    agents = []
    addrs = []
    controller_agent = RoleAgent(c)
    controller_agent.add_role(NegotiationTerminationDetectorRole())
    agents.append(controller_agent)
    for i in range(10):
        a = RoleAgent(c)
        cohda_role = COHDARole(lambda: s_array[0], lambda s: True)
        a.add_role(cohda_role)
        a.add_role(CoalitionParticipantRole())
        a.add_role(NegotiationTerminationParticipantRole())
        if i == 0:
            a.add_role(CohdaNegotiationStarterRole(([110, 110, 110, 110, 110], [1, 1, 1, 1, 1, ])))
        addrs.append((c.addr, a._aid))
        agents.append(a)

    agents[0].add_role(CoalitionInitiatorRole(addrs, 'cohda', 'cohda-negotiation'))

    await asyncio.wait_for(wait_for_coalition_built(agents), timeout=5)

    for a in agents:
        if a._check_inbox_task.done():
            if a._check_inbox_task.exception() is not None:
                raise a._check_inbox_task.exception()
            else:
                assert False, f'check_inbox terminated unexpectedly.'

    # await asyncio.wait_for(wait_for_term(agents), timeout=3)
    await asyncio.sleep(1)

    # gracefully shutdown
    for a in agents:
        await a.shutdown()
    await c.shutdown()

    assert len(asyncio.all_tasks()) == 1, f'Too many Tasks are running{asyncio.all_tasks()}'
    assert np.array_equal(next(iter(agents[1].roles[0]._cohda.values()))._memory.solution_candidate.schedules['1'],
                          [11, 11, 11, 11, 11])
    assert next(iter(agents[0].roles[0]._weight_map.values())) == 1


@pytest.mark.asyncio
async def test_coalition_to_cohda_with_termination_different_container():
    # create containers
    codec = mango.messages.codecs.JSON()
    codec2 = mango.messages.codecs.JSON()
    for serializer in util.cohda_serializers:
        codec.add_serializer(*serializer())
        codec2.add_serializer(*serializer())
    c_1 = await Container.factory(addr=('127.0.0.2', 5555), codec=codec)
    c_2 = await Container.factory(addr=('127.0.0.2', 5556), codec=codec2)

    s_array = [[[1, 1, 1, 1, 1], [4, 3, 3, 3, 3], [6, 6, 6, 6, 6], [9, 8, 8, 8, 8], [11, 11, 11, 11, 11]]]

    # create agents
    agents = []
    addrs = []
    for i in range(10):
        container = c_1 if i % 2 == 0 else c_2
        a = RoleAgent(container)
        cohda_role = COHDARole(lambda: np.array(s_array[0]), lambda s: True)
        a.add_role(cohda_role)
        a.add_role(CoalitionParticipantRole())
        a.add_role(NegotiationTerminationRole(i == 0))
        agents.append(a)
        addrs.append((container.addr, a._aid))

    agents[0].add_role(CoalitionInitiatorRole(addrs, 'cohda', 'cohda-negotiation'))

    await asyncio.wait_for(wait_for_coalition_built(agents), timeout=5)

    agents[0].add_role(CohdaNegotiationStarterRole(([110, 110, 110, 110, 110], [1, 1, 1, 1, 1, ])))

    for a in agents:
        if a._check_inbox_task.done():
            if a._check_inbox_task.exception() is not None:
                raise a._check_inbox_task.exception()
            else:
                assert False, f'check_inbox terminated unexpectedly.'

    await asyncio.wait_for(wait_for_term(agents), timeout=10)

    # gracefully shutdown
    for a in agents:
        await a.shutdown()
    await c_1.shutdown()
    await c_2.shutdown()

    assert len(asyncio.all_tasks()) == 1
    assert np.array_equal(next(iter(agents[0].roles[0]._cohda.values()))._memory.solution_candidate.schedules['1'],
                          [11, 11, 11, 11, 11])
    assert next(iter(agents[0].roles[2]._weight_map.values())) == 1


@pytest.mark.asyncio
async def test_coalition_to_cohda_with_termination_long_scenario():
    # create containers

    c = await Container.factory(addr=('127.0.0.2', 5555))
    s_array = [[1], [0]]
    n_agents = 40
    agents = []
    addrs = []

    # create agents
    for i in range(n_agents):
        a = RoleAgent(c)
        cohda_role = COHDARole(lambda: s_array)
        a.add_role(cohda_role)
        a.add_role(CoalitionParticipantRole())
        a.add_role(NegotiationTerminationRole(i == 0))
        agents.append(a)
        addrs.append((c.addr, a._aid))

    agents[0].add_role(CoalitionInitiatorRole(addrs, 'cohda', 'cohda-negotiation'))

    await asyncio.wait_for(wait_for_coalition_built(agents), timeout=5)

    agents[0].add_role(CohdaNegotiationStarterRole(
        ([n_agents//2], [1])
    ))

    for a in agents:
        if a._check_inbox_task.done():
            if a._check_inbox_task.exception() is not None:
                raise a._check_inbox_task.exception()
            else:
                assert False, f'check_inbox terminated unexpectedly.'

    await asyncio.wait_for(wait_for_term(agents), timeout=15)

    for agent in agents:
        if list(agent.roles[2]._weight_map.values())[0] != 0:
            print('Final weight:', agent.roles[2]._weight_map)

    # gracefully shutdown
    for a in agents:
        await a.shutdown()
    await c.shutdown()

    assert len(asyncio.all_tasks()) == 1
    assert np.array_equal(
        next(iter(agents[0].roles[0]._cohda.values()))._memory.solution_candidate.cluster_schedule.sum(axis=0),
        [n_agents//2])
    assert next(iter(agents[0].roles[2]._weight_map.values())) == 1


async def wait_for_coalition_built(agents):
    for agent in agents:
        while not agent.inbox.empty():
            await asyncio.sleep(0.1)


async def wait_for_term(agents):
    await asyncio.sleep(0.1)
    for agent in agents:
        while not agent.inbox.empty() or next(iter(agents[0].roles[2]._weight_map.values())) != 1:
            await asyncio.sleep(0.1)
