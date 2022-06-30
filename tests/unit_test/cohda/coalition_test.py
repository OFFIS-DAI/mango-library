import mango.messages.codecs
import pytest
from mango.core.container import Container
from mango.role.core import RoleAgent
from mango_library.coalition.core import *
from mango_library.negotiation.util import extra_serializers


def test_serialize_coalition_invite():
    codec = mango.messages.codecs.JSON()
    for serializer in extra_serializers:
        codec.add_serializer(*serializer())

    my_data = CoalitionInvite(coalition_id=uuid.uuid1(), topic='test_topic', details='Details')

    encoded = codec.encode(my_data)
    decoded = codec.decode(encoded)

    assert my_data.coalition_id == decoded.coalition_id
    assert my_data.topic == decoded.topic
    assert my_data.details == decoded.details


def test_serialize_coalition_response():
    codec = mango.messages.codecs.JSON()
    for serializer in extra_serializers:
        codec.add_serializer(*serializer())

    my_data = CoaltitionResponse(accept=True)
    my_data_2 = CoaltitionResponse(accept=False)

    encoded = codec.encode(my_data)
    decoded = codec.decode(encoded)
    assert my_data.accept == decoded.accept

    encoded = codec.encode(my_data_2)
    decoded = codec.decode(encoded)

    assert my_data_2.accept == decoded.accept


def test_serialize_coalition_assignment():
    codec = mango.messages.codecs.JSON()
    for serializer in extra_serializers:
        codec.add_serializer(*serializer())

    my_data = CoalitionAssignment(coalition_id=uuid.uuid1(), part_id='12',
                                  neighbors=[('1', ('127.0.0.2', 5555), 'agent0'),
                                             ('2', ('127.0.0.2', 5556), 'agent0')],
                                  controller_agent_id='agent_0', controller_agent_addr=('127.0.0.2', 5556),
                                  topic='test')

    encoded = codec.encode(my_data)
    decoded = codec.decode(encoded)
    assert my_data.coalition_id == decoded.coalition_id
    assert my_data.part_id == decoded.part_id
    for neighbor_1, neighbor_2 in zip(my_data.neighbors, decoded.neighbors):
        assert neighbor_1[0] == neighbor_2[0]
        assert neighbor_1[1][0] == neighbor_2[1][0]
        assert neighbor_1[1][1] == neighbor_2[1][1]
        assert neighbor_1[2] == neighbor_2[2]
    assert my_data.controller_agent_id == decoded.controller_agent_id
    assert my_data.controller_agent_addr[0] == decoded.controller_agent_addr[0]
    assert my_data.controller_agent_addr[1] == decoded.controller_agent_addr[1]
    assert my_data.topic == decoded.topic


@pytest.mark.asyncio
@pytest.mark.parametrize("num_part", [1, 2, 3, 4, 5])
async def test_build_coalition(num_part):
    # create containers
    
    c = await Container.factory(addr=('127.0.0.2', 5555))
    
    # create agents
    agents = []
    addrs = []
    for _ in range(num_part):
        a = RoleAgent(c)
        a.add_role(CoalitionParticipantRole())
        agents.append(a)
        addrs.append((c.addr, a._aid))

    controller_agent = RoleAgent(c)
    controller_agent.add_role(CoalitionInitiatorRole(addrs, 'cohda', 'cohda-negotiation'))
    agents.append(controller_agent)

    # all agents send ping request to all agents (including themselves)

    for a in agents:
        if a._check_inbox_task.done():
            if a._check_inbox_task.exception() is not None:
                raise a._check_inbox_task.exception()
            else:
                assert False, f'check_inbox terminated unexpectedly.'
    
    for a in agents:
        await a.tasks_complete()

    await asyncio.wait_for(wait_for_coalition_built(agents[0:num_part]), timeout=5)

    # gracefully shutdown
    for a in agents:
        await a.shutdown()
    await c.shutdown()

    assert len(asyncio.all_tasks()) == 1
    for a in agents[0:num_part]:
        assignments = a.roles[0].context.get_or_create_model(CoalitionModel).assignments
        assert list(assignments.values())[0].coalition_id is not None
        assert list(assignments.values())[0].controller_agent_id == controller_agent.aid
        assert list(assignments.values())[0].controller_agent_addr == c.addr
        assert len(list(assignments.values())[0].neighbors) == num_part-1


async def wait_for_coalition_built(agents):
    for agent in agents:
        while len(agent.roles[0].context.get_or_create_model(CoalitionModel).assignments) == 0:
            await asyncio.sleep(0.1)
