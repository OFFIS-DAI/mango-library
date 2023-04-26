import asyncio
import pytest
import uuid
import random

import mango.messages.codecs
from mango import create_container
from mango import RoleAgent
from mango_library.coalition.core import (
    CoalitionInvite,
    CoaltitionResponse,
    CoalitionAssignment,
    CoalitionInitiatorRole,
    CoalitionParticipantRole,
    CoalitionModel,
    small_world_creator,
    CoalitionAssignmentConfirm,
    CoalitionBuildConfirm,
)
from mango_library.negotiation.cohda.cohda_starting import (
    CohdaNegotiationDirectStarterRole,
)
from mango_library.negotiation.util import cohda_serializers


def test_serialize_coalition_invite():
    codec = mango.messages.codecs.JSON()
    for serializer in cohda_serializers:
        codec.add_serializer(*serializer())

    my_data = CoalitionInvite(
        coalition_id=uuid.uuid1(), topic="test_topic", details="Details"
    )

    encoded = codec.encode(my_data)
    decoded = codec.decode(encoded)

    assert my_data.coalition_id == decoded.coalition_id
    assert my_data.topic == decoded.topic
    assert my_data.details == decoded.details


def test_serialize_coalition_response():
    codec = mango.messages.codecs.JSON()
    for serializer in cohda_serializers:
        codec.add_serializer(*serializer())

    my_data = CoaltitionResponse(accept=True)
    my_data_2 = CoaltitionResponse(accept=False)

    encoded = codec.encode(my_data)
    decoded = codec.decode(encoded)
    assert my_data.accept == decoded.accept

    encoded = codec.encode(my_data_2)
    decoded = codec.decode(encoded)

    assert my_data_2.accept == decoded.accept


def test_serialize_coalition_assignment_confirm():
    codec = mango.messages.codecs.JSON()
    for serializer in cohda_serializers:
        codec.add_serializer(*serializer())
    coal_id = "12345678123456781234567812345678"
    my_data = CoalitionAssignmentConfirm(coalition_id=uuid.UUID(coal_id))

    encoded = codec.encode(my_data)
    decoded = codec.decode(encoded)
    assert my_data.coalition_id == decoded.coalition_id


def test_serialize_coalition_build_confirm():
    codec = mango.messages.codecs.JSON()
    for serializer in cohda_serializers:
        codec.add_serializer(*serializer())
    coal_id = "12345678123456781234567812345678"
    my_data = CoalitionBuildConfirm(coalition_id=uuid.UUID(coal_id))

    encoded = codec.encode(my_data)
    decoded = codec.decode(encoded)
    assert my_data.coalition_id == decoded.coalition_id


def test_serialize_coalition_assignment():
    codec = mango.messages.codecs.JSON()
    for serializer in cohda_serializers:
        codec.add_serializer(*serializer())

    my_data = CoalitionAssignment(
        coalition_id=uuid.uuid1(),
        part_id="12",
        neighbors=[
            ("1", ("127.0.0.2", 5555), "agent0"),
            ("2", ("127.0.0.2", 5556), "agent0"),
        ],
        controller_agent_id="agent_0",
        controller_agent_addr=("127.0.0.2", 5556),
        topic="test",
    )

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


def test_colition_initiator_with_str_as_addr():
    c_init_role = CoalitionInitiatorRole(
        participants=[
            ("agent_addr_0", "Agent0"),
            ("agent_addr_1", "Agent0"),
            (("localhost", 5555), "Agent0"),
        ],
        topic="",
        details="",
    )

    msg = CoaltitionResponse(accept=True)
    c_init_role.handle_coalition_response_msg(
        content=msg, meta={"sender_addr": "agent_addr_0", "sender_id": "Agent0"}
    )
    assert ("agent_addr_0", "Agent0") in c_init_role._part_to_state.keys()
    c_init_role.handle_coalition_response_msg(
        content=msg, meta={"sender_addr": ["localhost", 5555], "sender_id": "Agent0"}
    )
    assert (("localhost", 5555), "Agent0") in c_init_role._part_to_state.keys()


@pytest.mark.parametrize(
    "participants,expected,k",
    [
        (
                [("1", "1", "1"), ("2", "2", "2")],
                {("1", "1", "1"): [("2", "2", "2")], ("2", "2", "2"): [("1", "1", "1")]},
                1,
        ),
        (
                [("1", ("127.0.0.3", 5555), "1"), ("2", ("127.0.0.3", 5555), "2")],
                {
                    ("1", ("127.0.0.3", 5555), "1"): [("2", ("127.0.0.3", 5555), "2")],
                    ("2", ("127.0.0.3", 5555), "2"): [("1", ("127.0.0.3", 5555), "1")],
                },
                2,
        ),
        (["1", "2"], {"1": ["2"], "2": ["1"]}, 1),
        (
                ["1", "2", "3", "4"],
                {"1": ["4", "2"], "2": ["1", "3"], "3": ["2", "4"], "4": ["3", "1"]},
                1,
        ),
        (
                ["1", "2", "3", "4"],
                {
                    "1": ["4", "2", "3"],
                    "2": ["1", "3", "4"],
                    "3": ["2", "4", "1"],
                    "4": ["3", "1", "2"],
                },
                2,
        ),
    ],
)
def test_small_world_creator(participants, expected, k):
    assert small_world_creator(participants=participants, k=k) == expected


def test_small_world_creator_with_w():
    random.seed(42)
    participants = ["1", "2", "3", "4", "5", "6", "7", "8"]
    neighborhood = small_world_creator(participants=participants, k=1, w=0.5)
    max_len = 0
    for key, value in neighborhood.items():
        for participant in value:
            assert key in neighborhood[participant]
        assert len(value) >= 2
        max_len = max(max_len, len(value))

    # with this random seed we will get some extra connections
    assert max_len > 2


@pytest.mark.asyncio
@pytest.mark.parametrize("num_part", [1, 2, 3, 4, 5])
async def test_build_coalition(num_part):
    # create containers

    c = await create_container(addr=("127.0.0.2", 5555))

    # create agents
    agents = []
    addrs = []
    for _ in range(num_part):
        a = RoleAgent(c)
        a.add_role(CoalitionParticipantRole())
        agents.append(a)
        addrs.append((c.addr, a.aid))

    controller_agent = RoleAgent(c)
    controller_agent.add_role(
        CoalitionInitiatorRole(
            addrs,
            "cohda",
            "cohda-negotiation",
            topology_creator_kwargs={"k": 2, "w": 0.1},
        )
    )
    agents.append(controller_agent)

    # all agents send ping request to all agents (including themselves)

    for a in agents:
        if a._check_inbox_task.done():
            if a._check_inbox_task.exception() is not None:
                raise a._check_inbox_task.exception()
            else:
                assert False, f"check_inbox terminated unexpectedly."

    for a in agents:
        await a.tasks_complete()

    await asyncio.wait_for(wait_for_coalition_built(agents[0:num_part]), timeout=20)

    # gracefully shutdown
    for a in agents:
        await a.shutdown()
    await c.shutdown()

    for a in agents[0:num_part]:
        assignments = a.roles[0].context.get_or_create_model(CoalitionModel).assignments
        assert list(assignments.values())[0].coalition_id is not None
        assert list(assignments.values())[0].controller_agent_id == controller_agent.aid
        assert list(assignments.values())[0].controller_agent_addr == c.addr
        assert len(list(assignments.values())[0].neighbors) == num_part - 1


@pytest.mark.asyncio
@pytest.mark.parametrize("num_part", [1, 2, 3, 4, 5])
async def test_build_coalition_with_negotiation_starter(num_part):
    # create containers

    c = await create_container(addr=("127.0.0.2", 5556))

    # create agents
    agents = []
    addrs = []
    for _ in range(num_part):
        a = RoleAgent(c)
        a.add_role(CoalitionParticipantRole())
        agents.append(a)
        addrs.append((c.addr, a.aid))

    agents[0].add_role(CohdaNegotiationDirectStarterRole(target_params=None))
    controller_agent = RoleAgent(c)
    controller_agent.add_role(
        CoalitionInitiatorRole(
            addrs,
            "cohda",
            "cohda-negotiation",
            topology_creator_kwargs={"k": 2, "w": 0.1},
        )
    )
    agents.append(controller_agent)

    # all agents send ping request to all agents (including themselves)

    for a in agents:
        if a._check_inbox_task.done():
            if a._check_inbox_task.exception() is not None:
                raise a._check_inbox_task.exception()
            else:
                assert False, f"check_inbox terminated unexpectedly."

    await asyncio.wait_for(wait_for_coalition_built(agents[0:num_part]), timeout=10)
    # If the coalition build was successful and all assignments were confirmed, the CoalitionInitiator informs the
    # other agents about it. The agent with the NegotiationStarterRole expects this message and stores the ID of
    # the successful coalition.
    # It takes some time until the last message was sent after the coalition was build.
    while len(agents[0].roles[1]._coalitions) < 1:
        await asyncio.sleep(1)

    # When the CoalitionInitiator received all confirmations regarding the assignments, the future is done for each
    # agent in the list with expected assignment confirms.
    for agent_addr, fut in controller_agent.roles[0]._assignments_confirmed.items():
        assert fut.done()

    # The coalition ID stored by the agent with the NegotiationStarterRole equals the coalition ID
    # of the CoalitionInitiator
    assert agents[0].roles[1]._coalitions[0] == controller_agent.roles[0]._coal_id
    # gracefully shutdown
    for a in agents:
        await a.shutdown()
    await c.shutdown()

    for a in agents[0:num_part]:
        assignments = a.roles[0].context.get_or_create_model(CoalitionModel).assignments
        assert list(assignments.values())[0].coalition_id is not None
        assert list(assignments.values())[0].controller_agent_id == controller_agent.aid
        assert list(assignments.values())[0].controller_agent_addr == c.addr
        assert len(list(assignments.values())[0].neighbors) == num_part - 1


async def wait_for_coalition_built(agents):
    for agent in agents:
        while (
                len(agent.roles[0].context.get_or_create_model(CoalitionModel).assignments)
                == 0
        ):
            await asyncio.sleep(0.1)
