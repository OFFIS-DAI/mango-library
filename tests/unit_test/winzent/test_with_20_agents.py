import pytest

from tests.unit_test.winzent.util_functions import create_agents, shutdown


@pytest.mark.asyncio
async def test_negotiations_twenty_agents():
    """
    Test negotiation with Winzent with 20 agents. In this case, the agents have enough flexibility
    to solve the problem successfully.
    """
    number_of_agents = 20
    # the topology of the agents is a simple ring topology, therefore it needs a time to live of 19 to make
    # sure one message can be forwarded through the complete network
    ttl = 19
    agents, container = await create_agents(number_of_agents=number_of_agents, ttl=ttl, time_to_sleep=20)

    for agent in agents:
        agent.update_flexibility(t_start=0, min_p=0, max_p=10)

    await agents[1].start_negotiation(ts=[0, 900], value=200)
    await agents[1].negotiation_done

    # to solve the negotiation, all agents need to take part of it. The agent does not store its own flexibility
    # as part of the final dictionary, since it already calculates its own share before sending the request.
    # Therefore, the solution must contain number_of_agents - 1  entries.
    assert len(agents[1].final) == number_of_agents - 1

    await shutdown(agents, [container])


@pytest.mark.asyncio
async def test_negotiations_twenty_agents_multiple_intervals_successively():
    """
    Test negotiation with Winzent with 20 agents. In this case, the agents have enough flexibility
    to solve the problem successfully. Two different agents start to negotiate for different intervals successively.
    """
    number_of_agents = 20
    # the topology of the agents is a simple ring topology, therefore it needs a time to live of 19 to make
    # sure one message can be forwarded through the complete network
    ttl = 19
    agents, container = await create_agents(number_of_agents=number_of_agents, ttl=ttl, time_to_sleep=40)

    for agent in agents:
        agent.update_flexibility(t_start=0, min_p=0, max_p=10)
        agent.update_flexibility(t_start=900, min_p=0, max_p=10)

    await agents[1].start_negotiation(ts=[0, 900], value=200)

    await agents[1].negotiation_done
    await agents[10].start_negotiation(ts=[900, 1800], value=200)
    await agents[10].negotiation_done

    # to solve the negotiation, all agents need to take part of it. The agent does not store its own flexibility
    # as part of the result dictionary, since it already calculates its own share before sending the request.
    # Therefore, the solution must contain number_of_agents - 1  entries.
    assert len(agents[1].final) == number_of_agents - 1

    assert len(agents[10].final) == number_of_agents - 1

    # each agent needs to have updated their flexibility for both intervals
    for agent in agents:
        assert agent.flex[0] == [0, 0]
        assert agent.flex[900] == [0, 0]

    await shutdown(agents, [container])


@pytest.mark.asyncio
async def test_negotiations_twenty_agents_multiple_intervals_at_the_same_time():
    """
    Test negotiation with Winzent with 20 agents. In this case, the agents have enough flexibility
    to solve the problem successfully. Two different agents start to negotiate for different intervals at the same time.
    """
    number_of_agents = 20
    # the topology of the agents is a simple ring topology, therefore it needs a time to live from 19 to make
    # sure one message can be forwarded through the complete network
    ttl = 19
    agents, container = await create_agents(number_of_agents=number_of_agents, ttl=ttl, time_to_sleep=35)

    for agent in agents:
        agent.update_flexibility(t_start=0, min_p=0, max_p=10)
        agent.update_flexibility(t_start=900, min_p=0, max_p=10)

    await agents[1].start_negotiation(ts=[0, 900], value=200)
    await agents[10].start_negotiation(ts=[900, 1800], value=200)

    await agents[1].negotiation_done
    await agents[10].negotiation_done

    # To solve the negotiations, all agents needs to take part of it. The agent does not store its own flexibility
    # as part of the final dictionary, since it already calculates its own share before sending the request.
    # Therefore, the solution must contain number_of_agents - 1  entries.
    assert len(agents[1].final) == number_of_agents - 1

    assert len(agents[10].final) == number_of_agents - 1

    # each agent needs to have updated their flexibility for both intervals
    for agent in agents:
        assert agent.flex[0] == [0, 0]
        assert agent.flex[900] == [0, 0]

    await shutdown(agents, [container])
