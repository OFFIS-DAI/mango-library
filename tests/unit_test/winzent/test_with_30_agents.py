import pytest

from tests.unit_test.winzent.util_functions import create_agents, shutdown


@pytest.mark.asyncio
async def test_successful_negotiation_thirty_agents():
    """
    Test negotiation with Winzent with 30 agents. In this case, the agents have enough flexibility
    to solve the problem successfully.
    """
    number_of_agents = 30
    # the topology of the agents is a simple ring topology, therefore it needs a time to live from number_of_agents -1
    # to make sure one message can be forwarded through the complete network
    ttl = 29
    agents, container = await create_agents(number_of_agents=number_of_agents, ttl=ttl, time_to_sleep=25)

    for agent in agents:
        agent.update_flexibility(t_start=0, min_p=0, max_p=10)

    await agents[0].start_negotiation(ts=[0, 900], value=100)
    await agents[15].start_negotiation(ts=[0, 900], value=100)

    await agents[0].negotiation_done
    await agents[15].negotiation_done

    # in this case, the agents were far enough apart, so that their different neighbors could afford enough
    # flexibility without offering flexibility to another agent
    # For both agents to solve the negotiation, for each solution, 10 - 1 (the requesting agent) agents are necessary
    assert len(agents[0].final.keys()) == 9
    assert len(agents[15].final.keys()) == 9

    # To solve each problem, every agent can only be part of one solution. The agents in the result of the
    # first negotiation, can not be part of the second negotiation
    for agent in agents[0].final:
        assert agent not in agents[15].final.keys()

    await shutdown(agents, [container])


@pytest.mark.asyncio
async def test_successful_negotiation_thirty_agents_with_withdrawals():
    """
    Test negotiation with Winzent with 30 agents. In this case, the agents have enough flexibility
    to solve the problem successfully.
    """
    number_of_agents = 30
    # the topology of the agents is a simple ring topology, therefore it needs a time to live from number_of_agents -1
    # to make sure one message can be forwarded through the complete network
    ttl = 29
    agents, container = await create_agents(number_of_agents=number_of_agents, ttl=ttl, time_to_sleep=15)

    for agent in agents:
        agent.update_flexibility(t_start=0, min_p=0, max_p=10)

    await agents[0].start_negotiation(ts=[0, 900], value=160)
    await agents[15].start_negotiation(ts=[0, 900], value=160)

    await agents[0].negotiation_done
    await agents[15].negotiation_done

    # In this case, there was not enough flexibility for both agents to solve the problem and at least one agent
    # has an invalid negotiation
    assert any([len(agents[0]._unsuccessful_negotiations) > 0, len(agents[15]._unsuccessful_negotiations) > 0])

    await shutdown(agents, [container])


@pytest.mark.asyncio
async def test_successful_negotiation_thirty_agents_the_first_5():
    """
    Test negotiation with Winzent with 30 agents. In this case, the agents have enough flexibility
    to solve the problem successfully.
    """

    number_of_agents = 30
    # the topology of the agents is a simple ring topology, therefore it needs a time to live from number_of_agents -1
    # to make sure one message can be forwarded through the complete network
    ttl = 29
    agents, container = await create_agents(number_of_agents=number_of_agents, ttl=ttl, time_to_sleep=25)

    for agent in agents:
        agent.update_flexibility(t_start=0, min_p=0, max_p=10)

    await agents[0].start_negotiation(ts=[0, 900], value=15)
    await agents[5].start_negotiation(ts=[0, 900], value=15)
    await agents[10].start_negotiation(ts=[0, 900], value=15)
    await agents[15].start_negotiation(ts=[0, 900], value=15)
    await agents[25].start_negotiation(ts=[0, 900], value=15)

    await agents[0].negotiation_done
    await agents[5].negotiation_done
    await agents[10].negotiation_done
    await agents[15].negotiation_done
    await agents[25].negotiation_done

    # In this case, there was not enough flexibility for both agents to solve the problem and at least one agent
    # has an invalid negotiation
    assert any([len(agents[0]._unsuccessful_negotiations) == 0,
                len(agents[1]._unsuccessful_negotiations) == 0,
                len(agents[2]._unsuccessful_negotiations) == 0,
                len(agents[3]._unsuccessful_negotiations) == 0,
                len(agents[4]._unsuccessful_negotiations) == 0])

    await shutdown(agents, [container])
