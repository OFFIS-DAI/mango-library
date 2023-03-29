import pytest

from util_functions import create_three_agents, shutdown


@pytest.mark.asyncio
async def test_successful_negotiation_three_agents_demand():
    """
    Method to test simple negotiation with three agents and enough flexibility
    to solve the problem. The problem is smaller than 0 and negotiation starts with a DemandNotification.
    """
    agent_a, agent_b, agent_c, container = await create_three_agents()

    # first, hand flexibility to agents
    agent_a.update_flexibility(t_start=900, min_p=0, max_p=0)
    agent_b.update_flexibility(t_start=900, min_p=-30, max_p=-30)
    agent_c.update_flexibility(t_start=900, min_p=-10, max_p=0)

    # value is to negotiate is -50, so each of the agents need to take part in the negotiation to solve it
    await agent_b.start_negotiation(ts=[900, 1800], value=-50)
    await agent_b.negotiation_done
    # after the negotiation, the agents should have updated their flexibility
    print("Lösung!" + str(agent_b.final))
    assert agent_a.flex[900] == [0, 0]
    assert agent_b.flex[900] == [0, 0]
    assert agent_c.flex[900] == [0, 0]
    assert 'agent0' and 'agent2' in agent_b.final
    await shutdown([agent_a, agent_b, agent_c], [container])


@pytest.mark.asyncio
async def test_successful_negotiation_three_agents_offer():
    """
    Test negotiation with three agents with enough flexibility to solve the problem. The problem is greater than 0 and
    the negotiation starts with an Offer (Notification).
    """
    agent_a, agent_b, agent_c, container = await create_three_agents()
    agent_a.update_flexibility(t_start=900, min_p=0, max_p=10)
    agent_b.update_flexibility(t_start=900, min_p=0, max_p=30)
    agent_c.update_flexibility(t_start=900, min_p=0, max_p=10)

    await agent_b.start_negotiation(ts=[900, 1800], value=50)
    await agent_b.negotiation_done

    # after the negotiation, the agents should have updated their flexibility
    assert agent_a.flex[900] == [0, 0]
    assert agent_b.flex[900] == [0, 0]
    assert agent_c.flex[900] == [0, 0]
    assert 'agent0' and 'agent2' in agent_b.final
    await shutdown([agent_a, agent_b, agent_c], [container])


@pytest.mark.asyncio
async def test_negotiation_with_not_enough_flexibility():
    """
    Test negotiation with three agents and not enough flexibility to solve the problem.
    """
    agent_a, agent_b, agent_c, container = await create_three_agents()
    # not enough flexibility
    agent_a.update_flexibility(t_start=0, min_p=0, max_p=10)
    agent_a.update_flexibility(t_start=0, min_p=0, max_p=10)
    agent_b.update_flexibility(t_start=0, min_p=0, max_p=30)
    agent_c.update_flexibility(t_start=0, min_p=0, max_p=10)

    await agent_b.start_negotiation(ts=[0, 900], value=150)
    await agent_b.negotiation_done

    # after the negotiation, the agents should have updated their flexibility
    # the negotiation is not completely solved, but the agents fulfilled as much as possible
    assert agent_a.flex[0] == [0, 0]
    assert agent_b.flex[0] == [0, 0]
    assert agent_c.flex[0] == [0, 0]
    assert 'agent0' and 'agent2' in agent_b.final
    assert not agent_b._solution_found

    await shutdown([agent_a, agent_b, agent_c], [container])
