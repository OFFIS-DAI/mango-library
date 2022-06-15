import pytest

from tests.unit_test.winzent.util_functions import create_six_agents, shutdown


@pytest.mark.asyncio
async def test_negotiations_six_agents():
    """
    Test negotiation with Winzent with six agents. In this case, the agents have enough flexibility
    to solve the problem successfully.
    """
    agent_a, agent_b, agent_c, agent_d, agent_e, agent_f, container = await create_six_agents()
    agent_a.update_flexibility(t_start=0, min_p=0, max_p=10)
    agent_b.update_flexibility(t_start=0, min_p=0, max_p=30)
    agent_c.update_flexibility(t_start=0, min_p=0, max_p=10)
    agent_d.update_flexibility(t_start=0, min_p=0, max_p=20)
    agent_e.update_flexibility(t_start=0, min_p=0, max_p=20)
    agent_f.update_flexibility(t_start=0, min_p=0, max_p=20)

    await agent_b.start_negotiation(ts=[0, 900], value=110)
    await agent_b.negotiation_done
    # after the negotiation, the agents should have updated their flexibility
    assert agent_a.flex[0] == [0, 0]
    assert agent_b.flex[0] == [0, 0]
    assert agent_c.flex[0] == [0, 0]
    assert agent_d.flex[0] == [0, 0]
    assert agent_e.flex[0] == [0, 0]
    assert agent_f.flex[0] == [0, 0]
    assert 'agent0' and 'agent2' and 'agent3' and 'agent4' and 'agent5' in agent_b.final

    await shutdown([agent_a, agent_b, agent_c, agent_d, agent_e, agent_f], [container])


@pytest.mark.asyncio
async def test_successful_negotiation_multiple_agents_multiple_intervals():
    """
    Test negotiation with six agents for several intervals in succession. For both intervals, the agents
    have enough flexibility to finish the negotiations successfully.
    """
    agent_a, agent_b, agent_c, agent_d, agent_e, agent_f, container = await create_six_agents()
    agent_a.update_flexibility(t_start=0, min_p=0, max_p=10)
    agent_b.update_flexibility(t_start=0, min_p=0, max_p=30)
    agent_c.update_flexibility(t_start=0, min_p=0, max_p=10)
    agent_d.update_flexibility(t_start=0, min_p=0, max_p=20)
    agent_e.update_flexibility(t_start=0, min_p=0, max_p=20)
    agent_f.update_flexibility(t_start=0, min_p=0, max_p=20)

    agent_a.update_flexibility(t_start=900, min_p=0, max_p=10)
    agent_b.update_flexibility(t_start=900, min_p=0, max_p=30)
    agent_c.update_flexibility(t_start=900, min_p=0, max_p=10)
    agent_d.update_flexibility(t_start=900, min_p=0, max_p=20)
    agent_e.update_flexibility(t_start=900, min_p=0, max_p=20)
    agent_f.update_flexibility(t_start=900, min_p=0, max_p=20)

    await agent_b.start_negotiation(ts=[0, 900], value=110)
    await agent_b.negotiation_done

    assert 'agent0' and 'agent2' and 'agent3' and 'agent4' and 'agent5' in agent_b.final

    await agent_b.start_negotiation(ts=[900, 1800], value=110)
    await agent_b.negotiation_done

    # after the negotiation, the agents should have updated their flexibility
    assert agent_a.flex[0] == [0, 0]
    assert agent_b.flex[0] == [0, 0]
    assert agent_c.flex[0] == [0, 0]
    assert agent_d.flex[0] == [0, 0]
    assert agent_e.flex[0] == [0, 0]
    assert agent_f.flex[0] == [0, 0]

    assert agent_a.flex[900] == [0, 0]
    assert agent_b.flex[900] == [0, 0]
    assert agent_c.flex[900] == [0, 0]
    assert agent_d.flex[900] == [0, 0]
    assert agent_e.flex[900] == [0, 0]
    assert agent_f.flex[900] == [0, 0]
    assert 'agent0' and 'agent2' and 'agent3' and 'agent4' and 'agent5' in agent_b.final

    await shutdown([agent_a, agent_b, agent_c, agent_d, agent_e, agent_f], [container])


@pytest.mark.asyncio
async def test_multiple_agents_start_negotiation():
    """
    Test negotiations with six agents where more than one agent starts a negotiation for one time
    interval. In this case, there is only enough flexibility to solve one problem.
    """
    agent_a, agent_b, agent_c, agent_d, agent_e, agent_f, container = await create_six_agents()

    agent_a.update_flexibility(t_start=0, min_p=0, max_p=10)
    agent_b.update_flexibility(t_start=0, min_p=0, max_p=30)
    agent_c.update_flexibility(t_start=0, min_p=0, max_p=10)
    agent_d.update_flexibility(t_start=0, min_p=0, max_p=20)
    agent_e.update_flexibility(t_start=0, min_p=0, max_p=20)
    agent_f.update_flexibility(t_start=0, min_p=0, max_p=20)

    await agent_b.start_negotiation(ts=[0, 900], value=90)
    await agent_c.start_negotiation(ts=[0, 900], value=90)

    await agent_b.negotiation_done
    await agent_c.negotiation_done

    # The agents first answer agent b and then agent c. Agent b can therefore solve its problem with enough
    # flexibility and forwards the solution before agent c. This leads to all agents sending acknowledgements to
    # agent b and not to agent c. Agent c first receives the replies and offers from other agents (at this time,
    # agent b did not find its solution) and finds a solution, later does not receiver acknowledgements and
    # therefore stops its negotiation after waiting for the acknowledgements.

    # agent c still stores its initial flexibility
    assert agent_c.flex[0] == [0, 10]

    # agent b found a solution (with all agents) and updated its flexibility
    assert 'agent0' and 'agent3' and 'agent4' and 'agent5' in agent_b.final
    assert agent_b.flex[0] == [0, 0]

    # agent b stores one unsuccessful negotiation
    assert len(agent_c._unsuccessful_negotiations) > 0

    await shutdown([agent_a, agent_b, agent_c, agent_d, agent_e, agent_f], [container])


@pytest.mark.asyncio
async def test_multiple_agents_start_negotiation_diff_direction():
    """
    Test negotiation with six agents where multiple agents start a negotiation. One agent starts a negotiation
    with an OfferNotification, another agent with a DemandNotification for the same interval and the same value.
    """
    agent_a, agent_b, agent_c, agent_d, agent_e, agent_f, container = await create_six_agents()

    agent_a.update_flexibility(t_start=0, min_p=0, max_p=10)
    agent_b.update_flexibility(t_start=0, min_p=0, max_p=30)
    agent_c.update_flexibility(t_start=0, min_p=0, max_p=10)
    agent_d.update_flexibility(t_start=0, min_p=0, max_p=20)
    agent_e.update_flexibility(t_start=0, min_p=0, max_p=20)
    agent_f.update_flexibility(t_start=0, min_p=0, max_p=20)

    await agent_b.start_negotiation(ts=[0, 900], value=90)
    await agent_c.start_negotiation(ts=[0, 900], value=-90)

    await agent_b.negotiation_done
    await agent_c.negotiation_done

    # In this case, agent_c withdraws its request since the other request is for the same value in the opposite
    # direction. Therefore, there is no result stored
    assert not agent_c.final

    # agent_b found a solution, since the flexibility is enough
    assert 'agent0' and 'agent2' and 'agent3' and 'agent4' in agent_b.final
    await shutdown([agent_a, agent_b, agent_c, agent_d, agent_e, agent_f], [container])
