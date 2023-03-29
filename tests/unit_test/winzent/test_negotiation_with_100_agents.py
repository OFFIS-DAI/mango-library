import pytest

from util_functions import create_agents, shutdown


@pytest.mark.asyncio
async def test_successful_negotiation_hundred_agents():
    """
    Test negotiation with Winzent with 100 agents. In this case, the agents have more than enough flexibility
    to solve the problem successfully.
    """
    number_of_agents = 100
    ttl = 99
    agents, container = await create_agents(number_of_agents=number_of_agents, ttl=ttl, time_to_sleep=35)

    # since with 100 agents, the time_to_sleep needs to be really large, which slows down the complete negotiation,
    # new connections in the topology are added
    for idx in range(20):
        agents[idx].add_neighbor(aid=agents[idx + 10].aid, addr=container.addr)

    for agent in agents:
        agent.update_flexibility(t_start=0, min_p=0, max_p=10)

    await agents[0].start_negotiation(ts=[0, 900], value=400)

    await agents[0].negotiation_done
    assert not agents[0]._unsuccessful_negotiations
    print(agents[0].final)
    await shutdown(agents, [container])
