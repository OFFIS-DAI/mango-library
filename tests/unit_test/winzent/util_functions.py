from mango.core.container import Container

from mango_library.negotiation.winzent.winzent_agent import WinzentAgent


async def shutdown(agents, containers):
    """
    Shutdown all agents and the container.
    """
    for agent in agents:
        await agent.stop_agent()
        await agent.shutdown()

    for container in containers:
        await container.shutdown()


async def create_three_agents():
    """
    Creates 3 simple agents, all living in one container and a neighborhood.
    """
    # container addr
    addr = ('127.0.0.1', 5555)

    # multiple container are possible, here just one is taken
    container = await Container.factory(addr=addr)

    # create agents
    agent_a = WinzentAgent(container=container, ttl=2)
    agent_b = WinzentAgent(container=container, ttl=2)
    agent_c = WinzentAgent(container=container, ttl=2)

    # create random neighbors for agents
    agent_a.add_neighbor(aid=agent_b.aid,
                         addr=addr)
    agent_a.add_neighbor(aid=agent_c.aid,
                         addr=addr)

    agent_b.add_neighbor(aid=agent_a.aid, addr=addr)
    agent_c.add_neighbor(aid=agent_a.aid, addr=addr)

    return agent_a, agent_b, agent_c, container


async def create_six_agents():
    """
    Creates 6 simple agents, all living in one container and a neighborhood.
    """
    # container addr
    addr = ('127.0.0.1', 5555)

    # multiple container are possible, here just one is taken
    container = await Container.factory(addr=addr)

    # create agents
    agent_a = WinzentAgent(container=container, ttl=2, time_to_sleep=3)
    agent_b = WinzentAgent(container=container, ttl=2, time_to_sleep=3)
    agent_c = WinzentAgent(container=container, ttl=2, time_to_sleep=3)
    agent_d = WinzentAgent(container=container, ttl=2, time_to_sleep=3)
    agent_e = WinzentAgent(container=container, ttl=2, time_to_sleep=3)
    agent_f = WinzentAgent(container=container, ttl=2, time_to_sleep=3)

    # create random neighbors for agents
    agent_a.add_neighbor(aid=agent_b.aid,
                         addr=addr)
    agent_a.add_neighbor(aid=agent_c.aid,
                         addr=addr)

    agent_b.add_neighbor(aid=agent_a.aid, addr=addr)
    agent_b.add_neighbor(aid=agent_e.aid, addr=addr)
    agent_e.add_neighbor(aid=agent_b.aid, addr=addr)

    agent_b.add_neighbor(aid=agent_c.aid, addr=addr)
    agent_c.add_neighbor(aid=agent_a.aid, addr=addr)
    agent_c.add_neighbor(aid=agent_b.aid, addr=addr)
    agent_c.add_neighbor(aid=agent_d.aid, addr=addr)

    agent_d.add_neighbor(aid=agent_a.aid, addr=addr)
    agent_a.add_neighbor(aid=agent_d.aid, addr=addr)

    agent_e.add_neighbor(aid=agent_d.aid, addr=addr)
    agent_d.add_neighbor(aid=agent_e.aid, addr=addr)

    agent_e.add_neighbor(aid=agent_c.aid, addr=addr)
    agent_c.add_neighbor(aid=agent_e.aid, addr=addr)

    agent_f.add_neighbor(aid=agent_a.aid, addr=addr)
    agent_a.add_neighbor(aid=agent_f.aid, addr=addr)

    return agent_a, agent_b, agent_c, agent_d, agent_e, agent_f, container


async def create_agents(number_of_agents, ttl, time_to_sleep):
    """
    Function to create simple agents, all living in one container and a neighborhood.
    """
    # container addr
    addr = ('127.0.0.1', 5555)

    # multiple container are possible, here just one is taken
    container = await Container.factory(addr=addr)

    agents = []

    # create agents
    for idx in range(number_of_agents):
        agents.append(WinzentAgent(container=container, ttl=ttl, time_to_sleep=time_to_sleep))

    # create ring
    for agent_idx in range(len(agents)):
        if agent_idx % 2 != 0:
            continue

        if agent_idx == 0:
            agents[agent_idx].add_neighbor(aid=agents[agent_idx+1].aid, addr=addr)
            agents[agent_idx + 1].add_neighbor(aid=agents[agent_idx].aid, addr=addr)

            agents[agent_idx].add_neighbor(aid=agents[-1].aid, addr=addr)
            agents[-1].add_neighbor(aid=agents[-1].aid, addr=addr)
        elif agent_idx == -1:
            agents[-1].add_neighbor(aid=agents[-2].aid, addr=addr)
            agents[-2].add_neighbor(aid=agents[-1].aid, addr=addr)
        else:
            agents[agent_idx].add_neighbor(aid=agents[agent_idx + 1].aid, addr=addr)
            agents[agent_idx + 1].add_neighbor(aid=agents[agent_idx].aid, addr=addr)

            agents[agent_idx].add_neighbor(aid=agents[agent_idx - 1].aid, addr=addr)
            agents[agent_idx - 1].add_neighbor(aid=agents[agent_idx].aid, addr=addr)

    return agents, container