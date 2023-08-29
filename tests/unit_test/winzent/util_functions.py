from mango.core.container import Container

from negotiation.winzent.winzent_base_agent import WinzentBaseAgent
from negotiation.winzent.winzent_ethical_agent import WinzentEthicalAgent


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
    agent_a = WinzentBaseAgent(container=container, ttl=2)
    agent_b = WinzentBaseAgent(container=container, ttl=2)
    agent_c = WinzentBaseAgent(container=container, ttl=2)

    # create random neighbors for agents
    agent_a.add_neighbor(aid=agent_b.aid,
                         addr=addr)
    agent_a.add_neighbor(aid=agent_c.aid,
                         addr=addr)

    agent_b.add_neighbor(aid=agent_a.aid, addr=addr)
    agent_c.add_neighbor(aid=agent_a.aid, addr=addr)

    return agent_a, agent_b, agent_c, container


async def create_six_base_agents(agent_tts=5):
    """
    Creates 6 simple agents, all living in one container and a neighborhood.
    """
    # container addr
    addr = ('127.0.0.1', 5555)

    # multiple container are possible, here just one is taken
    container = await Container.factory(addr=addr)

    tts = agent_tts
    # create agents
    agent_a = WinzentBaseAgent(container=container, ttl=6, time_to_sleep=tts, ethics_score=1)
    agent_b = WinzentBaseAgent(container=container, ttl=6, time_to_sleep=tts, ethics_score=2)
    agent_c = WinzentBaseAgent(container=container, ttl=6, time_to_sleep=tts, ethics_score=3)
    agent_d = WinzentBaseAgent(container=container, ttl=6, time_to_sleep=tts, ethics_score=4)
    agent_e = WinzentBaseAgent(container=container, ttl=6, time_to_sleep=tts, ethics_score=5)
    agent_f = WinzentBaseAgent(container=container, ttl=6, time_to_sleep=tts, ethics_score=6)

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
    agent_f.add_neighbor(aid=agent_b.aid, addr=addr)
    agent_a.add_neighbor(aid=agent_f.aid, addr=addr)

    #agent_f.add_neighbor(aid=agent_b.aid, addr=addr)
    #agent_b.add_neighbor(aid=agent_f.aid, addr=addr)

    return agent_a, agent_b, agent_c, agent_d, agent_e, agent_f, container


async def create_six_simple_ethical_agents():
    """
    Creates 6 simple agents, all living in one container and a neighborhood.
    """
    # container addr
    addr = ('127.0.0.1', 5555)

    # multiple container are possible, here just one is taken
    container = await Container.factory(addr=addr)

    tts = 3
    # create agents
    agent_a = WinzentEthicalAgent(container=container, ttl=6, time_to_sleep=tts, ethics_score=1)
    agent_b = WinzentEthicalAgent(container=container, ttl=6, time_to_sleep=tts, ethics_score=2)
    agent_c = WinzentEthicalAgent(container=container, ttl=6, time_to_sleep=tts, ethics_score=3)
    agent_d = WinzentEthicalAgent(container=container, ttl=6, time_to_sleep=tts, ethics_score=4)
    agent_e = WinzentEthicalAgent(container=container, ttl=6, time_to_sleep=tts, ethics_score=5)
    agent_f = WinzentEthicalAgent(container=container, ttl=6, time_to_sleep=tts, ethics_score=6)

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

    #agent_f.add_neighbor(aid=agent_b.aid, addr=addr)
    #agent_b.add_neighbor(aid=agent_f.aid, addr=addr)

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
        agents.append(WinzentBaseAgent(container=container, ttl=ttl, time_to_sleep=time_to_sleep))

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
