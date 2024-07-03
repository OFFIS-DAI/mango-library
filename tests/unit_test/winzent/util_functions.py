import mango.container.factory as factory

from mango_library.negotiation.winzent.winzent_base_agent import WinzentBaseAgent
from mango_library.negotiation.winzent.winzent_ethical_agent import WinzentEthicalAgent


async def shutdown(agents, containers):
    """
    Shutdown all agents and the container.
    """
    for agent in agents:
        await agent.stop_agent()
        await agent.shutdown()

    for container in containers:
        await container.shutdown()


async def create_six_base_agents(agent_tts=5):
    """
    Creates 6 simple agents, all living in one container and a neighborhood.
    """
    # container addr
    addr = ('127.0.0.1', 5555)

    # multiple container are possible, here just one is taken
    container = await factory.create(addr=addr)
    tts = agent_tts
    # create agents
    agent_a = WinzentBaseAgent(container=container, ttl=6, time_to_sleep=tts, ethics_score=1)
    agent_b = WinzentBaseAgent(container=container, ttl=6, time_to_sleep=tts, ethics_score=2)
    agent_c = WinzentBaseAgent(container=container, ttl=6, time_to_sleep=tts, ethics_score=3)
    agent_d = WinzentBaseAgent(container=container, ttl=6, time_to_sleep=tts, ethics_score=4)
    agent_e = WinzentBaseAgent(container=container, ttl=6, time_to_sleep=tts, ethics_score=5)
    agent_f = WinzentBaseAgent(container=container, ttl=6, time_to_sleep=tts, ethics_score=6)

    # create neighbors for agents
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
    # agent_a.add_neighbor(aid=agent_d.aid, addr=addr)

    agent_e.add_neighbor(aid=agent_d.aid, addr=addr)
    agent_d.add_neighbor(aid=agent_e.aid, addr=addr)

    agent_e.add_neighbor(aid=agent_c.aid, addr=addr)
    agent_c.add_neighbor(aid=agent_e.aid, addr=addr)

    # agent_f.add_neighbor(aid=agent_a.aid, addr=addr)
    # agent_a.add_neighbor(aid=agent_f.aid, addr=addr)

    agent_f.add_neighbor(aid=agent_b.aid, addr=addr)
    agent_b.add_neighbor(aid=agent_f.aid, addr=addr)

    return agent_a, agent_b, agent_c, agent_d, agent_e, agent_f, container


async def create_six_ethical_agents(
        agent_a_ethics_score=1,
        agent_b_ethics_score=1,
        agent_c_ethics_score=1,
        agent_d_ethics_score=1,
        agent_e_ethics_score=1,
        agent_f_ethics_score=1,
        setup="default"
):
    """
    Creates 6 simple agents, all living in one container and a neighborhood.
    """
    # container addr
    addr = ('127.0.0.1', 5555)

    # multiple container are possible, here just one is taken
    container = await factory.create(addr=addr)
    tts = 3
    use_ethics_score_as_negotiator = True
    use_ethics_score_as_contributor = True
    # create agents
    agent_a = WinzentEthicalAgent(container=container, ttl=6, time_to_sleep=tts,
                                  use_ethics_score_as_negotiator=use_ethics_score_as_negotiator,
                                  use_ethics_score_as_contributor=use_ethics_score_as_contributor,
                                  ethics_score=agent_a_ethics_score,
                                  min_coverage=0.9,
                                  coverage_weight=0.4)
    agent_b = WinzentEthicalAgent(container=container, ttl=6, time_to_sleep=tts,
                                  use_ethics_score_as_negotiator=use_ethics_score_as_negotiator,
                                  use_ethics_score_as_contributor=use_ethics_score_as_contributor,
                                  ethics_score=agent_b_ethics_score)
    agent_c = WinzentEthicalAgent(container=container, ttl=6, time_to_sleep=tts,
                                  use_ethics_score_as_negotiator=use_ethics_score_as_negotiator,
                                  use_ethics_score_as_contributor=use_ethics_score_as_contributor,
                                  ethics_score=agent_c_ethics_score)
    agent_d = WinzentEthicalAgent(container=container, ttl=6, time_to_sleep=tts,
                                  use_ethics_score_as_negotiator=use_ethics_score_as_negotiator,
                                  use_ethics_score_as_contributor=use_ethics_score_as_contributor,
                                  ethics_score=agent_d_ethics_score)
    agent_e = WinzentEthicalAgent(container=container, ttl=6, time_to_sleep=tts,
                                  use_ethics_score_as_negotiator=use_ethics_score_as_negotiator,
                                  use_ethics_score_as_contributor=use_ethics_score_as_contributor,
                                  ethics_score=agent_e_ethics_score)
    agent_f = WinzentEthicalAgent(container=container, ttl=6, time_to_sleep=tts,
                                  use_ethics_score_as_negotiator=use_ethics_score_as_negotiator,
                                  use_ethics_score_as_contributor=use_ethics_score_as_contributor,
                                  ethics_score=agent_f_ethics_score)

    if setup == "default":
        await create_restartable_test_case_neighboring(agent_a, agent_b, agent_c, agent_d, agent_e, agent_f, addr)
    elif setup == "max_coverage_vs_best_ethics":
        await create_max_coverage_vs_best_ethics_neighboring(agent_a, agent_b, agent_c, agent_d, agent_e, agent_f, addr)
    return agent_a, agent_b, agent_c, agent_d, agent_e, agent_f, container


async def create_restartable_test_case_neighboring(agent_a, agent_b, agent_c, agent_d, agent_e, agent_f, addr):
    agent_a.add_neighbor(aid=agent_b.aid, addr=addr)
    agent_a.add_neighbor(aid=agent_d.aid, addr=addr)

    agent_b.add_neighbor(aid=agent_a.aid, addr=addr)
    agent_b.add_neighbor(aid=agent_c.aid, addr=addr)
    agent_b.add_neighbor(aid=agent_e.aid, addr=addr)

    agent_c.add_neighbor(aid=agent_b.aid, addr=addr)
    agent_c.add_neighbor(aid=agent_f.aid, addr=addr)

    agent_d.add_neighbor(aid=agent_a.aid, addr=addr)
    agent_d.add_neighbor(aid=agent_e.aid, addr=addr)

    agent_e.add_neighbor(aid=agent_b.aid, addr=addr)
    agent_e.add_neighbor(aid=agent_d.aid, addr=addr)
    agent_e.add_neighbor(aid=agent_f.aid, addr=addr)

    agent_f.add_neighbor(aid=agent_c.aid, addr=addr)
    agent_f.add_neighbor(aid=agent_e.aid, addr=addr)


async def create_max_coverage_vs_best_ethics_neighboring(agent_a, agent_b, agent_c, agent_d, agent_e, agent_f,
                                                         addr):
    agent_a.add_neighbor(aid=agent_b.aid, addr=addr)

    agent_b.add_neighbor(aid=agent_a.aid, addr=addr)
    agent_b.add_neighbor(aid=agent_c.aid, addr=addr)
    agent_b.add_neighbor(aid=agent_e.aid, addr=addr)

    agent_c.add_neighbor(aid=agent_b.aid, addr=addr)
    agent_c.add_neighbor(aid=agent_f.aid, addr=addr)

    agent_d.add_neighbor(aid=agent_e.aid, addr=addr)

    agent_e.add_neighbor(aid=agent_b.aid, addr=addr)
    agent_e.add_neighbor(aid=agent_d.aid, addr=addr)
    agent_e.add_neighbor(aid=agent_f.aid, addr=addr)

    agent_f.add_neighbor(aid=agent_c.aid, addr=addr)
    agent_f.add_neighbor(aid=agent_e.aid, addr=addr)
