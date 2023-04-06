import asyncio
from datetime import datetime

from util_functions import create_six_agents, shutdown

"""
Test negotiation with Winzent with six agents. In this case, the agents have enough flexibility
to solve the problem successfully.
"""
async def run_six():
    now = datetime.now()
    test = False
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

    agent_a, agent_b, agent_c, agent_d, agent_e, agent_f, container = await create_six_agents()
    agent_a.update_flexibility(t_start=0, min_p=0, max_p=10)
    agent_b.update_flexibility(t_start=0, min_p=0, max_p=30)
    agent_c.update_flexibility(t_start=0, min_p=0, max_p=10)
    agent_d.update_flexibility(t_start=0, min_p=0, max_p=20)
    agent_e.update_flexibility(t_start=0, min_p=0, max_p=20)
    agent_f.update_flexibility(t_start=0, min_p=0, max_p=110)

    await agent_b.start_negotiation(ts=[0, 900], value=140)
    await agent_b.negotiation_done
    print(agent_b.final)
    # print(agent_c.final)
    print(agent_b._current_inquiries_from_agents)
    # after the negotiation, the agents should have updated their flexibility
    #assert agent_a.flex[0] == [0, 0]
    #assert agent_b.flex[0] == [0, 0]
    #assert agent_c.flex[0] == [0, 0]
    #assert agent_d.flex[0] == [0, 0]
    #assert agent_e.flex[0] == [0, 0]
    #assert agent_f.flex[0] == [0, 0]
    #assert 'agent0' and 'agent2' and 'agent3' and 'agent4' and 'agent5' in agent_b.final
    await shutdown([agent_a, agent_b, agent_c, agent_d, agent_e, agent_f], [container])

asyncio.run(run_six())