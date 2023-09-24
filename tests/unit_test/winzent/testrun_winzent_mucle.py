import asyncio
import math
from datetime import datetime
from typing import Optional, Dict, List, Tuple

from util_functions import shutdown, create_six_ethical_agents, create_six_base_agents

"""
Test negotiation with Winzent with six agents. In this case, the agents have enough flexibility
to solve the problem successfully.
"""


async def run_six():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

    agent_a, agent_b, agent_c, agent_d, agent_e, agent_f, container = await create_six_base_agents()
    agent_a.update_flexibility(t_start=2700, min_p=0, max_p=10)
    agent_b.update_flexibility(t_start=2700, min_p=0, max_p=30)
    agent_c.update_flexibility(t_start=2700, min_p=0, max_p=10)
    agent_d.update_flexibility(t_start=2700, min_p=0, max_p=10)
    agent_e.update_flexibility(t_start=2700, min_p=0, max_p=10)
    agent_f.update_flexibility(t_start=2700, min_p=0, max_p=10)

    await agent_b.start_negotiation(ts=[2700, 3600], value=140)
    await agent_a.start_negotiation(ts=[2700, 3600], value=140)
    await agent_c.start_negotiation(ts=[2700, 3600], value=140)
    await agent_b.negotiation_done, agent_c.negotiation_done, agent_a.negotiation_done

    await shutdown([agent_a, agent_b, agent_c, agent_d, agent_e, agent_f], [container])


def calc_ethics_score_params(end, step_size):
    total_amount_of_steps = end / step_size
    sub_tier_size = 1.0 / total_amount_of_steps
    decay_rate = sub_tier_size / total_amount_of_steps
    return sub_tier_size, decay_rate


async def run_six_simple():
    # now = datetime.now()
    # current_time = now.strftime("%H:%M:%S")
    start_time = 2700
    end_time = start_time + 900
    # print("Current Time =", current_time)

    agent_a, agent_b, agent_c, agent_d, agent_e, agent_f, container = await create_six_ethical_agents()
    agent_a.update_flexibility(t_start=0, min_p=0, max_p=0)
    agent_b.update_flexibility(t_start=0, min_p=0, max_p=0)
    agent_c.update_flexibility(t_start=0, min_p=0, max_p=300)
    agent_d.update_flexibility(t_start=0, min_p=0, max_p=100)
    agent_e.update_flexibility(t_start=0, min_p=0, max_p=100)
    agent_f.update_flexibility(t_start=0, min_p=0, max_p=100)

    agent_a.update_flexibility(t_start=900, min_p=0, max_p=0)
    agent_b.update_flexibility(t_start=900, min_p=0, max_p=0)
    agent_c.update_flexibility(t_start=900, min_p=0, max_p=300)
    agent_d.update_flexibility(t_start=900, min_p=0, max_p=50)
    agent_e.update_flexibility(t_start=900, min_p=0, max_p=50)
    agent_f.update_flexibility(t_start=900, min_p=0, max_p=50)

    await agent_a.start_negotiation(start_dates=[0, 900], values=[150, 150])
    await agent_b.start_negotiation(start_dates=[0, 900], values=[150, 150])
    await agent_b.negotiation_done
    print("now shutting down" + str(agent_c.negotiation_done))
    print(agent_a.aid + str(agent_a.final))
    print(agent_b.aid + str(agent_b.final))
    # assert agent_a.flex[0] == [0, 0]
    # assert agent_b.flex[0] == [0, 0]
    # assert agent_c.flex[0] == [0, 0]
    # assert agent_d.flex[0] == [0, 0]
    # assert agent_e.flex[0] == [0, 0]
    # assert agent_f.flex[0] == [0, 0]
    #
    # assert agent_a.flex[900] == [0, 0]
    # assert agent_b.flex[900] == [0, 0]
    # assert agent_c.flex[900] == [0, 0]
    # assert agent_d.flex[900] == [0, 0]
    # assert agent_e.flex[900] == [0, 0]
    # assert agent_f.flex[900] == [0, 0]
    # after the negotiation, the agents should have updated their flexibility
    # assert agent_a.flex[0] == [0, 0]
    # assert agent_b.flex[0] == [0, 0]
    # assert agent_c.flex[0] == [0, 0]
    # assert agent_d.flex[0] == [0, 0]
    # assert agent_e.flex[0] == [0, 0]
    # assert agent_f.flex[0] == [0, 0]
    # assert 'agent0' and 'agent2' and 'agent3' and 'agent4' and 'agent5' in agent_b.final

    await shutdown([agent_a, agent_b, agent_c, agent_d, agent_e, agent_f], [container])


async def run_six_min_coverage_vs_best_ethics():
    start_time = 2700
    end_time = start_time + 900
    # print("Current Time =", current_time)

    agent_a, agent_b, agent_c, agent_d, agent_e, agent_f, container = await create_six_ethical_agents()
    agent_a.update_flexibility(t_start=0, min_p=0, max_p=0)
    agent_b.update_flexibility(t_start=0, min_p=0, max_p=100)
    agent_c.update_flexibility(t_start=0, min_p=0, max_p=3)
    agent_d.update_flexibility(t_start=0, min_p=0, max_p=0)
    agent_e.update_flexibility(t_start=0, min_p=0, max_p=0)
    agent_f.update_flexibility(t_start=0, min_p=0, max_p=0)

    agent_a.update_flexibility(t_start=900, min_p=0, max_p=0)
    agent_b.update_flexibility(t_start=900, min_p=0, max_p=50)
    agent_c.update_flexibility(t_start=900, min_p=0, max_p=0)
    agent_d.update_flexibility(t_start=900, min_p=0, max_p=50)
    agent_e.update_flexibility(t_start=900, min_p=0, max_p=50)
    agent_f.update_flexibility(t_start=900, min_p=0, max_p=50)

    agent_a.update_flexibility(t_start=1800, min_p=0, max_p=0)
    agent_b.update_flexibility(t_start=1800, min_p=0, max_p=50)
    agent_c.update_flexibility(t_start=1800, min_p=0, max_p=0)
    agent_d.update_flexibility(t_start=1800, min_p=0, max_p=50)
    agent_e.update_flexibility(t_start=1800, min_p=0, max_p=50)
    agent_f.update_flexibility(t_start=1800, min_p=0, max_p=50)

    await agent_a.start_negotiation(start_dates=[0, 900, 1800], values=[105, 150, 150])
    await agent_a.negotiation_done
    print("now shutting down" + str(agent_c.negotiation_done))
    print(agent_a.aid + str(agent_a.final))
    # assert agent_a.flex[0] == [0, 0]
    # assert agent_b.flex[0] == [0, 0]
    # assert agent_c.flex[0] == [0, 0]
    # assert agent_d.flex[0] == [0, 0]
    # assert agent_e.flex[0] == [0, 0]
    # assert agent_f.flex[0] == [0, 0]
    #
    # assert agent_a.flex[900] == [0, 0]
    # assert agent_b.flex[900] == [0, 0]
    # assert agent_c.flex[900] == [0, 0]
    # assert agent_d.flex[900] == [0, 0]
    # assert agent_e.flex[900] == [0, 0]
    # assert agent_f.flex[900] == [0, 0]
    # after the negotiation, the agents should have updated their flexibility
    # assert agent_a.flex[0] == [0, 0]
    # assert agent_b.flex[0] == [0, 0]
    # assert agent_c.flex[0] == [0, 0]
    # assert agent_d.flex[0] == [0, 0]
    # assert agent_e.flex[0] == [0, 0]
    # assert agent_f.flex[0] == [0, 0]
    # assert 'agent0' and 'agent2' and 'agent3' and 'agent4' and 'agent5' in agent_b.final

    await shutdown([agent_a, agent_b, agent_c, agent_d, agent_e, agent_f], [container])

def save_ethics_score_development(ethics_score_list, agent, success):
    ethics_score_tiers = list(ethics_score_list.keys())
    for tier in ethics_score_tiers:
        if tier <= agent.ethics_score < tier + 1.0:
            ethics_score_list[tier][0] = ethics_score_list[tier][0] + agent.ethics_score
            ethics_score_list[tier][0] += 1
            if not success:
                ethics_score_list[tier][1] += 1


async def run_muscle():
    agents_with_started_negotiation = []
    rounded_load_values: Dict[str:int] = {}
    start_time = 2700
    end_time = start_time + 900
    time_span = [start_time, end_time]
    # print("Current Time =", current_time)

    agent_a, agent_b, agent_c, agent_d, agent_e, agent_f, container = await create_six_ethical_agents()


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

    await agent_b.start_negotiation(start_dates=[0, 900], values=[110, 110])

    #agent_b_values = -100
    #await agent_b.start_negotiation(start_dates=[2700], values=[agent_b_values])
    rounded_load_values[agent_b.aid] = [110, 110]
    agents_with_started_negotiation.append(agent_b)

    # agent_a_values = 100
    # await agent_a.start_negotiation(ts=[2700, 3600], value=agent_a_values)
    # agents_with_started_negotiation.append(agent_a)
    # rounded_load_values[agent_a.aid] = agent_a_values
    #
    # agent_c_values = 100
    # await agent_c.start_negotiation(ts=[2700, 3600], value=agent_c_values)
    # agents_with_started_negotiation.append(agent_c)
    # rounded_load_values[agent_c.aid] = agent_c_values

    number_of_restarted_negotiations = 6
    ethics_score_list = {1.0: [0.0, 0, 0], 2.0: [0.0, 0, 0], 3.0: [0.0, 0, 0]}
    while len(agents_with_started_negotiation) > 0:
        agent = agents_with_started_negotiation.pop(0)
        try:
            await asyncio.wait_for(agent.negotiation_done, timeout=16)
        except asyncio.TimeoutError:
            print(f"{agent.aid} could not finish its negotiation in time. Result is set to zero.")
            agent.result = {}
        print(f"{agent.aid} negotiation done")
        # restart unsuccessful negotiations
        # only allow a restricted number of restarts
        agent_result_sum = 0
        for num in agent.result.values():
            agent_result_sum += num
        # check if negotiation fulfills requirements

        if agent_result_sum < rounded_load_values[agent.aid]:
            if number_of_restarted_negotiations > 0:
                # get sum of already negotiated values for this agent
                # negotiation was not fully successful, therefore restart
                agents_with_started_negotiation.append(agent)
                # restart the negotiation with the missing value
                await agent.start_negotiation(
                    ts=time_span,
                    value=rounded_load_values[agent.aid] - agent_result_sum,
                )
                print(
                    f"{agent.aid} restarted negotiation for value "
                    f"of {rounded_load_values[agent.aid] - agent_result_sum}"
                )
                number_of_restarted_negotiations -= 1
            elif agent_result_sum > rounded_load_values[agent.aid]:
                print(
                    f"Too much power")
            else:
                # agent.ethics_score = calculate_new_ethics_score(False, agent.ethics_score)
                # agents_ethics_score_list[agent.aid] = [False, agent.ethics_score]
                save_ethics_score_development(ethics_score_list, agent, False)
        else:
            # agent.ethics_score = calculate_new_ethics_score(True, agent.ethics_score)
            # agents_ethics_score_list[agent.aid] = [True, agent.ethics_score]
            save_ethics_score_development(ethics_score_list, agent, True)
    print(ethics_score_list)
    await shutdown([agent_a, agent_b, agent_c, agent_d, agent_e, agent_f], [container])



asyncio.run(run_six_min_coverage_vs_best_ethics())
