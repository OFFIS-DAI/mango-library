import asyncio
import math
from datetime import datetime

from util_functions import create_six_agents, shutdown, create_six_ethical_agents

"""
Test negotiation with Winzent with six agents. In this case, the agents have enough flexibility
to solve the problem successfully.
"""


async def run_six():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

    agent_a, agent_b, agent_c, agent_d, agent_e, agent_f, container = await create_six_agents()
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
    calculate_new_ethics_score(False)
    # print(agent_a.aid + str(agent_a.final))
    # print(agent_b.aid + str(agent_b.final))
    # print(agent_c.aid + str(agent_c.final))
    # after the negotiation, the agents should have updated their flexibility
    # assert agent_a.flex[0] == [0, 0]
    # assert agent_b.flex[0] == [0, 0]
    # assert agent_c.flex[0] == [0, 0]
    # assert agent_d.flex[0] == [0, 0]
    # assert agent_e.flex[0] == [0, 0]
    # assert agent_f.flex[0] == [0, 0]
    # assert 'agent0' and 'agent2' and 'agent3' and 'agent4' and 'agent5' in agent_b.final

    await shutdown([agent_a, agent_b, agent_c, agent_d, agent_e, agent_f], [container])


def calc_ethics_score_params(end, step_size):
    total_amount_of_steps = end / step_size
    sub_tier_size = 1.0 / total_amount_of_steps
    decay_rate = sub_tier_size / total_amount_of_steps
    return sub_tier_size, decay_rate


def calculate_new_ethics_score(success, ethics_score):
    params = calc_ethics_score_params(24 * 60 * 60, 900)
    decay_rate = params[1]
    sub_tier_size = params[0]
    max_len_of_ethics_score = "{:." + str(len(str(decay_rate).replace('.', ''))) + "f}"
    min_len_of_ethics_score = "{:." + str(len(str(sub_tier_size).replace('.', ''))) + "f}"
    initial_ethics_score = float(math.floor(ethics_score))
    str_eth_score = list(str(ethics_score))
    str_eth_score[0] = "0"
    str_eth_score = float("".join(str_eth_score))
    amount_of_outages = int(str_eth_score / sub_tier_size)
    # print(amount_of_outages)
    current_tier_low = max(float(str(ethics_score)[0]) + (amount_of_outages * (sub_tier_size)),
                           initial_ethics_score)
    current_tier_high = max(float(str(ethics_score)[0]) + ((amount_of_outages + 1) * sub_tier_size),
                            initial_ethics_score)
    if not success:
        temp = math.floor(ethics_score * 10) / 10
        print(temp)
        if (math.floor(float(temp)) + 1) > (float(temp) + sub_tier_size):
            if ethics_score == initial_ethics_score:
                return float(max_len_of_ethics_score.format(initial_ethics_score + sub_tier_size - decay_rate))
            return float(max_len_of_ethics_score.format(current_tier_high + sub_tier_size - decay_rate))
        else:
            return float(max_len_of_ethics_score.format((math.floor(float(ethics_score)) + 1) - decay_rate))
    else:
        # print(amount_of_outages)
        temp_ethics_score = float(max_len_of_ethics_score.format(ethics_score - decay_rate))
        # print(lower_tier_end)
        print(temp_ethics_score)
        if temp_ethics_score <= current_tier_low:
            return current_tier_low
        else:
            return temp_ethics_score


async def run_six_simple():
    # now = datetime.now()
    # current_time = now.strftime("%H:%M:%S")
    start_time = 2700
    end_time = start_time + 900
    # print("Current Time =", current_time)

    agent_a, agent_b, agent_c, agent_d, agent_e, agent_f, container = await create_six_agents()
    agent_a.update_flexibility(t_start=2700, min_p=0, max_p=10)
    agent_b.update_flexibility(t_start=2700, min_p=0, max_p=30)
    agent_c.update_flexibility(t_start=2700, min_p=0, max_p=10)
    agent_d.update_flexibility(t_start=2700, min_p=0, max_p=1000)
    agent_e.update_flexibility(t_start=2700, min_p=0, max_p=1000)
    agent_f.update_flexibility(t_start=2700, min_p=0, max_p=1000)

    await agent_b.start_negotiation(ts=[2700, 3600], value=140)
    await agent_a.start_negotiation(ts=[2700, 3600], value=140)
    await agent_c.start_negotiation(ts=[2700, 3600], value=140)
    await agent_b.negotiation_done
    await agent_c.negotiation_done
    await agent_a.negotiation_done
    agent_a.update_flexibility(t_start=3600, min_p=0, max_p=10)
    agent_b.update_flexibility(t_start=3600, min_p=0, max_p=30)
    agent_c.update_flexibility(t_start=3600, min_p=0, max_p=10)
    agent_d.update_flexibility(t_start=3600, min_p=0, max_p=1000)
    agent_e.update_flexibility(t_start=3600, min_p=0, max_p=1000)
    agent_f.update_flexibility(t_start=3600, min_p=0, max_p=1000)
    await agent_b.start_negotiation(ts=[3600, 4500], value=140)
    await agent_a.start_negotiation(ts=[3600, 4500], value=140)
    await agent_c.start_negotiation(ts=[3600, 4500], value=140)
    await agent_b.negotiation_done
    await agent_c.negotiation_done
    await agent_a.negotiation_done
    print("now shutting down" + str(agent_c.negotiation_done))
    print(agent_a.aid + str(agent_a.final))
    print(agent_b.aid + str(agent_b.final))
    print(agent_c.aid + str(agent_c.final))
    # after the negotiation, the agents should have updated their flexibility
    # assert agent_a.flex[0] == [0, 0]
    # assert agent_b.flex[0] == [0, 0]
    # assert agent_c.flex[0] == [0, 0]
    # assert agent_d.flex[0] == [0, 0]
    # assert agent_e.flex[0] == [0, 0]
    # assert agent_f.flex[0] == [0, 0]
    # assert 'agent0' and 'agent2' and 'agent3' and 'agent4' and 'agent5' in agent_b.final

    await shutdown([agent_a, agent_b, agent_c, agent_d, agent_e, agent_f], [container])

asyncio.run(run_six_simple())
# test = {3: ["Klinikum", "PV", "Wind"], 2: ["Households", "Abfall"], 1: [""]}
# ethics_score = 2.0
# ethics_score = calculate_new_ethics_score(True, ethics_score)
# print("new ethics score: " + str(ethics_score))
# ethics_score = calculate_new_ethics_score(True, ethics_score)
# print("new ethics score: " + str(ethics_score))
# ethics_score = calculate_new_ethics_score(False, ethics_score)
# print("new ethics score: " + str(ethics_score))
# ethics_score = calculate_new_ethics_score(False, ethics_score)
# print("new ethics score: " + str(ethics_score))
# ethics_score = calculate_new_ethics_score(False, ethics_score)
# print("new ethics score: " + str(ethics_score))
# ethics_score = calculate_new_ethics_score(True, ethics_score)
# print("new ethics score: " + str(ethics_score))
# ethics_score = calculate_new_ethics_score(False, ethics_score)
# print("new ethics score: " + str(ethics_score))
# ethics_score = calculate_new_ethics_score(True, ethics_score)
# print("new ethics score: " + str(ethics_score))
# ethics_score = calculate_new_ethics_score(True, ethics_score)
# print("new ethics score: " + str(ethics_score))
# ethics_score = calculate_new_ethics_score(True, ethics_score)
# print("new ethics score: " + str(ethics_score))
# ethics_score = calculate_new_ethics_score(False, ethics_score)
# print("new ethics score: " + str(ethics_score))
# ethics_score = calculate_new_ethics_score(False, ethics_score)
# print("new ethics score: " + str(ethics_score))
# ethics_score = calculate_new_ethics_score(False, ethics_score)
# print("new ethics score: " + str(ethics_score))
# ethics_score = calculate_new_ethics_score(False, ethics_score)
# print("new ethics score: " + str(ethics_score))
# ethics_score = calculate_new_ethics_score(False, ethics_score)
# print("new ethics score: " + str(ethics_score))
# ethics_score = calculate_new_ethics_score(False, ethics_score)
# print("new ethics score: " + str(ethics_score))
# ethics_score = calculate_new_ethics_score(False, ethics_score)
# print("new ethics score: " + str(ethics_score))
# ethics_score = calculate_new_ethics_score(True, ethics_score)
# print("new ethics score: " + str(ethics_score))
# ethics_score = calculate_new_ethics_score(True, ethics_score)
# print("new ethics score: " + str(ethics_score))
# ethics_score = calculate_new_ethics_score(True, ethics_score)
# print("new ethics score: " + str(ethics_score))
# ethics_score = calculate_new_ethics_score(True, ethics_score)
# print("new ethics score: " + str(ethics_score))
# ethics_score = calculate_new_ethics_score(True, ethics_score)
# print("new ethics score: " + str(ethics_score))
# ethics_score = calculate_new_ethics_score(True, ethics_score)
# print("new ethics score: " + str(ethics_score))
# ethics_score = calculate_new_ethics_score(True, ethics_score)
# print("new ethics score: " + str(ethics_score))
# ethics_score = calculate_new_ethics_score(True, ethics_score)
# print("new ethics score: " + str(ethics_score))
# ethics_score = calculate_new_ethics_score(True, ethics_score)
# print("new ethics score: " + str(ethics_score))
# ethics_score = calculate_new_ethics_score(True, ethics_score)
# print("new ethics score: " + str(ethics_score))
# ethics_score = calculate_new_ethics_score(True, ethics_score)
# print("new ethics score: " + str(ethics_score))
# ethics_score = calculate_new_ethics_score(True, ethics_score)
