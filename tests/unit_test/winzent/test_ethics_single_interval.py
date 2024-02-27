import asyncio
from typing import Dict

from util_functions import shutdown, create_six_ethical_agents


async def get_correct_solution_through_restarts():
    """
     In this test case, agent a, agent b and agent c are negotiating for a solution over
     two intervals. Agent d, agent e and agent f possess the needed flexibility for a, b and c.
     Three restarts are needed to match each agent to his corresponding supplier.
     Furthermore, the highest ranked consumer (ethically) matches with the highest ranked producer.
     """
    agents_with_started_negotiation = []
    rounded_load_values: Dict[str:int] = {}
    first_interval = 0
    time_span = [first_interval]
    # this variable controls the amount of allowed restarts
    number_of_restarted_negotiations_allowed = 3

    agent_a, agent_b, agent_c, agent_d, agent_e, agent_f, container = await create_six_ethical_agents(
        agent_a_ethics_score=2,
        agent_b_ethics_score=3,
        agent_c_ethics_score=4,
        agent_d_ethics_score=2,
        agent_e_ethics_score=3,
        agent_f_ethics_score=4,
    )

    agent_a.update_flexibility(t_start=first_interval, min_p=0, max_p=0)
    agent_b.update_flexibility(t_start=first_interval, min_p=0, max_p=0)
    agent_c.update_flexibility(t_start=first_interval, min_p=0, max_p=0)
    agent_d.update_flexibility(t_start=first_interval, min_p=0, max_p=100)
    agent_e.update_flexibility(t_start=first_interval, min_p=0, max_p=100)
    agent_f.update_flexibility(t_start=first_interval, min_p=0, max_p=100)

    agent_a_values = [100]
    await agent_a.start_negotiation(start_dates=[first_interval], values=agent_a_values)
    agents_with_started_negotiation.append(agent_a)
    rounded_load_values[agent_a.aid] = agent_a_values

    agent_b_values = [100]
    await agent_b.start_negotiation(start_dates=[first_interval], values=agent_b_values)
    rounded_load_values[agent_b.aid] = agent_b_values
    agents_with_started_negotiation.append(agent_b)

    agent_c_values = [100]
    await agent_c.start_negotiation(start_dates=[first_interval], values=agent_c_values)
    agents_with_started_negotiation.append(agent_c)
    rounded_load_values[agent_c.aid] = agent_c_values

    while len(agents_with_started_negotiation) > 0:
        agent = agents_with_started_negotiation.pop(0)
        try:
            await asyncio.wait_for(agent.negotiation_done, timeout=5)
        except asyncio.TimeoutError:
            print(f"{agent.aid} could not finish its negotiation in time. Result is set to zero.")
            agent.result = {}
        # restart unsuccessful negotiations
        # only allow a restricted number of restarts
        agent_result_sum = [0 for _ in time_span]
        for key in agent.final:
            it = 0
            for sub_key in agent.final[key]:
                agent_result_sum[it] += abs(agent.final[key][sub_key][0])
                it += 1
        # check if negotiation fulfills requirements

        negotiation_successful = sum(agent_result_sum) >= sum(rounded_load_values[agent.aid])
        if not negotiation_successful:
            if number_of_restarted_negotiations_allowed > 0:
                # get sum of already negotiated values for this agent
                # negotiation was not fully successful, therefore restart
                agents_with_started_negotiation.append(agent)
                # restart the negotiation with the missing value
                await agent.start_negotiation(
                    start_dates=[first_interval],
                    values=[a - b for a, b in zip(rounded_load_values[agent.aid], agent_result_sum)],
                )
                print(
                    f"{agent.aid} restarted negotiation for value "
                    f"of {[a - b for a, b in zip(rounded_load_values[agent.aid], agent_result_sum)]}"
                )
                print(agent.calculate_new_ethics_score(negotiation_successful))
                number_of_restarted_negotiations_allowed -= 1
            else:
                print(agent.calculate_new_ethics_score(negotiation_successful))
        else:
            print(agent.calculate_new_ethics_score(negotiation_successful))
            print(f"Negotiation successful! {agent.aid} has the final solution: {agent.final}")

    await shutdown([agent_a, agent_b, agent_c, agent_d, agent_e, agent_f], [container])

    assert agent_a.flex[first_interval] == [0, 0]
    assert agent_b.flex[first_interval] == [0, 0]
    assert agent_c.flex[first_interval] == [0, 0]
    assert agent_d.flex[first_interval] == [0, 0]
    assert agent_e.flex[first_interval] == [0, 0]
    assert agent_f.flex[first_interval] == [0, 0]

    assert agent_a.ethics_score == 2.18
    assert agent_b.ethics_score == 3.08
    assert agent_c.ethics_score == 4.0


asyncio.run(get_correct_solution_through_restarts())
