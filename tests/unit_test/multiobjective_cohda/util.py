import asyncio
from copy import deepcopy

import numpy as np
from mango.role.core import RoleAgent

from mango_library.coalition.core import CoalitionParticipantRole, CoalitionInitiatorRole, CoalitionModel
from mango_library.negotiation.multiobjective_cohda.cohda_messages import MoCohdaNegotiationMessage
from mango_library.negotiation.multiobjective_cohda.data_classes import Target
from mango_library.negotiation.multiobjective_cohda.multiobjective_cohda import MultiObjectiveCOHDARole, \
    MoCohdaNegotiationModel
from mango_library.negotiation.multiobjective_cohda.mocohda_starting import MoCohdaNegotiationDirectStarterRole
from mango_library.negotiation.termination import NegotiationTerminationParticipantRole, \
    NegotiationTerminationDetectorRole


def get_solution(agents):
    resulting_candidates = {}

    for a in agents:
        for role in a.roles:
            if isinstance(role, MultiObjectiveCOHDARole):
                cohda_objs = role.context.get_or_create_model(MoCohdaNegotiationModel)._negotiations
                print('cohda_objs', cohda_objs)
                for key, value in cohda_objs.items():
                    resulting_candidates[
                        value._part_id] = value._memory.solution_candidate

    final_candidate = list(resulting_candidates.values())[0]
    for part_id, candidate in resulting_candidates.items():
        assert np.array_equal(final_candidate.cluster_schedules,
                           candidate.cluster_schedules)

    return final_candidate


async def create_agents(container, targets, possible_schedules,
                        num_candidates, num_iterations,
                        check_msg_queue_interval, num_agents,
                        pick_fkt=None,
                        mutate_fkt=None,
                        schedules_all_equal=False,
                        use_fixed_ref_point=True,
                        offsets=None):
    agents = []
    addrs = []

    for i in range(num_agents):
        if isinstance(container, list):
            this_container = container[i % len(container)]
        else:
            this_container = container
        a = RoleAgent(this_container)

        def provide_schedules(index):
            if not schedules_all_equal:
                return deepcopy(lambda: possible_schedules[index])
            else:
                return lambda: possible_schedules

        cohda_role = MultiObjectiveCOHDARole(
            schedule_provider=provide_schedules(i % len(possible_schedules)),
            targets=targets,
            local_acceptable_func=lambda s: True,
            num_solution_points=num_candidates, num_iterations=num_iterations,
            check_inbox_interval=check_msg_queue_interval,
            pick_func=pick_fkt, mutate_func=mutate_fkt, use_fixed_ref_point=use_fixed_ref_point, offsets=offsets)
        a.add_role(cohda_role)
        a.add_role(CoalitionParticipantRole())
        a.add_role(NegotiationTerminationParticipantRole(negotiation_model_class=MoCohdaNegotiationModel,
                                                         negotiation_message_class=MoCohdaNegotiationMessage))
        agents.append(a)
        addrs.append((this_container.addr, a._aid))

    if isinstance(container, list):
        controller_agent = RoleAgent(container[-1])
    else:
        controller_agent = RoleAgent(container)
    controller_agent.add_role(NegotiationTerminationDetectorRole())
    controller_agent.add_role(CoalitionInitiatorRole(participants=addrs, details='', topic=''))

    await asyncio.wait_for(wait_for_coalition_built(agents), timeout=5)
    print('Coalition build done')
    agents[0].add_role(
        MoCohdaNegotiationDirectStarterRole(num_solution_points=num_candidates, target_params=None))

    print('Negotiation started')

    return agents, addrs, controller_agent


async def wait_for_coalition_built(agents):
    for agent in agents:
        while len(agent._agent_context.get_or_create_model(CoalitionModel)._assignments) < 1:
            await asyncio.sleep(0.1)


def determine_deviations(cs, target_params):
    dev_per_time_step = []
    sum_cs = sum(map(abs, cs))
    for idx, entry in enumerate(sum_cs):
        if idx + 1 < len(sum_cs):
            dev_per_time_step.append(abs(entry - sum_cs[idx + 1]))
    return float(np.mean(dev_per_time_step))


def determine_sums(cs, target_params):
    sum_cs = sum(map(abs, cs))
    return np.max(sum_cs)


# targets must be a list of tuples of length 2
# first item is the method, which is used to calculate the performance of
# the target second item is the reference point for this target
MINIMIZE_TARGETS = [
    Target(target_function=determine_deviations, ref_point=1, maximize=False),
    Target(target_function=determine_sums, ref_point=1, maximize=False)
]

MAXIMIZE_TARGETS = [
    Target(target_function=determine_deviations, ref_point=-1, maximize=True),
    Target(target_function=determine_sums, ref_point=-1, maximize=True)
]
