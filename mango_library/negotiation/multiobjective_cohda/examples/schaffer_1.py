import asyncio
from copy import deepcopy
import time
import numpy as np
from typing import List
import h5py

from mango_library.negotiation.multiobjective_cohda.data_classes import Target
from mango_library.negotiation.multiobjective_cohda.multiobjective_cohda import MultiObjectiveCOHDARole, COHDA, CohdaNegotiationStarterRole
from mango_library.coalition.core import CoalitionParticipantRole, CoalitionInitiatorRole, CoalitionModel
from mango_library.negotiation.termination import NegotiationTerminationParticipantRole, NegotiationTerminationDetectorRole
from mango.core.container import Container
from mango.role.core import RoleAgent

A = 10
NUM_AGENTS = 10
NUM_SCHEDULES = 20
NUM_SOLUTION_POINTS = 5
NUM_ITERATIONS = 1
TIMEOUT = 100
CHECK_INBOX_INTERVAL = 0.1

PICK_FKT = COHDA.pick_all_points
# PICK_FKT = COHDA.pick_random_point
MUTATE_FKT = COHDA.mutate_with_all_possible
# MUTATE_FKT = COHDA.mutate_with_one_random

NUM_SIMULATIONS = 1

def target_func_1(cs):
    """
    x ** 2
    """
    return cs.sum() ** 2

def target_func_2(cs):
    """
    (x - 2) ** 2
    """
    return (cs.sum() - 2) ** 2

TARGET_1 = Target(target_function=target_func_1, ref_point=A ** 2 * 1.1)
TARGET_2 = Target(target_function=target_func_2, ref_point=(A + 2) ** 2 * 1.1)

TARGETS = [TARGET_1, TARGET_2]

SCHEDULE_THRESHOLD = A / NUM_AGENTS
SCHEDULE_STEP_SIZE = (SCHEDULE_THRESHOLD * 2) / (NUM_SCHEDULES - 1)
POSSIBLE_SCHEDULES = []
for i in range(NUM_SCHEDULES):
    POSSIBLE_SCHEDULES.append(np.array([-SCHEDULE_THRESHOLD + i * SCHEDULE_STEP_SIZE]))

async def simulate_schaffer(db_file):
    container = await Container.factory(addr=('127.0.0.2', 5555))
    agents = []
    addrs = []

    for i in range(NUM_AGENTS):
        a = RoleAgent(container)
        a.add_role(MultiObjectiveCOHDARole(
            schedule_provider=lambda: POSSIBLE_SCHEDULES,
            targets=TARGETS,
            local_acceptable_func=lambda s: True,
            num_solution_points=NUM_SOLUTION_POINTS, num_iterations=NUM_ITERATIONS,
            check_inbox_interval=CHECK_INBOX_INTERVAL,
            pick_func=PICK_FKT, mutate_func=MUTATE_FKT)
        )
        a.add_role(CoalitionParticipantRole())
        a.add_role(NegotiationTerminationParticipantRole())
        agents.append(a)
        addrs.append((container.addr, a._aid))

    controller_agent = RoleAgent(container)
    controller_agent.add_role(NegotiationTerminationDetectorRole())
    controller_agent.add_role(CoalitionInitiatorRole(participants=addrs, details='', topic=''))
    await asyncio.wait_for(wait_for_coalition_built(agents), timeout=5)

    print('Coalition build done')
    start_time = time.time()
    agents[0].add_role(
        CohdaNegotiationStarterRole(num_solution_points=NUM_SOLUTION_POINTS, target_params=None))

    await asyncio.wait_for(wait_for_term(controller_agent), timeout=TIMEOUT)
    end_time = time.time()

    final_memory = next(iter(agents[0].roles[0]._cohda.values()))._memory
    for a in agents:
        assert final_memory == next(iter(a.roles[0]._cohda.values()))._memory

    # fill database now
    with h5py.File('Testfile.hdf5', 'w') as f:
        dtype_general = np.dtype([
            ('Name', 'S100'),
            ('n_agents', 'uint64'),
            ('n_objectives', 'uint64'),
            ('n_solution_points', 'uint64'),
            ('n_iterations', 'uint64'),
            ('msg_queue_interval', 'float64'),
            ('mutate_func', 'S100'),
            ('pick_func', 'S100'),
        ])
        data_general = np.array([('Schaffer1'.encode(), NUM_AGENTS, len(TARGETS), NUM_SOLUTION_POINTS, NUM_ITERATIONS,
                                  CHECK_INBOX_INTERVAL, MUTATE_FKT.__name__, PICK_FKT.__name__)], dtype=dtype_general)
        dtype_targets = np.dtype([
            ('Function', 'S100'),
            ('Ref Point', 'float64')
        ])
        data_targets = np.array([(t._target_function.__doc__, t.ref_point) for t in TARGETS], dtype=dtype_targets)

        dtype_schedules = [('aid', 'S100')]
        for i in range(NUM_SCHEDULES):
            dtype_schedules.append((f'schedule_{i}', 'float64'))
        dtype_schedules = np.dtype(dtype_schedules)
        data_schedules = []
        for a in agents:
            aid = a.aid
            schedules = a.roles[0]._schedule_provider()
            single_point_schedules = [s[0] for s in schedules]
            data_schedules.append(tuple([aid] + single_point_schedules))
        data_schedules = np.array(data_schedules, dtype=dtype_schedules)

        solution_points = final_memory.solution_candidate.solution_points

        general_group = f.create_group('General infos')
        general_group.create_dataset('general_info', data=data_general)
        general_group.create_dataset('targets', data=data_targets)
        general_group.create_dataset('schedules', data=data_schedules)
        results_group = f.create_group('Results')
        results_group.create_dataset('solution_point_1', solution_points[0].cluster_schedule)

    print('final Candidate', final_memory.solution_candidate)

    await container.shutdown()

async def simulate(*, num_agents: int, possible_schedules: List[List[List[float]]], targets: List[Target],
                   num_solution_points: int, pick_func, mutat_func, num_iteration: int, check_inbox_interval: float,
                   schedules_all_equal=False):
    container = await Container.factory(addr=('127.0.0.2', 5555))
    agents = []
    addrs = []

    for i in range(num_agents):
        a = RoleAgent(container)
        def provide_schedules(index):
            if not schedules_all_equal:
                return deepcopy(lambda: possible_schedules[index])
            else:
                return lambda: possible_schedules
        a.add_role(MultiObjectiveCOHDARole(
            schedule_provider=lambda: provide_schedules(i),
            targets=targets,
            local_acceptable_func=lambda s: True,
            num_solution_points=num_solution_points, num_iterations=num_iteration,
            check_inbox_interval=check_inbox_interval,
            pick_func=pick_func, mutate_func=mutat_func)
        )
        a.add_role(CoalitionParticipantRole())
        a.add_role(NegotiationTerminationParticipantRole())
        agents.append(a)
        addrs.append((container.addr, a._aid))

    controller_agent = RoleAgent(container)
    controller_agent.add_role(NegotiationTerminationDetectorRole())
    controller_agent.add_role(CoalitionInitiatorRole(participants=addrs, details='', topic=''))
    await asyncio.wait_for(wait_for_coalition_built(agents), timeout=5)

    print('Coalition build done')
    start_time = time.time()
    agents[0].add_role(
        CohdaNegotiationStarterRole(num_solution_points=NUM_SOLUTION_POINTS, target_params=None))

    await wait_for_term(controller_agent)

    end_time = time.time()

    final_memory = next(iter(agents[0].roles[0]._cohda.values()))._memory
    for a in agents:
        assert final_memory == next(iter(a.roles[0]._cohda.values()))._memory

    return final_memory, end_time - start_time



async def wait_for_term(controller_agent):
    while len(controller_agent.roles[0]._weight_map.values()) != 1 or list(controller_agent.roles[0]._weight_map.values())[0] != 1:
        await asyncio.sleep(0.1)
    print('Terminated!')

async def wait_for_coalition_built(agents):
    for agent in agents:
        while len(agent._agent_context.get_or_create_model(CoalitionModel)._assignments) < 1:
            await asyncio.sleep(0.1)

if __name__ == '__main__':
    asyncio.run(simulate_schaffer(None))
