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
CHECK_INBOX_INTERVAL = 0.05

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
for schedule_no in range(NUM_SCHEDULES):
    POSSIBLE_SCHEDULES.append(np.array([-SCHEDULE_THRESHOLD + schedule_no * SCHEDULE_STEP_SIZE]))

async def simulate_schaffer(name, db_file):
    results = {}
    for sim_no in range(NUM_SIMULATIONS):
        results[sim_no] = await simulate(
            num_agents=NUM_AGENTS,
            possible_schedules=POSSIBLE_SCHEDULES, schedules_all_equal=True,
            targets=TARGETS, num_solution_points=NUM_SOLUTION_POINTS, num_iteration=NUM_ITERATIONS,
            check_inbox_interval=CHECK_INBOX_INTERVAL, pick_func=PICK_FKT, mutat_func=MUTATE_FKT
        )

    store_in_db(
        db_file='Testfile.hdf5', sim_name='Schaffer_1', n_agents=NUM_AGENTS, targets=TARGETS,
        n_solution_points=NUM_SOLUTION_POINTS, n_iterations=NUM_ITERATIONS, check_inbox_interval=CHECK_INBOX_INTERVAL,
        mutate_func=MUTATE_FKT, pick_func=PICK_FKT, results=results
    )


    # fill database now

def store_in_db(*, db_file, sim_name, n_agents, targets, n_solution_points, n_iterations, check_inbox_interval,
                mutate_func, pick_func, results):
    with h5py.File(db_file, 'w') as f:
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
        data_general = np.array([(sim_name.encode(), n_agents, len(targets), n_solution_points, n_iterations,
                                  check_inbox_interval, mutate_func.__name__, pick_func.__name__)],
                                dtype=dtype_general)
        dtype_targets = np.dtype([
            ('Function', 'S100'),
            ('Ref Point', 'float64')
        ])
        data_targets = np.array([(t._target_function.__doc__, t.ref_point) for t in targets], dtype=dtype_targets)

        dtype_schedules = [('aid', 'S100')]
        for i in range(NUM_SCHEDULES):
            dtype_schedules.append((f'schedule_{i}', 'float64'))
        dtype_schedules = np.dtype(dtype_schedules)
        data_schedules = []
        schedules_per_agent = results[0][2]
        for _, _, schedules_in_simulation, _ in results.values():
            assert schedules_per_agent == schedules_in_simulation

        for aid, sched_list in schedules_per_agent.items():
            single_point_schedules = [s[0] for s in sched_list]
            data_schedules.append(tuple([aid] + single_point_schedules))
        data_schedules = np.array(data_schedules, dtype=dtype_schedules)

        general_group = f.create_group('General infos')
        general_group.create_dataset('general_info', data=data_general)
        general_group.create_dataset('targets', data=data_targets)
        general_group.create_dataset('Schedules', data=data_schedules)
        results_group = f.create_group('Results')

        for sim_no, (final_memory, duration, _, overlay) in results.items():
            sim_results_grp = results_group.create_group(f'Results_{sim_no}')
            solution_candidate = final_memory.solution_candidate
            dtype_general_result = np.dtype([
                ('Duration', 'float64'),
                ('Hypervolume', 'float64')
            ])
            # for i in range(n_solution_points):
            #     dtype_general_result.append((f'Performance_{i}', 'S100'))
            data_general_results = np.array([(duration, solution_candidate.hypervolume)], dtype=dtype_general_result)
            sim_results_grp.create_dataset('general results', data=data_general_results)

            dtype_performances = np.dtype([
                (f'Perfomace_{i}', 'float64') for i, _ in enumerate(targets)
            ])
            data_perf = np.array(sorted(solution_candidate.perf), dtype=dtype_performances)
            sim_results_grp.create_dataset('performances', data=data_perf)

            dtype_solution_points = np.dtype([
                ('part_id', 'S100'),
                ('value', 'float64')
            ])

            for i, solution_point in enumerate(sorted(solution_candidate.solution_points)):
                data_solution_points = []
                for part_id, index in solution_point.idx.items():
                    data_solution_points.append((part_id, solution_point.cluster_schedule[index][0]))
                data_solution_points = np.array(data_solution_points, dtype=dtype_solution_points)
                sim_results_grp.create_dataset(f'solutionpoint_{i}', data=data_solution_points)


async def simulate(*, num_agents: int, possible_schedules, targets: List[Target],
                   num_solution_points: int, pick_func, mutat_func, num_iteration: int, check_inbox_interval: float,
                   schedules_all_equal=False):
    container = await Container.factory(addr=('127.0.0.2', 5555))
    agents = []
    addrs = []
    schedules_per_agent = {}

    for i in range(num_agents):
        a = RoleAgent(container)
        def provide_schedules(index):
            if not schedules_all_equal:
                return deepcopy(lambda: possible_schedules[index])
            else:
                return lambda: possible_schedules
        a.add_role(MultiObjectiveCOHDARole(
            schedule_provider=provide_schedules(i),
            targets=targets,
            local_acceptable_func=lambda s: True,
            num_solution_points=num_solution_points, num_iterations=num_iteration,
            check_inbox_interval=check_inbox_interval,
            pick_func=pick_func, mutate_func=mutat_func)
        )
        schedules_per_agent[a.aid] = provide_schedules(i)()
        a.add_role(CoalitionParticipantRole())
        a.add_role(NegotiationTerminationParticipantRole())
        agents.append(a)
        addrs.append((container.addr, a._aid))

    controller_agent = RoleAgent(container)
    controller_agent.add_role(NegotiationTerminationDetectorRole())
    controller_agent.add_role(CoalitionInitiatorRole(participants=addrs, details='', topic=''))
    await asyncio.wait_for(wait_for_coalition_built(agents), timeout=5)
    print('Coalition build done')
    overlay = {}
    for a in agents:
        assignment = next(iter(a._agent_context.get_or_create_model(CoalitionModel)._assignments.values()))
        overlay[assignment.part_id] = [n[0] for n in assignment.neighbors]
    start_time = time.time()
    agents[0].add_role(
        CohdaNegotiationStarterRole(num_solution_points=num_solution_points, target_params=None))

    await wait_for_term(controller_agent)

    end_time = time.time()

    final_memory = next(iter(agents[0].roles[0]._cohda.values()))._memory
    for a in agents:
        assert final_memory == next(iter(a.roles[0]._cohda.values()))._memory

    await container.shutdown()

    return final_memory, end_time - start_time, schedules_per_agent, overlay


async def wait_for_term(controller_agent):
    while len(controller_agent.roles[0]._weight_map.values()) != 1 or list(controller_agent.roles[0]._weight_map.values())[0] != 1:
        await asyncio.sleep(0.01)
    print('Terminated!')


async def wait_for_coalition_built(agents):
    for agent in agents:
        while len(agent._agent_context.get_or_create_model(CoalitionModel)._assignments) < 1:
            await asyncio.sleep(0.1)

if __name__ == '__main__':
    asyncio.run(simulate_schaffer('Schaffer_1', None))
