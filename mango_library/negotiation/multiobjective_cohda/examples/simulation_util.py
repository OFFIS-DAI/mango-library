import asyncio
import os
import random
import time
from copy import deepcopy
from typing import List, Callable, Dict, Any

import h5py
import numpy as np
from mango import RoleAgent
from mango import create_container
from mango.messages.codecs import JSON
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.result import Result
from pymoo.optimize import minimize
from pymoo.problems import get_problem

from mango_library.coalition.core import (
    CoalitionParticipantRole,
    CoalitionInitiatorRole,
    CoalitionModel,
)
from mango_library.negotiation.multiobjective_cohda.cohda_messages import (
    MoCohdaNegotiationMessage,
)
from mango_library.negotiation.multiobjective_cohda.data_classes import Target
from mango_library.negotiation.multiobjective_cohda.mocohda_starting import (
    MoCohdaNegotiationDirectStarterRole,
)
from mango_library.negotiation.multiobjective_cohda.multiobjective_cohda import (
    MultiObjectiveCOHDARole,
    MoCohdaNegotiationModel,
)
from mango_library.negotiation.termination import (
    NegotiationTerminationParticipantRole,
    NegotiationTerminationDetectorRole,
)
from mango_library.negotiation.util import multi_objective_serializers

CODEC = JSON()
for serializer in multi_objective_serializers:
    CODEC.add_serializer(*serializer())


def store_in_db(
        *,
        db_file: str,
        sim_name: str,
        n_agents: int,
        targets: List[Target],
        n_solution_points: int,
        check_inbox_interval: float,
        mutate_func: Callable,
        pick_func: Callable,
        n_iterations: int,
        results: List[Dict],
):
    """
    Function that creates a hdf5 file which stores the simulation configurations together with the simulation results.
    CAUTION: Any existing file with the same name as provided in 'db_file' will be overwritten
    :param db_file: The db_file with the ending '.hdf5' (e.g. 'Testfile.hdf5')
    :param sim_name: The name of the simulation (e. g. 'Schaffer_1')
    :param n_agents: The number of agents that were part of the simulation
    :param targets: The List of Targets
    :param n_solution_points: The number of SolutionPoints
    :param check_inbox_interval: The check_inbox_intervall
    :param mutate_func: The mutate function that the agents use within decide()
    :param pick_func: The pick function that the agents use within decide()
    :param n_iterations: The number of iteration of the agents within decide()
    :param results: The simulation results as returned from simulate_mo_cohda
    """
    if len(results) == 0:
        print("Cannot store anything in db as results dict is empty.")
        return

    # open the file
    with h5py.File(db_file, "w") as f:

        if os.path.isfile("UnitAgent_0.h5"):
            # open updates from agents, if those are stored
            agent_updates = [f"UnitAgent_{i}.h5" for i in range(n_agents)]

            for agent_name in agent_updates:
                file = h5py.File(agent_name, "a")
                groups = [key for key in file.keys()]
                groups.sort()
                for key in groups:
                    file.copy(key, f, name=f"{agent_name[:-2]}_{key}")
                os.remove(agent_name)

        # General group
        general_group = f.create_group("General infos")

        # General Infos dataset
        dtype_general = np.dtype(
            [
                ("Name", "S100"),
                ("n_agents", "uint64"),
                ("n_objectives", "uint64"),
                ("n_solution_points", "uint64"),
                ("n_iterations", "uint64"),
                ("msg_queue_interval", "float64"),
                ("mutate_func", "S100"),
                ("pick_func", "S100"),
            ]
        )
        data_general = np.array(
            [
                (
                    sim_name,
                    n_agents,
                    len(targets),
                    n_solution_points,
                    n_iterations,
                    check_inbox_interval,
                    mutate_func.__name__,
                    pick_func.__name__,
                )
            ],
            dtype=dtype_general,
        )
        general_group.create_dataset("general_info", data=data_general)

        # Targets dataset
        dtype_targets = np.dtype([("Function", "S100"), ("Ref Point", "float64")])
        data_targets = np.array(
            [(t._target_function.__doc__, t.ref_point) for t in targets],
            dtype=dtype_targets,
        )
        general_group.create_dataset("targets", data=data_targets)

        # Schedules dataset
        dtype_schedules = [("aid", "S100")]
        schedules_per_agent = results[0]["schedules"]
        max_num_schedules = max([len(s) for s in schedules_per_agent.values()])

        len_of_schedules = len(list(schedules_per_agent.values())[0][0])
        for i in range(max_num_schedules):
            for j in range(len_of_schedules):
                dtype_schedules.append((f"schedule_{i}[{j}]", "float64"))
        dtype_schedules = np.dtype(dtype_schedules)
        data_schedules = []

        for results_dict in results:
            assert schedules_per_agent == results_dict["schedules"]

        for aid, sched_list in schedules_per_agent.items():
            single_point_schedules = [
                s[pos] for s in sched_list for pos in range(len_of_schedules)
            ]
            missing_schedules = max_num_schedules - len(single_point_schedules)
            data_schedules.append(
                tuple([aid] + single_point_schedules + [np.nan] * missing_schedules)
            )
        data_schedules = np.array(data_schedules, dtype=dtype_schedules)
        general_group.create_dataset("Schedules", data=data_schedules)

        # Results group
        results_group = f.create_group("Results")

        # Go through all results in result dict
        for sim_no, results_dict in enumerate(results):
            # General Results dataset
            sim_results_grp = results_group.create_group(f"Results_{sim_no}")
            solution_candidate = results_dict["final_memory"].solution_candidate
            dtype_general_result = np.dtype(
                [("Duration", "float64"), ("Hypervolume", "float64")]
            )
            data_general_results = np.array(
                [(results_dict["duration"], solution_candidate.hypervolume)],
                dtype=dtype_general_result,
            )
            sim_results_grp.create_dataset("general results", data=data_general_results)

            # Performance dataset
            dtype_performances = np.dtype(
                [(f"Performance_{i}", "float64") for i, _ in enumerate(targets)]
            )
            data_perf = np.array(
                sorted(solution_candidate.perf), dtype=dtype_performances
            )
            sim_results_grp.create_dataset("performances", data=data_perf)

            # Overlay dataset
            dtype_neighbors = [("part_id", "S100")]
            max_len_of_neighbors = max(
                len(neighbors) for neighbors in results_dict["overlay"].values()
            )
            for i in range(max_len_of_neighbors):
                dtype_neighbors.append((f"neighbor_{i}", "S100"))
            dtype_neighbors = np.dtype(dtype_neighbors)
            data_neighbors = []
            for part_id, neighbors in results_dict["overlay"].items():
                num_empty_neighbors = len(dtype_neighbors) - 1 - len(neighbors)
                data_neighbors.append(
                    (part_id,) + (*neighbors,) + ((None,) * num_empty_neighbors)
                )
            data_neighbors = np.array(data_neighbors, dtype=dtype_neighbors)
            sim_results_grp.create_dataset("overlay", data=data_neighbors)

            # Solution Points datasets
            dtype_solution_points = [("part_id", "S100")]
            for i in range(len_of_schedules):
                dtype_solution_points.append(((f"value_{i}", "float64")))
            dtype_solution_points = np.dtype(dtype_solution_points)

            for i, solution_point in enumerate(
                    sorted(solution_candidate.solution_points)
            ):
                data_solution_points = []
                for part_id, index in solution_point.idx.items():
                    data_solution_points.append(
                        (part_id,) + tuple(solution_point.cluster_schedule[index])
                    )
                data_solution_points = np.array(
                    data_solution_points, dtype=dtype_solution_points
                )
                sim_results_grp.create_dataset(
                    f"Solutionpoint_{i}", data=data_solution_points
                )


async def simulate_mo_cohda_NSGA2(*, possible_interval: float, num_agents: int, targets: List[Target],
                                  num_solution_points: int,
                                  pick_func: Callable, mutate_func: Callable,
                                  num_iterations: int, check_inbox_interval: float, topology_creator: Callable = None,
                                  num_simulations: int, problem='zdt3', lower_limit=0, upper_limit=1,
                                  store_updates_to_db: bool = False, sim_name: str = '', population_size: int = 1,
                                  control_all_variables=True):
    """
    Function that will execute a multi-objective simulation and return a dict consisting of the results.
    :param num_agents: The number of agents
        if schedules_all_equal == True, the parameter should be a list of schedules which will then be
            assigned to all agents
        if schedules_all_equal == False, the parameter should be a list of lists of schedules. List i will then be
        assigned to agent i. So len(possible_schedules) should be equal to num_agents.
    :param possible_interval: The possible interval of values for each agent.
    :param targets: The List of targets as defined in multiobjective_cohda.data_classes.Target
    :param num_solution_points: Number of solution points to describe the pareto frong
    :param pick_func: The pick function that all agents will use
    :param mutate_func: The mutate function that all agents will use
    :param num_iterations: The number of iterations that all agents will execute in decide()
    :param check_inbox_interval: The interval which the agents use to interpret new messages
    :param topology_creator: The method that the CoalitionInitiator uses to connect the neighbors. If it is not
            provided, the default method within the CoalitionInitiatorRole is taken
            see: .coalition.core.CoalitionInitiatorRole
    :param num_simulations: The number of simulations to execute
    :param problem: The problem to simulate, e.g. ztd-3.
    :param lower_limit: The lower limit of the problem to simulate.
    :param upper_limit: The upper limit of the problem to simulate.
    :param store_updates_to_db: If true, all updates of the agents are stored during the simulation.
    :param sim_name: Name of the simulation, also used as database name.
    :param population_size: Size of the population to simulate.
    :param control_all_variables: If true, each agent is able to control all variables of a given level. If false,
    agent can only control one variable.
    :return: A List of dictionaries, each of which is structured as follows:
        'final_memory': WorkingMemory,
        'duration': float,
        'schedules': Dict[str, List[np.array]],
        'overlay': Dict[str, List[str]]
    """
    # problem, algorithm
    results = []
    algorithm = NSGA2(pop_size=population_size)

    if topology_creator is None:
        # if no topology creator is defined, use a small world ring topology
        def build_small_world_ring_topology(unit_agents: List, k=1, w=0.3) -> dict:
            """
            Builds a small world ring topology with neighbors in a distance of k and with random neighbors with the
            probability w
            :param unit_agents:
            :param k: maximum distance of connections in the ring
            :param w: probability of random connections
            """
            neighborhood: Dict = {}

            for agent in unit_agents:
                neighborhood[agent] = []

            # create the ring
            for index, agent in enumerate(unit_agents):
                for distance in range(1, k + 1):
                    neighborhood[agent].append(unit_agents[(index - distance)])
                    neighborhood[unit_agents[(index - distance)]].append(agent)

            # create a random connections
            for index, agent in enumerate(unit_agents):
                if random.random() < w:
                    random_agent = random.choice(unit_agents)
                    if (
                            random_agent not in neighborhood[agent]
                            and random_agent != agent
                    ):
                        neighborhood[agent].append(random_agent)
                        neighborhood[random_agent].append(agent)
            return neighborhood

        topology_creator = build_small_world_ring_topology
    for simulation_idx in range(num_simulations):
        port = 5555
        container = await create_container(addr=("127.0.0.2", port), codec=CODEC, copy_internal_messages=False)
        db_file = sim_name + '_simulation_idx_' + str(simulation_idx)
        agents = []  # Instance of agents
        addrs = []  # Tuples of addr, aid

        schedules_per_agent = (
            {}
        )  # will be filled and returned (for storing in database)
        overlay = {}  # # will be filled and returned (for storing in database)

        # create agents for negotiation
        for i in range(num_agents):
            a = RoleAgent(container, suggested_aid=f"UnitAgent_{i}")

            def provide_schedules(solution_point=None, agent_id=None, candidate=None):
                diff_to_lower_limit = 0
                diff_to_upper_limit = 0

                if isinstance(problem, str):
                    p = get_problem(problem)
                elif isinstance(problem, Problem):
                    p = problem
                else:
                    raise ValueError("Invalid type for problem given.")

                if solution_point is None:
                    if control_all_variables:
                        example_sum = [0. for _ in range(len(p.xl))]
                    else:
                        example_sum = [0]
                else:
                    # determine the sum of the cluster schedule
                    cluster_schedule = np.copy(solution_point.cluster_schedule)
                    # determine the schedule of the agent from the cluster schedule
                    agent_schedule = cluster_schedule[solution_point.idx[agent_id]]
                    # set agents partition to 0
                    agent_schedule = [0 for _ in agent_schedule]
                    cluster_schedule[solution_point.idx[agent_id]] = agent_schedule
                    example_sum = [sum(entry) for entry in zip(*cluster_schedule)]
                assert [lower_limit <= idx <= upper_limit for idx in example_sum]

                for idx in range(len(example_sum)):
                    # adapt the lower (xl) and upper (xu) limits of the algorithm by the sum of the cluster schedule
                    # without the partition of the current agent. Therefore, the solution without the agent is
                    # considered. The new partition of the agent will be added after the optimisation.

                    if example_sum[idx] - possible_interval >= lower_limit:
                        p.xl[idx] = example_sum[idx] - possible_interval
                    elif example_sum[idx] < lower_limit:
                        if lower_limit == 0:
                            raise ValueError('Somehow the sum is lower than lower limit!')
                        diff_to_lower_limit = lower_limit - example_sum[idx]
                        p.xl[idx] = diff_to_lower_limit
                    else:
                        diff = example_sum[idx] - lower_limit
                        p.xl[idx] = example_sum[idx] - diff

                    if example_sum[idx] + possible_interval <= upper_limit:
                        p.xu[idx] = example_sum[idx] + possible_interval
                    elif example_sum[idx] > upper_limit:
                        if example_sum[idx] < lower_limit:
                            raise ValueError(
                                "Value without agents participation is below lower and above higher limit!")
                        diff_to_upper_limit = example_sum[idx] - upper_limit
                        p.xu[idx] = diff_to_upper_limit
                    else:
                        diff = upper_limit - example_sum[idx]
                        p.xu[idx] = example_sum[idx] + diff

                    assert lower_limit <= p.xl[idx] <= upper_limit
                    assert lower_limit <= p.xu[idx] <= upper_limit

                # assert that the lower and upper limits between lower and upper limit
                assert all(lower_limit <= x <= upper_limit for x in p.xl)
                assert all(lower_limit <= x <= upper_limit for x in p.xu)

                # minimize problem
                result: Result = minimize(p, algorithm)
                solution = result.X.tolist()

                if control_all_variables:
                    # extract the actual partition of the current agent by determining what is contained in the solution
                    # additionally to the previously calculated sum of the current cluster schedule
                    for first_idx, sol_point in enumerate(solution):
                        for idx, single_val in enumerate(sol_point):
                            # single_val = solution with participation of agent
                            # example_sum = solution without participation of agent
                            # difference = participation of agent
                            if abs(example_sum[idx]) > abs(single_val):
                                diff = example_sum[idx] - single_val
                            else:
                                diff = single_val - example_sum[idx]

                            assert lower_limit <= single_val <= upper_limit
                            correct_value = float(diff)

                            if diff_to_lower_limit != 0:
                                correct_value += diff_to_lower_limit
                            if diff_to_upper_limit != 0:
                                correct_value += diff_to_upper_limit

                            solution[first_idx][idx] = correct_value

                    solution = [np.asarray(sol) for sol in solution]
                else:
                    # if agent only controls one variable, only choose this position
                    if agent_id is None:
                        agent_id = int(candidate.agent_id) - 1
                    else:
                        agent_id = int(agent_id) - 1
                    for idx in range(len(solution)):
                        agents_sol = solution[idx][agent_id]
                        solution[idx] = [0 if i != agent_id else agents_sol for i in range(len(solution[idx]))]
                return solution

            a.add_role(MultiObjectiveCOHDARole(
                schedule_provider=provide_schedules,
                targets=targets,
                local_acceptable_func=lambda s: True,
                num_solution_points=num_solution_points, num_iterations=num_iterations,
                check_inbox_interval=check_inbox_interval,
                pick_func=pick_func, mutate_func=mutate_func,
                store_updates_to_db=store_updates_to_db)
            )

            a.add_role(CoalitionParticipantRole())
            a.add_role(
                NegotiationTerminationParticipantRole(
                    negotiation_model_class=MoCohdaNegotiationModel,
                    negotiation_message_class=MoCohdaNegotiationMessage,
                )
            )

            agents.append(a)
            addrs.append((container.addr, a.aid))
            schedules_per_agent[a.aid] = provide_schedules(agent_id=i+1)
        # Controller agent will be a different agent, that is not part of the negotiation
        # Its tasks are creating a coalition and detecting the termination
        controller_agent = RoleAgent(container)
        controller_agent.add_role(NegotiationTerminationDetectorRole())
        controller_agent.add_role(
            CoalitionInitiatorRole(
                participants=addrs,
                details="",
                topic="",
                topology_creator=topology_creator,
            )
        )
        await asyncio.wait_for(wait_for_coalition_built(agents), timeout=5)
        print("Done building a coalition.")

        # fill the overlay dictionary
        for a in agents:
            assignment = next(
                iter(
                    a.roles[0]
                    .context.get_or_create_model(CoalitionModel)
                    ._assignments.values()
                )
            )
            overlay[assignment.part_id] = [n[0] for n in assignment.neighbors]

        # start the negotiation
        start_time = time.time()
        agents[0].add_role(
            MoCohdaNegotiationDirectStarterRole(
                num_solution_points=num_solution_points, target_params=None
            )
        )
        await wait_for_term(controller_agent)
        end_time = time.time()
        print("Negotiation terminated.")

        # get final memory of first agent
        final_memory = next(
            iter(
                agents[0]
                .roles[0]
                .context.get_or_create_model(MoCohdaNegotiationModel)
                ._negotiations.values()
            )
        )._memory

        # make sure all working memories are equal
        for a in agents:
            assert (
                    final_memory
                    == next(
                iter(
                    a.roles[0]
                    .context.get_or_create_model(MoCohdaNegotiationModel)
                    ._negotiations.values()
                )
            )._memory
            ), "Working memories of different agents are not equal."

        print("All working memories are equal!")
        # shutdown container
        await container.shutdown()

        # append results
        results.append(
            {
                "final_memory": final_memory,
                "duration": end_time - start_time,
                "schedules": schedules_per_agent,
                "overlay": overlay,
            }
        )
        # store to db
        store_in_db(
            db_file=db_file, sim_name=sim_name, n_agents=num_agents, targets=targets,
            n_solution_points=num_solution_points, n_iterations=num_iterations,
            check_inbox_interval=check_inbox_interval,
            mutate_func=mutate_func, pick_func=pick_func, results=results
        )
        results = []


async def simulate_mo_cohda(*, num_agents: int, possible_schedules: List, schedules_all_equal: bool = False,
                            targets: List[Target], num_solution_points: int, pick_func: Callable, mutate_func: Callable,
                            num_iterations: int, check_inbox_interval: float, topology_creator: Callable = None,
                            num_simulations: int, store_updates_to_db: bool = False, sim_name: str = ''):
    """
    Function that will execute a multi-objective simulation and return a dict consisting of the results.
    :param num_agents: The number of agents
    :param possible_schedules: The list of possible schedules per agent.
        if schedules_all_equal == True, the parameter should be a list of schedules which will then be
            assigned to all agents
        if schedules_all_equal == False, the parameter should be a list of lists of schedules. List i will then be
        assigned to agent i. So len(possible_schedules) should be equal to num_agents.
    :param schedules_all_equal: See above
    :param targets: The List of targets as defined in multiobjective_cohda.data_classes.Target
    :param num_solution_points: Number of solution points to describe the pareto frong
    :param pick_func: The pick function that all agents will use
    :param mutate_func: The mutate function that all agents will use
    :param num_iterations: The number of iterations that all agents will execute in decide()
    :param check_inbox_interval: The interval which the agents use to interpret new messages
    :param topology_creator: The method that the CoalitionInitiator uses to connect the neighbors. If it is not
            provided, the default method within the CoalitionInitiatorRole is taken
            see: .coalition.core.CoalitionInitiatorRole
    :param num_simulations: The number of simulations to execute
    :param store_updates_to_db: whether the agents should store each update into a database or not
    :param sim_name: Name of simulation
    :return: A List of dictionaries, each of which is structured as follows:
        'final_memory': WorkingMemory,
        'duration': float,
        'schedules': Dict[str, List[np.array]],
        'overlay': Dict[str, List[str]]
    """

    results = []
    if topology_creator is None:
        # if no topology creator is defined, use a small world ring topology
        def build_small_world_ring_topology(unit_agents: List, k=1, w=0.3) -> dict:
            """
            Builds a small world ring topology with neighbors in a distance of k and with random neighbors with the
            probability w
            :param unit_agents:
            :param k: maximum distance of connections in the ring
            :param w: probability of random connections
            """
            neighborhood: Dict = {}

            for agent in unit_agents:
                neighborhood[agent] = []

            # create the ring
            for index, agent in enumerate(unit_agents):
                for distance in range(1, k + 1):
                    neighborhood[agent].append(unit_agents[(index - distance)])
                    neighborhood[unit_agents[(index - distance)]].append(agent)

            # create a random connections
            for index, agent in enumerate(unit_agents):
                if random.random() < w:
                    random_agent = random.choice(unit_agents)
                    if (
                            random_agent not in neighborhood[agent]
                            and random_agent != agent
                    ):
                        neighborhood[agent].append(random_agent)
                        neighborhood[random_agent].append(agent)
            return neighborhood

        topology_creator = build_small_world_ring_topology

    for simulation_idx in range(num_simulations):
        port = 5555
        container = await create_container(addr=('127.0.0.2', port), codec=CODEC, copy_internal_messages=False)
        db_file = sim_name + '_simulation_idx_' + str(simulation_idx) + '.hdf5'
        agents = []  # Instance of agents
        addrs = []  # Tuples of addr, aid

        schedules_per_agent = {}  # will be filled and returned (for storing in database)
        overlay = {}  # # will be filled and returned (for storing in database)
        # create agents for negotiation
        for i in range(num_agents):
            a = RoleAgent(container, suggested_aid=f"UnitAgent_{i}")

            def provide_schedules(index):
                # we need an inline function here, otherwise to let the lamda functions actually point to
                # different places within the list
                if not schedules_all_equal:
                    return deepcopy(lambda: possible_schedules[index])
                else:
                    return lambda: possible_schedules

            a.add_role(MultiObjectiveCOHDARole(
                schedule_provider=provide_schedules(i),
                targets=targets,
                local_acceptable_func=lambda s: True,
                num_solution_points=num_solution_points, num_iterations=num_iterations,
                check_inbox_interval=check_inbox_interval,
                pick_func=pick_func, mutate_func=mutate_func,
                store_updates_to_db=store_updates_to_db)
            )

            # Fill dictionary with schedules per agent (only import for return values for database purposes)
            schedules_per_agent[a.aid] = provide_schedules(i)()

            a.add_role(CoalitionParticipantRole())
            a.add_role(
                NegotiationTerminationParticipantRole(
                    negotiation_model_class=MoCohdaNegotiationModel,
                    negotiation_message_class=MoCohdaNegotiationMessage,
                )
            )
            agents.append(a)
            addrs.append((container.addr, a.aid))

        # Controller agent will be a different agent, that is not part of the negotiation
        # Its tasks are creating a coalition and detecting the termination
        controller_agent = RoleAgent(container)
        controller_agent.add_role(NegotiationTerminationDetectorRole())
        controller_agent.add_role(
            CoalitionInitiatorRole(
                participants=addrs,
                details="",
                topic="",
                topology_creator=topology_creator,
            )
        )
        await asyncio.wait_for(wait_for_coalition_built(agents), timeout=5)
        print("Done building a coalition.")

        # fill the overlay dictionary
        for a in agents:
            assignment = next(
                iter(
                    a.roles[0]
                    .context.get_or_create_model(CoalitionModel)
                    ._assignments.values()
                )
            )
            overlay[assignment.part_id] = [n[0] for n in assignment.neighbors]

        # start the negotiation
        start_time = time.time()
        agents[0].add_role(
            MoCohdaNegotiationDirectStarterRole(
                num_solution_points=num_solution_points, target_params=None
            )
        )
        await wait_for_term(controller_agent)
        end_time = time.time()
        print("Negotiation terminated.")

        # get final memory of first agent
        final_memory = next(
            iter(
                agents[0]
                .roles[0]
                .context.get_or_create_model(MoCohdaNegotiationModel)
                ._negotiations.values()
            )
        )._memory

        # make sure all working memories are equal
        for a in agents:
            assert (
                    final_memory
                    == next(
                iter(
                    a.roles[0]
                    .context.get_or_create_model(MoCohdaNegotiationModel)
                    ._negotiations.values()
                )
            )._memory
            ), "Working memories of different agents are not equal."

        print("All working memories are equal!")
        # shutdown container
        await container.shutdown()

        # append results
        results.append(
            {
                "final_memory": final_memory,
                "duration": end_time - start_time,
                "schedules": schedules_per_agent,
                "overlay": overlay,
            }
        )

        # store to db
        store_in_db(
            db_file=db_file, sim_name=sim_name, n_agents=num_agents, targets=targets,
            n_solution_points=num_solution_points, n_iterations=num_iterations,
            check_inbox_interval=check_inbox_interval,
            mutate_func=mutate_func, pick_func=pick_func, results=results
        )
        results = []


async def wait_for_term(controller_agent):
    # Function that will return once the first weight map of the given agent equals to one
    while (
            len(controller_agent.roles[0]._weight_map.values()) != 1
            or list(controller_agent.roles[0]._weight_map.values())[0] != 1
    ):
        await asyncio.sleep(0.05)


async def wait_for_coalition_built(agents):
    # Function that will return once the given agent has one coalition assignment
    for agent in agents:
        while (
                len(agent.roles[0].context.get_or_create_model(CoalitionModel)._assignments)
                < 1
        ):
            await asyncio.sleep(0.1)
