"""Module for distributed real power planning with COHDA. Contains roles, which
integrate COHDA in the negotiation system and the core COHDA-decider together with its model.
"""
import asyncio
import json
import random
from typing import Dict, List, Any, Tuple, Callable
from typing import Optional

import numpy as np
from evoalgos.selection import HyperVolumeContributionSelection
from mango.messages.codecs import json_serializable

from mango_library.coalition.core import CoalitionAssignment
from mango_library.negotiation.core import NegotiationParticipant, \
    NegotiationStarterRole, Negotiation
from mango_library.negotiation.multiobjective_cohda.data_classes import SolutionCandidate, WorkingMemory, \
    SystemConfig, ScheduleSelections, Target, SolutionPoint


@json_serializable
class CohdaMessage:
    """
    Message for a COHDa negotiation.
    Contains the working memory of an agent.
    """

    def __init__(self, working_memory: WorkingMemory):
        self._working_memory = working_memory

    @property
    def working_memory(self) -> WorkingMemory:
        """Return the working memory of the sender agent

        :return: the working memory of the sender
        """
        return self._working_memory


class CohdaNegotiationStarterRole(NegotiationStarterRole):
    """
    Convienience role for starting a COHDA negotiation
    """

    def __init__(self, target_params, num_solution_points, coalition_model_matcher=None, coalition_uuid=None) -> None:
        """

        :param target_params: Parameter that are necessary for the agents to calculate the performance.
        Could be e.g. the target schedule.
        :param coalition_model_matcher:
        :param coalition_uuid:
        """
        super().__init__(
            lambda assignment:
            CohdaMessage(WorkingMemory(
                target_params=target_params,
                system_config=SystemConfig(schedule_choices={}, num_solution_points=num_solution_points),
                solution_candidate=SolutionCandidate(agent_id=assignment.part_id, schedules={},
                                                     num_solution_points=num_solution_points))),
            coalition_model_matcher=coalition_model_matcher, coalition_uuid=coalition_uuid
        )


class COHDA:
    """
    COHDA-decider
    """

    def __init__(self, *, schedule_provider,
                 is_local_acceptable: Callable,
                 part_id: str,
                 perf_func: Callable,
                 reference_point: Tuple,
                 num_iterations: int,
                 pick_func: Callable = None,
                 mutate_func: Callable = None):
        """

        :param schedule_provider:
        :param is_local_acceptable:
        :param part_id:
        :param perf_func:
        :param reference_point:
        :param num_iterations:
        :param pick_func:
        :param mutate_func
        """
        self._schedule_provider = schedule_provider
        self._is_local_acceptable = is_local_acceptable
        self._part_id = part_id
        self._ref_point = reference_point
        empty_candidate = SolutionCandidate(agent_id=self._part_id,
                                            schedules={},
                                            hypervolume=float('-inf'),
                                            perf=None,
                                            num_solution_points=0)
        # create an empty working memory
        self._memory = WorkingMemory(
            target_params=None, solution_candidate=empty_candidate,
            system_config=SystemConfig(schedule_choices={}, num_solution_points=0)
        )
        self._counter = 0
        self._perf_func = perf_func
        self._num_iterations = num_iterations
        self._pick_func = pick_func if pick_func is not None else self.pick_all_points
        self._mutate_func = mutate_func if mutate_func is not None else self.mutate_with_all_possible
        self._selection = HyperVolumeContributionSelection(prefer_boundary_points=False)
        self._selection.construct_ref_point = self.construct_ref_point

    def construct_ref_point(self, solution_points, offsets=None):
        # ref point is given, but possible solution calculation could be here
        return self._ref_point

    @staticmethod
    def pick_all_points(solution_points: List[SolutionPoint]) -> List[SolutionPoint]:
        return solution_points

    @staticmethod
    def pick_random_point(solution_points: List[SolutionPoint]) -> List[SolutionPoint]:
        return [random.choice(solution_points)]

    @staticmethod
    def mutate_with_one_random(solution_point: SolutionPoint, schedule_creator, agent_id, perf_func, target_params) \
            -> List[SolutionPoint]:
        new_schedule = random.choice(schedule_creator())
        new_cs = np.copy(solution_point.cluster_schedule)
        new_cs[solution_point.idx[agent_id]] = new_schedule
        new_perf = perf_func([new_cs], target_params)[0]
        return [SolutionPoint(cluster_schedule=new_cs, performance=new_perf, idx=solution_point.idx)]

    @staticmethod
    def mutate_with_all_possible(solution_point: SolutionPoint, schedule_creator, agent_id, perf_func, target_params) \
            -> List[SolutionPoint]:
        possible_schedules = schedule_creator()
        new_solution_points = []
        for new_schedule in possible_schedules:
            new_cs = np.copy(solution_point.cluster_schedule)
            new_cs[solution_point.idx[agent_id]] = new_schedule
            new_perf = perf_func([new_cs], target_params)[0]
            new_solution_points.append(SolutionPoint(cluster_schedule=new_cs, performance=new_perf,
                                                     idx=solution_point.idx))
        return new_solution_points

    def handle_cohda_msgs(self, messages: List[CohdaMessage]) -> Optional[CohdaMessage]:
        """
        This called by the COHDARole. It takes a List of COHDA messages, executes perceive, decide, act and returns
        a CohdaMessage in case the working memory has changed and None otherwise
        :param messages: The list of received CohdaMessages
        :return: The message to be sent to the neighbors, None if no message has to be sent
        """
        old_sysconf = self._memory.system_config
        old_candidate = self._memory.solution_candidate

        # perceive
        sysconf, candidate = self._perceive(messages)

        # decide
        if sysconf is not old_sysconf or candidate is not old_candidate:
            sysconf, candidate = self._decide(sysconfig=sysconf, candidate=candidate)
            # act
            return self._act(new_sysconfig=sysconf, new_candidate=candidate)
        else:
            return None

    def _perceive(self, messages: List[CohdaMessage]) -> Tuple[SystemConfig, SolutionCandidate]:
        """
        Updates the current knowledge
        :param messages: The List of received CohdaMessages
        :return: a tuple of SystemConfig, Candidate as a result of perceive
        """
        current_sysconfig = None
        current_candidate = None
        for message in messages:
            if self._memory.target_params is None:
                # get target parameters if not known
                self._memory.target_params = message.working_memory.target_params

            if current_sysconfig is None:
                if self._part_id not in self._memory.system_config.schedule_choices:
                    # if you have not yet selected any schedule in the sysconfig, choose any to start with
                    schedule_choices = self._memory.system_config.schedule_choices
                    num_solution_points = message.working_memory.system_config.num_solution_points
                    inital_schedules = [self._schedule_provider()[0] for _ in range(num_solution_points)]
                    schedule_choices[self._part_id] = ScheduleSelections(
                        np.array(inital_schedules), self._counter + 1)
                    self._counter += 1
                    # we need to create a new class of Systemconfig so the updates are
                    # recognized in handle_cohda_msgs()
                    current_sysconfig = SystemConfig(schedule_choices=schedule_choices,
                                                     num_solution_points=num_solution_points)
                else:
                    current_sysconfig = self._memory.system_config

            if current_candidate is None:
                if self._part_id not in self._memory.solution_candidate.schedules:
                    # if you have not yet selected any schedule in the sysconfig, choose any to start with
                    schedules = self._memory.solution_candidate.schedules
                    num_solution_points = message.working_memory.system_config.num_solution_points
                    inital_schedules = [self._schedule_provider()[0] for _ in range(num_solution_points)]
                    schedules[self._part_id] = np.array(inital_schedules)
                    # we need to create a new class of SolutionCandidate so the updates are
                    # recognized in handle_cohda_msgs()
                    current_candidate = SolutionCandidate(agent_id=self._part_id, schedules=schedules,
                                                          num_solution_points=num_solution_points)
                    current_candidate.perf = self._perf_func(current_candidate.cluster_schedules,
                                                             self._memory.target_params)

                    performances = current_candidate.perf
                    current_candidate.hypervolume = self.get_hypervolume(performances)
                else:
                    current_candidate = self._memory.solution_candidate

            new_sysconf = message.working_memory.system_config
            new_candidate = message.working_memory.solution_candidate

            # Merge new information into current_sysconfig and current_candidate
            current_sysconfig = self._merge_sysconfigs(sysconfig_i=current_sysconfig, sysconfig_j=new_sysconf)
            current_candidate = self._merge_candidates(
                candidate_i=current_candidate, candidate_j=new_candidate, agent_id=self._part_id,
                perf_func=self._perf_func, target_params=self._memory.target_params)

        return current_sysconfig, current_candidate

    def _decide(self, sysconfig: SystemConfig, candidate: SolutionCandidate) -> Tuple[SystemConfig, SolutionCandidate]:
        """
        Check whether a better SolutionCandidate can be created based on the current state of the negotiation
        :param sysconfig: Current SystemConfig
        :param candidate: Current SolutionCandidate
        :return: Tuple of SystemConfig, SolutionCandidate. Unchanged to parameters if no new SolutionCandidate was
        found. Else it consists of the new SolutionCandidate and an updated SystemConfig
        """
        current_best_candidate = candidate

        for iteration in range(self._num_iterations):
            candidate_from_sysconfig: SolutionCandidate = \
                SolutionCandidate.create_from_sysconf(sysconfig=sysconfig, agent_id=self._part_id)
            candidate_from_sysconfig.perf = self._perf_func(candidate_from_sysconfig.cluster_schedules,
                                                            self._memory.target_params)
            all_solution_points = candidate_from_sysconfig.solution_points

            # pick solution points to mutate
            solution_points_to_mutate = self._pick_func(solution_points=candidate_from_sysconfig.solution_points)
            # execute mutate for all solution points
            for solution_point in solution_points_to_mutate:
                # add new solution points to list of all solution points
                new_solution_points = self._mutate_func(
                    solution_point=solution_point, agent_id=self._part_id, perf_func=self._perf_func,
                    target_params=self._memory.target_params, schedule_creator=self._schedule_provider)
                all_solution_points.extend(new_solution_points)

            self._selection.reduce_to(population=all_solution_points, number=candidate.num_solution_points)

            # calculate hypervolume of new front
            new_hyper_volume = self.get_hypervolume(performances=[ind.objective_values for ind in all_solution_points])

            print(
                f'Candidate after decide:\nPerformance: '
                f'{sorted([(round(ind.objective_values[0], 2), round(ind.objective_values[1], 2)) for ind in all_solution_points], key=lambda l: l[0])}\n'
                f'Hypervolume: {round(new_hyper_volume, 4)}')

            # if new is better than current, exchange current
            if new_hyper_volume > current_best_candidate.hypervolume:
                idx = solution_points_to_mutate[0].idx
                new_schedule_dict = {aid: [] for aid in idx.keys()}
                new_perf = []
                for individual in all_solution_points:
                    new_perf.append(individual.objective_values)
                    for aid, cs_idx in idx.items():
                        new_schedule_dict[aid].append(individual.cluster_schedule[cs_idx])
                for aid in idx.keys():
                    new_schedule_dict[aid] = np.array(new_schedule_dict[aid])

                current_best_candidate = SolutionCandidate(schedules=new_schedule_dict, hypervolume=new_hyper_volume,
                                                           perf=new_perf, agent_id=self._part_id,
                                                           num_solution_points=candidate.num_solution_points)

            # change own schedules choices in sysconf if they differ from candidate
            schedules_in_candidate = current_best_candidate.schedules.get(self._part_id, None)
            schedule_choices_in_sysconfig = sysconfig.schedule_choices.get(self._part_id, None)
            if schedule_choices_in_sysconfig is None or \
                    not np.array_equal(schedules_in_candidate, schedule_choices_in_sysconfig.schedules):
                # update Sysconfig if your schedule in the current sysconf is different to the one in the candidate
                sysconfig.schedule_choices[self._part_id] = ScheduleSelections(
                    schedules=schedules_in_candidate, counter=self._counter + 1)
                # update counter
                self._counter += 1

            return sysconfig, current_best_candidate

    def _act(self, new_sysconfig: SystemConfig, new_candidate: SolutionCandidate) -> CohdaMessage:
        """
        Stores the new SystemConfig and SolutionCandidate in Memory and returns the COHDA message that should be sent
        :param new_sysconfig: The SystemConfig as a result from perceive and decide
        :param new_candidate: The SolutionCandidate as a result from perceive and decide
        :return: The COHDA message that should be sent
        """
        # update memory
        self._memory.system_config = new_sysconfig
        self._memory.solution_candidate = new_candidate
        # return COHDA message
        return CohdaMessage(working_memory=self._memory)

    @staticmethod
    def _merge_sysconfigs(sysconfig_i: SystemConfig, sysconfig_j: SystemConfig):
        """
        Merge *sysconf_i* and *sysconf_j* and return the result.

        Returns a merged systemconfig. If the sysconfig_i remains unchanged, the same instance of sysconfig_i is
        returned, otherwise a new object is created.
        """

        sysconfig_i_schedules: Dict[str, ScheduleSelections] = sysconfig_i.schedule_choices
        sysconfig_j_schedules: Dict[str, ScheduleSelections] = sysconfig_j.schedule_choices
        key_set_i = set(sysconfig_i_schedules.keys())
        key_set_j = set(sysconfig_j_schedules.keys())

        new_sysconfig: Dict[str, ScheduleSelections] = {}
        modified = False

        for i, a in enumerate(sorted(key_set_i | key_set_j)):
            # An a might be in key_set_i, key_set_j or in both!
            if a in key_set_i and \
                    (a not in key_set_j or sysconfig_i_schedules[a].counter >= sysconfig_j_schedules[a].counter):
                # Use data of sysconfig_i
                schedule_selections = sysconfig_i_schedules[a]
            else:
                # Use data of sysconfig_j
                schedule_selections = sysconfig_j_schedules[a]
                modified = True

            new_sysconfig[a] = schedule_selections

        if modified:
            sysconf = SystemConfig(new_sysconfig, num_solution_points=sysconfig_i.num_solution_points)
        else:
            sysconf = sysconfig_i

        return sysconf

    def get_hypervolume(self, performances):
        self._selection.sorting_component.hypervolume_indicator.reference_point = self._ref_point
        return self._selection.sorting_component.hypervolume_indicator. \
            assess_non_dom_front(performances)

    def _merge_candidates(self, candidate_i: SolutionCandidate, candidate_j: SolutionCandidate, agent_id: str,
                          perf_func, target_params=None):
        """
        Merge *candidate_i* and *candidate_j* and return the result.

        Returns a merged Candidate. If the scandidate_i remains unchanged, the same instance of candidate_i is
        returned, otherwise a new object is created with agent_id as candidate.agent_id
        :param candidate_i: First Candidate
        :param candidate_j: Second Candidate
        :param agent_id: Id of the agent that executes merge
        :param perf_func: Performance Function
        :param reference_point: Reference Point
        :return: An Instance of SolutionCandidate
        """

        key_set_i = set(candidate_i.schedules.keys())
        key_set_j = set(candidate_j.schedules.keys())

        candidate = candidate_i  # Default candidate is *i*

        if key_set_i < key_set_j:
            # Use *j* if *K_i* is a true subset of *K_j*
            candidate = candidate_j

        elif key_set_i == key_set_j:

            # Compare the performance if the key_sets are equal
            if candidate_j.hypervolume > candidate_i.hypervolume:
                # Choose *j* if it performs better
                candidate = candidate_j

            elif candidate_j.hypervolume == candidate_i.hypervolume:
                # If both perform equally well, order them by agent_name
                if candidate_j.agent_id < candidate_i.agent_id:
                    candidate = candidate_j

        # Key sets are not equal and key_set_i is NOT a true subset of key_set_j
        elif key_set_j - key_set_i:
            # If there are elements in key_set_j but not in key_set_i,
            # update *candidate_i*

            new_candidate: Dict[str, np.array] = {}
            for a in set(key_set_i | key_set_j):
                if a in key_set_i:
                    data = candidate_i.schedules[a]
                else:
                    data = candidate_j.schedules[a]
                new_candidate[a] = data

            # create new candidate
            candidate = SolutionCandidate(schedules=new_candidate,
                                          agent_id=agent_id, perf=None,
                                          hypervolume=None,
                                          num_solution_points=candidate_i.num_solution_points)
            # calculate and set perf
            candidate.perf = perf_func(candidate.cluster_schedules, target_params=target_params)
            # calculate and set hypervolume
            candidate.hypervolume = self.get_hypervolume(candidate.perf)

        return candidate


class MultiObjectiveCOHDARole(NegotiationParticipant):
    """Negotiation role for COHDA.
    """

    def __init__(self, *, schedule_provider, targets: List[Target], num_solution_points: int,
                 local_acceptable_func=None, check_inbox_interval: float = 0.1,
                 pick_func=None, mutate_func=None, num_iterations: int = 1, ):
        super().__init__()

        self._schedule_provider = schedule_provider
        self._is_local_acceptable = local_acceptable_func if local_acceptable_func is not None else lambda x: True
        self._perf_functions = [target.performance for target in targets]
        self._reference_point = tuple(
            [target.ref_point for target in targets])
        self._num_solution_points = num_solution_points
        self._cohda = {}
        self._cohda_msg_queues = {}
        self._cohda_tasks = []
        self._num_iterations = num_iterations
        self._check_inbox_interval = check_inbox_interval
        self._pick_func = pick_func
        self._mutate_func = mutate_func

    def create_cohda(self, part_id: str):
        """
        Create an instance of COHDA.
        :param part_id: participant id
        :return: COHDA object
        """

        return COHDA(schedule_provider=self._schedule_provider,
                     is_local_acceptable=self._is_local_acceptable,
                     part_id=part_id,
                     reference_point=self._reference_point,
                     perf_func=self._perf_func,
                     num_iterations=self._num_iterations,
                     pick_func=self._pick_func,
                     mutate_func=self._mutate_func)

    def _perf_func(self, cluster_schedules: List[np.array], target_params) -> List[Tuple]:
        """

        :param cluster_schedules:
        :return:
        """
        performances = []
        for cs in cluster_schedules:
            perf_of_current = []
            for perf_func in self._perf_functions:
                perf_of_current.append(perf_func(cs, target_params))
            performances.append(tuple(perf_of_current))

        return performances

    async def on_stop(self) -> None:
        """
        Will be called once the agent is shutdown
        """
        # cancel all cohda tasks
        for task in self._cohda_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    def handle(self,
               message,
               assignment: CoalitionAssignment,
               negotiation: Negotiation,
               meta: Dict[str, Any]):

        if negotiation.coalition_id in self._cohda:
            negotiation.active = True
            self._cohda_msg_queues[negotiation.coalition_id].append(message)
        else:
            self._cohda[negotiation.coalition_id] = self.create_cohda(assignment.part_id)
            self._cohda_msg_queues[negotiation.coalition_id] = [message]

            async def process_msg_queue():
                """
                Method to evaluate all incoming message of a cohda_message_queue for a certain negotiation
                """

                if len(self._cohda_msg_queues[negotiation.coalition_id]) > 0:
                    # get queue
                    cohda_message_queue, self._cohda_msg_queues[negotiation.coalition_id] = \
                        self._cohda_msg_queues[negotiation.coalition_id], []

                    message_to_send = self._cohda[negotiation.coalition_id].handle_cohda_msgs(cohda_message_queue)

                    if message_to_send is not None:
                        await self.send_to_neighbors(assignment, negotiation, message_to_send)

                    else:
                        # set the negotiation as inactive as the incoming information was known already
                        negotiation.active = False
                else:
                    # set the negotiation as inactive as no message has arrived
                    negotiation.active = False

            self._cohda_tasks.append(self.context.schedule_periodic_task(process_msg_queue,
                                                                         delay=self._check_inbox_interval))
