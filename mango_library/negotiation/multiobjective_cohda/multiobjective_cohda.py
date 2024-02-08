"""Module for distributed real power planning with COHDA. Contains roles, which
integrate COHDA in the negotiation system and the core COHDA-decider together with its model.
"""
import asyncio
import inspect
import logging
import random
from typing import Dict, List, Any, Tuple, Callable, Optional
from uuid import UUID

import h5py
import numpy as np
from evoalgos.selection import HyperVolumeContributionSelection
from mango import Role

from mango_library.coalition.core import CoalitionAssignment, CoalitionModel
from mango_library.negotiation.multiobjective_cohda.cohda_messages import (
    MoCohdaNegotiationMessage,
    MoCohdaProposedSolutionMessage,
    MoCohdaFinalSolutionMessage,
    ConfirmMoCohdaSolutionMessage,
    StopNegotiationMessage,
    MoCohdaSolutionRequestMessage,
)
from mango_library.negotiation.multiobjective_cohda.data_classes import (
    SolutionCandidate,
    WorkingMemory,
    SystemConfig,
    ScheduleSelections,
    Target,
    SolutionPoint,
)

logger = logging.getLogger(__name__)


class MoCohdaNegotiation:
    """
    Multi Objective Cohda Negotiation
    """

    def __init__(self, *, schedule_provider,
                 is_local_acceptable: Callable,
                 part_id: str,
                 perf_func: Callable,
                 reference_point: Tuple,
                 num_iterations: int,
                 pick_func: Callable = None,
                 mutate_func: Callable = None,
                 use_fixed_ref_point: bool = True,
                 offsets: list = None,
                 target_params: dict = {}):
        """

        :param schedule_provider:
        :param is_local_acceptable:
        :param part_id:
        :param perf_func:
        :param reference_point:
        :param num_iterations:
        :param pick_func:
        :param mutate_func
        :param use_fixed_ref_point
        """

        def complete_schedule_provider(system_config: SystemConfig, candidate: SolutionCandidate,
                                       target_params: Dict, solution_point: SolutionPoint = None,
                                       agent_id: str = None, cs=None):
            schedule_provider_args = inspect.signature(schedule_provider).parameters.keys()
            args = {}
            if "candidate" in schedule_provider_args:
                args["candidate"] = candidate
            if "system_config" in schedule_provider_args:
                args["system_config"] = system_config
            if "target_params" in schedule_provider_args:
                args["target_params"] = target_params
            if "solution_point" in schedule_provider_args:
                args["solution_point"] = solution_point
            if "agent_id" in schedule_provider_args:
                args["agent_id"] = agent_id
            if "cs" in schedule_provider_args:
                args["cs"] = cs
            return schedule_provider(**args)

        self._schedule_provider = complete_schedule_provider
        self._is_local_acceptable = is_local_acceptable
        self._part_id = part_id
        empty_candidate = SolutionCandidate(agent_id=self._part_id,
                                            schedules={},
                                            hypervolume=float("-inf"),
                                            perf=None,
                                            num_solution_points=0)
        # create an empty working memory
        self._memory = WorkingMemory(
            target_params=target_params, solution_candidate=empty_candidate,
            system_config=SystemConfig(schedule_choices={}, num_solution_points=0)
        )
        self._counter = 0
        self._stopped = False  # is Ture once a StopNegotiationMessage is received for this negotiation
        # self._active is False once an iteration of the process_message_queue function has not received any message
        self._active = True
        self._perf_func = perf_func
        self._num_iterations = num_iterations
        self._pick_func = pick_func if pick_func is not None else self.pick_all_points
        self._mutate_func = mutate_func if mutate_func is not None else self.mutate_with_all_possible

        # if the fixed reference point is used, the method to calculate the reference point from the
        # hypervolume contribution selection is not used. Otherwise, the reference point will be calculated each
        # time the population is reduced.

        # Without a given ref point, it is possible to give offsets for the calculation of the reference point.
        self._selection = HyperVolumeContributionSelection(prefer_boundary_points=False, offsets=offsets)

        if use_fixed_ref_point:
            self._selection.construct_ref_point = self.construct_ref_point
            self._selection.sorting_component.hypervolume_indicator.reference_point = reference_point
            self._ref_point = reference_point
        else:
            self._ref_point = None

    @property
    def active(self) -> bool:
        """Is seen as active

        :return: True if active, False otherwise
        """
        return self._active

    @active.setter
    def active(self, is_active) -> None:
        """Set is active
        :param is_active: active
        """
        self._active = is_active

    @property
    def stopped(self) -> bool:
        """
        Is seen as stopped
        :return: True if stopped, False otherwise
        """
        return self._stopped

    @stopped.setter
    def stopped(self, is_stopped) -> None:
        """
        Set is stopped
        :param is_stopped: stopped
        """
        self._stopped = is_stopped

    def construct_ref_point(self, solution_points, offsets=None):
        """
        Method to construct the reference point according to the given solution points.
        """
        # In this case, the reference point is given, but a possible solution calculation could be placed here
        return self._ref_point

    @staticmethod
    def pick_all_points(solution_points: List[SolutionPoint]) -> List[SolutionPoint]:
        """
        Function that picks all solution points
        :param solution_points:
        :return:
        """
        return solution_points

    @staticmethod
    def pick_random_point(solution_points: List[SolutionPoint]) -> List[SolutionPoint]:
        """
        Function that picks one random point
        :param solution_points:
        :return:
        """
        return [random.choice(solution_points)]

    @staticmethod
    def mutate_with_one_random(solution_points: List[SolutionPoint], schedule_creator, agent_id, target_params) \
            -> List[SolutionPoint]:
        """
        Function that mutates each solution point with one random schedule that is different from the original
        :param solution_points:
        :param schedule_creator:
        :param agent_id:
        :return:
        """
        schedules = schedule_creator(system_config=None,
                                     candidate=None,
                                     target_params=target_params)
        if len(schedules) > 1:
            new_solution_points = []
            for solution_point in solution_points:
                schedule_before = solution_point.cluster_schedule[solution_point.idx[agent_id]]
                new_schedule = random.choice(schedules)
                while new_schedule == schedule_before:
                    new_schedule = random.choice(schedules)
                new_cs = np.copy(solution_point.cluster_schedule)
                new_cs[solution_point.idx[agent_id]] = new_schedule
                new_solution_points.append(SolutionPoint(cluster_schedule=new_cs, idx=solution_point.idx))
            return new_solution_points

        else:
            return solution_points

    @staticmethod
    def mutate_NSGA2(solution_points: List[SolutionPoint], schedule_creator, agent_id, target_params) \
            -> List[SolutionPoint]:
        """
        Function that mutates a solution point with all possible schedules
        :param solution_points:
        :param schedule_creator:
        :param agent_id:
        :return:
        """
        new_solution_points = []
        for solution_point in solution_points:
            new_schedules = schedule_creator(solution_point=solution_point, agent_id=agent_id,
                                             system_config=None,
                                             candidate=None,
                                             target_params=target_params)
            for new_schedule in new_schedules:
                new_cs = np.copy(solution_point.cluster_schedule)
                new_cs[solution_point.idx[agent_id]] = new_schedule
                new_solution_points.append(SolutionPoint(cluster_schedule=new_cs,
                                                         idx=solution_point.idx))
        return new_solution_points

    @staticmethod
    def mutate_with_all_possible(solution_points: List[SolutionPoint], schedule_creator, agent_id, target_params) \
            -> List[SolutionPoint]:
        """
        Function that mutates a solution point with all possible schedules
        :param target_params:
        :param solution_points:
        :param schedule_creator:
        :param agent_id:
        :return:
        """
        possible_schedules = schedule_creator(system_config=None,
                                              candidate=None,
                                              target_params=target_params, )
        new_solution_points = []
        for solution_point in solution_points:
            for new_schedule in possible_schedules:
                new_cs = np.copy(solution_point.cluster_schedule)
                new_cs[solution_point.idx[agent_id]] = new_schedule
                new_solution_points.append(SolutionPoint(cluster_schedule=new_cs,
                                                         idx=solution_point.idx))
        return new_solution_points

    def handle_cohda_msgs(self, messages: List[WorkingMemory]) -> Optional[WorkingMemory]:
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
            new_wm = self._act(new_sysconfig=sysconf, new_candidate=candidate)
            return new_wm
        else:
            return None

    def _perceive(self, messages: List[WorkingMemory]) -> Tuple[SystemConfig, SolutionCandidate]:
        """
        Updates the current knowledge
        :param messages: The List of received CohdaMessages
        :return: a tuple of SystemConfig, Candidate as a result of perceive
        """
        current_sysconfig = None
        current_candidate = None
        for working_memory in messages:
            if not self._memory.target_params or len(self._memory.target_params.keys()) == 0:
                # get target parameters if not known
                self._memory.target_params = working_memory.target_params
            if self._memory.target_params is None:
                self._memory.target_params = {}
            self._memory.update_target_params(working_memory.target_params)

            if current_sysconfig is None or current_candidate is None:
                if self._part_id not in self._memory.system_config.schedule_choices:
                    # if you have not yet selected any schedule in the sysconfig, choose any to start with
                    schedule_choices = self._memory.system_config.schedule_choices
                    num_solution_points = working_memory.system_config.num_solution_points
                    initial_schedules = self._schedule_provider(system_config=self._memory.system_config,
                                                                candidate=self._memory.solution_candidate,
                                                                target_params=self._memory.target_params)
                    num_initial_schedules = len(initial_schedules)
                    initial_schedules = [initial_schedules[n % num_initial_schedules] for n in
                                         range(num_solution_points)]

                    schedule_choices[self._part_id] = ScheduleSelections(
                        np.array(initial_schedules), self._counter + 1)
                    self._counter += 1
                    # we need to create a new class of Systemconfig so the updates are recognized in handle_cohda_msgs()
                    current_sysconfig = SystemConfig(schedule_choices=schedule_choices,
                                                     num_solution_points=num_solution_points)

                    if self._part_id not in self._memory.solution_candidate.schedules:
                        # if you have not yet selected any schedule in the sysconfig, choose any to start with
                        schedules = self._memory.solution_candidate.schedules
                        schedules[self._part_id] = np.array(initial_schedules)
                        # we need to create a new class of SolutionCandidate so the updates are
                        # recognized in handle_cohda_msgs()
                        current_candidate = SolutionCandidate(agent_id=self._part_id, schedules=schedules,
                                                              num_solution_points=num_solution_points)
                        target_params = self._memory.target_params if self._memory.target_params is not None else {}
                        target_params.update(
                            {"selected_schedule": current_candidate.schedules[current_candidate.agent_id]})
                        current_candidate.perf = self._perf_func(current_candidate.cluster_schedules,
                                                                 target_params)

                        performances = current_candidate.perf
                        current_candidate.hypervolume = self.get_hypervolume(performances,
                                                                             current_candidate.solution_points)
                else:
                    current_sysconfig = self._memory.system_config

                if current_candidate is None:
                    current_candidate = self._memory.solution_candidate

            new_sysconf = working_memory.system_config
            new_candidate = working_memory.solution_candidate

            # Merge new information into current_sysconfig and current_candidate
            current_sysconfig = self._merge_sysconfigs(sysconfig_i=current_sysconfig, sysconfig_j=new_sysconf)
            current_candidate = self._merge_candidates(
                candidate_i=current_candidate, candidate_j=new_candidate, agent_id=self._part_id,
                perf_func=self._perf_func, target_params=self._memory.target_params,
                get_hypervolume=self.get_hypervolume)

        return current_sysconfig, current_candidate

    def _decide(self, sysconfig: SystemConfig, candidate: SolutionCandidate) -> Tuple[SystemConfig, SolutionCandidate]:
        """
        Check whether a better SolutionCandidate can be created based on the current state of the negotiation
        :param sysconfig: Current SystemConfig
        :param candidate: Current SolutionCandidate
        :return: Tuple of SystemConfig, SolutionCandidate. Unchanged to parameters if no new SolutionCandidate was
        found. Else it consists of the new SolutionCandidate and an updated SystemConfig
        """
        # t_start_decide = time.time()
        current_best_candidate = candidate

        for iteration in range(self._num_iterations):
            candidate_from_sysconfig: SolutionCandidate = \
                SolutionCandidate.create_from_sysconf(sysconfig=sysconfig, agent_id=self._part_id)
            target_params = self._memory.target_params if self._memory.target_params is not None else {}
            target_params.update(
                {"selected_schedule": candidate_from_sysconfig.schedules[candidate_from_sysconfig.agent_id]})
            candidate_from_sysconfig.perf = self._perf_func(candidate_from_sysconfig.cluster_schedules,
                                                            target_params)
            all_solution_points = candidate_from_sysconfig.solution_points

            # pick solution points to mutate
            solution_points_to_mutate = self._pick_func(solution_points=candidate_from_sysconfig.solution_points)

            # execute mutate for all solution points
            # add new solution points to list of all solution points
            new_solution_points = self._mutate_func(
                solution_points=solution_points_to_mutate, agent_id=self._part_id,
                schedule_creator=self._schedule_provider, target_params=self._memory.target_params)

            for new_point in new_solution_points:
                target_params = self._memory.target_params if self._memory.target_params is not None else {}
                target_params.update({'selected_schedule': new_point.idx})
                new_perf = self._perf_func([new_point.cluster_schedule], target_params)[0]
                new_point.performance = new_perf

            all_solution_points.extend(new_solution_points)

            population_set = set(all_solution_points)
            num_unique_solution_points = len(population_set)

            if num_unique_solution_points > candidate.num_solution_points:
                diff = len(all_solution_points) - num_unique_solution_points
                if diff < candidate.num_solution_points:
                    # choose forward-greedy, because if there are less enough unique points than the difference between
                    # all solution points and the number to reduce to, with "backward-greedy", more solution points
                    # will be deleted and the number of solution points after reduce_to is smaller than
                    # candidate.num_solution_points
                    self._selection.selection_variant = "forward-greedy"
                self._selection.reduce_to(population=all_solution_points, number=candidate.num_solution_points)
                # reset selection variant
                self._selection.selection_variant = "auto"
            else:
                indices = [idx for idx, val in enumerate(all_solution_points) if val in all_solution_points[:idx]]
                for idx in reversed(indices):
                    if len(all_solution_points) > candidate.num_solution_points:
                        del all_solution_points[idx]
                    else:
                        break

            # calculate hypervolume of new front
            new_hyper_volume = self.get_hypervolume(performances=[ind.objective_values for ind in all_solution_points],
                                                    population=all_solution_points)

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

    def _act(self, new_sysconfig: SystemConfig, new_candidate: SolutionCandidate) -> WorkingMemory:
        """
        Stores the new SystemConfig and SolutionCandidate in Memory and returns the COHDA message that should be sent
        :param new_sysconfig: The SystemConfig as a result from perceive and decide
        :param new_candidate: The SolutionCandidate as a result from perceive and decide
        :return: The COHDA message that should be sent
        """
        # update memory
        self._memory.system_config = new_sysconfig
        self._memory.solution_candidate = new_candidate
        # return memory
        return self._memory

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

    def get_hypervolume(self, performances, population=None):
        if self._selection.sorting_component.reference_point is None:
            reference_point = self._selection.construct_ref_point(population, self._selection.offsets)
            self._selection.sorting_component.reference_point = reference_point
            self._ref_point = reference_point

        return self._selection.sorting_component.hypervolume_indicator. \
            assess_non_dom_front(performances)

    @staticmethod
    def _merge_candidates(candidate_i: SolutionCandidate, candidate_j: SolutionCandidate, agent_id: str,
                          perf_func, get_hypervolume, target_params=None):
        """
        Merge *candidate_i* and *candidate_j* and return the result.

        Returns a merged Candidate. If the scandidate_i remains unchanged, the same instance of candidate_i is
        returned, otherwise a new object is created with agent_id as candidate.agent_id
        :param candidate_i: First Candidate
        :param candidate_j: Second Candidate
        :param agent_id: Id of the agent that executes merge
        :param perf_func: Performance Function
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
            if target_params is None:
                target_params = {}
            target_params.update({'selected_schedule': candidate.schedules[candidate.agent_id]})
            candidate.perf = perf_func(candidate.cluster_schedules, target_params=target_params)
            # calculate and set hypervolume
            candidate.hypervolume = get_hypervolume(candidate.perf, candidate.solution_points)

        return candidate


class MultiObjectiveCOHDARole(Role):
    """Negotiation role for COHDA."""

    def __init__(self, *, schedule_provider, targets: List[Target], num_solution_points: int,
                 local_acceptable_func=None, check_inbox_interval: float = 0.1,
                 pick_func=None, mutate_func=None, num_iterations: int = 1,
                 use_fixed_ref_point: bool = True, offsets: list = None, store_updates_to_db: bool = False,
                 target_params: dict = {}):
        super().__init__()

        self._schedule_provider = schedule_provider
        self._is_local_acceptable = local_acceptable_func if local_acceptable_func is not None else lambda x: True
        self._perf_functions = [target.performance for target in targets]
        self._reference_point = tuple(
            [target.ref_point for target in targets])
        if self._reference_point is None or not use_fixed_ref_point:
            self._use_fixed_ref_point = False
        else:
            self._use_fixed_ref_point = True
        self._offsets = offsets
        self._num_solution_points = num_solution_points
        self._cohda_msg_queues = {}
        self._cohda_tasks: Dict[UUID, asyncio.Task] = {}  # stores the tasks that process the inbox
        self._num_iterations = num_iterations
        self._check_inbox_interval = check_inbox_interval
        self._pick_func = pick_func
        self._mutate_func = mutate_func
        self._store_updates_to_db = store_updates_to_db
        self._hf = None
        self._updates_iter = 0
        self._target_params = target_params

    def setup(self):
        # negotiation message
        self.context.subscribe_message(self, self.handle_neg_msg,
                                       lambda c, _: isinstance(c, MoCohdaNegotiationMessage))
        # stop negotiation message
        self.context.subscribe_message(self, self.handle_neg_stop,
                                       lambda c, _: isinstance(c, StopNegotiationMessage))
        # solution request message
        self.context.subscribe_message(self, self.handle_solution_request,
                                       lambda c, _: isinstance(c, MoCohdaSolutionRequestMessage))
        # final solution message
        self.context.subscribe_message(self, self.handle_cohda_solution_msg,
                                       lambda c, _: isinstance(c, MoCohdaFinalSolutionMessage))

    def create_cohda(self, part_id: str):
        """
        Create an instance of COHDA.
        :param part_id: participant id
        :return: COHDA object
        """

        return MoCohdaNegotiation(schedule_provider=self._schedule_provider,
                                  is_local_acceptable=self._is_local_acceptable,
                                  part_id=part_id,
                                  reference_point=self._reference_point,
                                  perf_func=self._perf_func,
                                  num_iterations=self._num_iterations,
                                  pick_func=self._pick_func,
                                  mutate_func=self._mutate_func,
                                  use_fixed_ref_point=self._use_fixed_ref_point,
                                  offsets=self._offsets,
                                  target_params=self._target_params)

    def _perf_func(self, cluster_schedules: List[np.array], target_params: Dict) -> List[Tuple]:
        """

        :param cluster_schedules:
        :return:
        """
        performances = []
        for cs in cluster_schedules:
            perf_of_current = []
            for perf_func in self._perf_functions:
                perf_of_current.append(perf_func(cs=cs, target_params=target_params))

            performances.append(tuple(perf_of_current))

        return performances

    async def on_stop(self) -> None:
        """
        Will be called once the agent is shutdown
        """
        # cancel all cohda tasks
        for task in self._cohda_tasks.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    def handle_neg_msg(self,
                       content: MoCohdaNegotiationMessage,
                       meta: Dict[str, Any]):
        # check if there is a Coalition with the coalition_ID
        if not self.context.get_or_create_model(CoalitionModel).exists(content.coalition_id):
            logger.warning(f'Received a CohdaNegotiationMessage with the coalition_id {content.coalition_id}'
                           f'but there is no such Coalition known.')
            return

        # get coalition_assignment
        coalition_assignment: CoalitionAssignment = self.context.get_or_create_model(
            CoalitionModel).by_id(content.coalition_id)

        # get negotiation model
        cohda_negotiation_model: MoCohdaNegotiationModel = self.context.get_or_create_model(MoCohdaNegotiationModel)
        if not cohda_negotiation_model.exists(content.negotiation_id):
            cohda_negotiation_model.add(
                negotiation_id=content.negotiation_id,
                cohda_negotiation=self.create_cohda(part_id=coalition_assignment.part_id))

        cohda_negotiation = cohda_negotiation_model.by_id(negotiation_id=content.negotiation_id)

        # add message to the queue if negotiation is not stopped
        if not cohda_negotiation.stopped:
            if content.negotiation_id in self._cohda_msg_queues.keys():
                cohda_negotiation.active = True
                self._cohda_msg_queues[content.negotiation_id].append(content.working_memory)
            else:
                self._cohda_msg_queues[content.negotiation_id] = [content.working_memory]
                self._cohda_tasks[content.negotiation_id] = self.context.schedule_periodic_task(
                    self.get_process_msg_queue_coro(
                        cohda_negotiation=cohda_negotiation, negotiation_id=content.negotiation_id,
                        coalition_assignment=coalition_assignment),
                    delay=self._check_inbox_interval)

    def get_process_msg_queue_coro(self, cohda_negotiation: MoCohdaNegotiation,
                                   coalition_assignment: CoalitionAssignment,
                                   negotiation_id: UUID):
        """
        Method that returns a coroutine that process a message queue of a specific COHDA negotiation.
        :param cohda_negotiation: the COHDA instance
        :param coalition_assignment: the corresponding coalition assignment
        :param negotiation_id: the corresponding negotiation ID
        :return: a coroutine without arguments that can be scheduled as periodic task
        """

        async def process_msg():
            if len(self._cohda_msg_queues[negotiation_id]) > 0 and not cohda_negotiation.stopped:
                # get queue
                cohda_message_queue, self._cohda_msg_queues[negotiation_id] = \
                    self._cohda_msg_queues[negotiation_id], []

                wm_to_send = cohda_negotiation.handle_cohda_msgs(cohda_message_queue)

                if wm_to_send is not None:
                    # send message to all neighbors
                    if self._store_updates_to_db:
                        self.store_update_in_db(wm_to_send)

                    for neighbor in coalition_assignment.neighbors:
                        self.context.schedule_instant_task(self.context.send_acl_message(
                            content=MoCohdaNegotiationMessage(
                                negotiation_id=negotiation_id,
                                coalition_id=coalition_assignment.coalition_id,
                                working_memory=wm_to_send,
                            ),
                            receiver_addr=neighbor[1], receiver_id=neighbor[2],
                            acl_metadata={'sender_addr': self.context.addr, 'sender_id': self.context.aid}))

            else:
                # set the negotiation as inactive as no message has arrived
                cohda_negotiation.active = False

        return process_msg

    def store_update_in_db(self, wm_to_send):
        self._hf = h5py.File(f'{self.context.aid}.h5', 'a')
        try:
            general_group = self._hf.create_group(f'Update_{self._updates_iter}')
        except ValueError:
            raise ValueError(
                'Group cannot be created. Make sure to delete old h5-Files before restarting optimization.')
        dtype_general_result = np.dtype([
            ('Hypervolume', 'float64')
        ])
        data_general_results = np.array([(wm_to_send.solution_candidate.hypervolume)],
                                        dtype=dtype_general_result)
        general_group.create_dataset('general results', data=data_general_results)

        # Performance dataset
        perf_list = []
        for i in wm_to_send.solution_candidate.perf[0]:
            if (f'Performance_{i}', 'float64') in perf_list:
                perf_list.append((f'Performance_{i}_', 'float64'))
            else:
                perf_list.append((f'Performance_{i}', 'float64'))

        dtype_performances = np.dtype(perf_list)
        data_perf = np.array(sorted(wm_to_send.solution_candidate.perf), dtype=dtype_performances)
        general_group.create_dataset('performances', data=data_perf)

        # Solution Points datasets
        dtype_solution_points = [('part_id', 'S100')]
        len_of_schedules = len(wm_to_send.solution_candidate.cluster_schedules[0][0])
        for i in range(len_of_schedules):
            dtype_solution_points.append(((f'value_{i}', 'float64')))
        dtype_solution_points = np.dtype(dtype_solution_points)
        for i, solution_point in enumerate(sorted(wm_to_send.solution_candidate.solution_points)):
            data_solution_points = []
            for part_id, index in solution_point.idx.items():
                data_solution_points.append((part_id,) + tuple(solution_point.cluster_schedule[index]))
            data_solution_points = np.array(data_solution_points, dtype=dtype_solution_points)
            general_group.create_dataset(f'Solutionpoint_{i}', data=data_solution_points)
        self._updates_iter += 1
        self._hf.close()

    def handle_neg_stop(self, content: StopNegotiationMessage, _):
        """ Is called once a StopNegotiationMessage arrived
        """
        if content.negotiation_id in self._cohda_tasks.keys():
            # get negotiation
            cohda_negotiation_model: MoCohdaNegotiationModel = self.context.get_or_create_model(MoCohdaNegotiationModel)
            if not cohda_negotiation_model.exists(content.negotiation_id):
                logger.warning(f'Received a stop message for a negotiation with id {content.negotiation_id} '
                               'but no such negotiation is running.')
                return
            cohda_negotiation = cohda_negotiation_model.by_id(content.negotiation_id)

            cohda_negotiation.stopped = True

            # wait until current iteration of negotiation is done (in case it is still running)
            self.context.schedule_conditional_task(self.stop_cohda_task(content.negotiation_id),
                                                   condition_func=lambda: not cohda_negotiation.active,
                                                   lookup_delay=0.05)

    async def stop_cohda_task(self, negotiation_id):
        """
        Will stop the process message task of a certain negotiation
        :param negotiation_id: ID of the negotiation
        """
        # cancel task
        task = self._cohda_tasks[negotiation_id]
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    def handle_solution_request(self, content: MoCohdaSolutionRequestMessage, meta):
        """
        Handles a solution request for a certain negotiation
        :param content: The CohdaSolutionRequestMessage
        :param meta: meta of the message
        """
        # get negotiation
        mocohda_negotiation_model: MoCohdaNegotiationModel = self.context.get_or_create_model(MoCohdaNegotiationModel)
        if not mocohda_negotiation_model.exists(content.negotiation_id):
            logger.warning(f'Received a solution request message for a negotiation with id {content.negotiation_id} '
                           'but no such negotiation exists.')
            return
        mocohda_negotiation = mocohda_negotiation_model.by_id(content.negotiation_id)

        # get current solution candidate
        final_solution = mocohda_negotiation._memory.solution_candidate
        # send CohdaProposedSolutionMessage
        self.context.schedule_instant_task(
            self.context.send_acl_message(content=MoCohdaProposedSolutionMessage(
                solution_candidate=final_solution, negotiation_id=content.negotiation_id
            ),
                receiver_addr=meta['sender_addr'], receiver_id=meta['sender_id'],
                acl_metadata={'sender_id': self.context.aid}
            ),
        )

    def handle_cohda_solution_msg(self, content: MoCohdaFinalSolutionMessage, meta):
        """
        Is called once a CohdaFinalSolutionMessage arrives
        :param content: The CohdaFinalSolutionMessage
        :param meta: Meta dict
        :return:
        """
        final_candidate: SolutionPoint = content.solution_point
        neg_id = content.negotiation_id
        # get part id from negotiation
        part_id = self.context.get_or_create_model(MoCohdaNegotiationModel).by_id(neg_id)._part_id
        # get individual schedule from final candidate
        final_schedule = final_candidate.cluster_schedule[final_candidate.idx[part_id]]

        # add final schedule to CohdaSolutionModel
        model = self.context.get_or_create_model(MoCohdaSolutionModel)
        model.add(neg_id, final_schedule)
        self.context.update(model)
        # reply with a confirmation
        self.context.schedule_instant_task(
            self.context.send_acl_message(
                content=ConfirmMoCohdaSolutionMessage(negotiation_id=neg_id, solution_point=final_candidate),
                receiver_addr=meta['sender_addr'], receiver_id=meta['sender_id'],
                acl_metadata={'sender_id': self.context.aid})
        )


class MoCohdaNegotiationModel:
    """Model for storing all metadata regarding negotiations
    """

    def __init__(self) -> None:
        self._negotiations: Dict[UUID, MoCohdaNegotiation] = {}

    def by_id(self, negotiation_id: UUID) -> MoCohdaNegotiation:
        """Get a negotiation by id
        :param negotiation_id: id of the negotiation
        :return: the negotiation
        """
        return self._negotiations[negotiation_id]

    def exists(self, negotiation_id: UUID) -> bool:
        """Checks whether a negotiation exists
        :param negotiation_id: id of the negotiation
        :return: True if it exists, False otherwise
        """
        return negotiation_id in self._negotiations.keys()

    def add(self, negotiation_id: UUID, cohda_negotiation: MoCohdaNegotiation):
        """Add a concrete negotiation
        :param negotiation_id: the UUID of the negotiation
        :param cohda_negotiation: the cohda negotiation object
        """
        self._negotiations[negotiation_id] = cohda_negotiation


class MoCohdaSolutionModel:
    def __init__(self) -> None:
        self._final_schedules: Dict[UUID, np.array] = {}

    def by_id(self, negotiation_id: UUID) -> np.array:
        """Get a solution of a negotiation by id
        :param negotiation_id: id of the negotiation
        :return: the negotiation
        """
        return self._final_schedules[negotiation_id]

    def exists(self, negotiation_id: UUID) -> bool:
        """Checks whether a final schedule exists
        :param negotiation_id: id of the negotiation
        :return: True if it exists, False otherwise
        """
        return negotiation_id in self._final_schedules.keys()

    def add(self, negotiation_id: UUID, final_schedule: np.array):
        """Add a concrete negotiation solution
        :param negotiation_id: the UUID of the negotiation
        :param final_schedule: the final schedule for the agent
        """
        self._final_schedules[negotiation_id] = final_schedule
