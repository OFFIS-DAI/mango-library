"""Module for distributed real power planning with COHDA. Contains roles, which
integrate COHDA in the negotiation system and the core COHDA-decider together with its model.
"""
from typing import List, Dict, Any, Tuple, Optional, Callable
import asyncio
import numpy as np
import logging

from mango.messages.codecs import json_serializable
from mango_library.negotiation.core import NegotiationParticipantRole, NegotiationStarterRole, Negotiation
from mango_library.coalition.core import CoalitionAssignment
from mango_library.negotiation.cohda.data_classes import \
    SolutionCandidate, SystemConfig, WorkingMemory, ScheduleSelection

logger = logging.getLogger(__name__)


@json_serializable
class CohdaMessage:
    """
    Message for a COHDA negotiation.
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


@json_serializable
class CohdaSolution:
    """
    Message for a COHDA solution.
    Contains the candidate of an agent.
    """

    def __init__(self, solution_candidate: SolutionCandidate):
        self._solution_candidate = solution_candidate

    @property
    def solution_candidate(self) -> SolutionCandidate:
        """Return the solution candidate of the sender agent

        :return: the solution_candidate of the sender
        """
        return self._solution_candidate


class CohdaNegotiationStarterRole(NegotiationStarterRole):
    """Convenience role for starting a COHDA negotiation with simply providing a target schedule
    """

    # create an empyt Working memory and send it together with the target params
    def __init__(self, target_params, coalition_model_matcher=None, coalition_uuid=None) -> None:
        """

        :param target_params: Parameter that are necessary for the agents to calculate the performance.
        Could be e.g. the target schedule.
        :param coalition_model_matcher:
        :param coalition_uuid:
        """
        super().__init__(
            lambda assignment:
            CohdaMessage(WorkingMemory(target_params=target_params, system_config=SystemConfig({}),
                                       solution_candidate=SolutionCandidate(
                                           agent_id=assignment.part_id, schedules={}, perf=float('-inf')
                                       ))),
            coalition_model_matcher=coalition_model_matcher, coalition_uuid=coalition_uuid
        )


class COHDA:
    """COHDA-decider
    """

    def __init__(self, schedule_provider, is_local_acceptable, part_id: str, perf_func=None):
        self._schedule_provider = schedule_provider
        self._is_local_acceptable = is_local_acceptable
        self._memory = WorkingMemory(None, SystemConfig({}), SolutionCandidate(part_id, {}, float('-inf')))
        self._counter = 0
        self._part_id = part_id
        if perf_func is None:
            self._perf_func = self.deviation_to_target_schedule
        else:
            self._perf_func = perf_func

    @staticmethod
    def deviation_to_target_schedule(cluster_schedule: np.array, target_parameters):
        if cluster_schedule.size == 0:
            return float('-inf')
        target_schedule, weights = target_parameters
        sum_cs = cluster_schedule.sum(axis=0)  # sum for each interval
        diff = np.abs(np.array(target_schedule) - sum_cs)  # deviation to the target schedule
        w_diff = diff * np.array(weights)  # multiply with weight vector
        result = -np.sum(w_diff)
        return float(result)

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
                    schedule_choices[self._part_id] = ScheduleSelection(
                        np.array(self._schedule_provider()[0]), self._counter + 1)
                    self._counter += 1
                    # we need to create a new class of Systemconfig so the updates are
                    # recognized in handle_cohda_msgs()
                    current_sysconfig = SystemConfig(schedule_choices=schedule_choices)
                else:
                    current_sysconfig = self._memory.system_config

            if current_candidate is None:
                if self._part_id not in self._memory.solution_candidate.schedules:
                    # if you have not yet selected any schedule in the sysconfig, choose any to start with
                    schedules = self._memory.solution_candidate.schedules
                    schedules[self._part_id] = self._schedule_provider()[0]
                    # we need to create a new class of SolutionCandidate so the updates are
                    # recognized in handle_cohda_msgs()
                    current_candidate = SolutionCandidate(agent_id=self._part_id, schedules=schedules, perf=None)
                    current_candidate.perf = self._perf_func(current_candidate.cluster_schedule,
                                                             self._memory.target_params)
                else:
                    current_candidate = self._memory.solution_candidate

            new_sysconf = message.working_memory.system_config
            new_candidate = message.working_memory.solution_candidate

            # Merge new information into current_sysconfig and current_candidate
            current_sysconfig = self._merge_sysconfigs(sysconfig_i=current_sysconfig, sysconfig_j=new_sysconf)
            current_candidate = self._merge_candidates(candidate_i=current_candidate,
                                                        candidate_j=new_candidate,
                                                        agent_id=self._part_id,
                                                        perf_func=self._perf_func,
                                                        target_params=self._memory.target_params)

        return current_sysconfig, current_candidate

    def _decide(self, sysconfig: SystemConfig, candidate: SolutionCandidate) -> Tuple[SystemConfig, SolutionCandidate]:
        """
        Check whether a better SolutionCandidate can be created based on the current state of the negotiation
        :param sysconfig: Current SystemConfig
        :param candidate: Current SolutionCandidate
        :return: Tuple of SystemConfig, SolutionCandidate. Unchanged to parameters if no new SolutionCandidate was
        found. Else it consists of the new SolutionCandidate and an updated SystemConfig
        """
        possible_schedules = self._schedule_provider()
        current_best_candidate = candidate
        for schedule in possible_schedules:
            if self._is_local_acceptable(schedule):
                # create new candidate from sysconfig
                new_candidate = SolutionCandidate.create_from_updated_sysconf(
                    agent_id=self._part_id, sysconfig=sysconfig, new_schedule=np.array(schedule)
                )
                new_performance = self._perf_func(new_candidate.cluster_schedule, self._memory.target_params)
                # only keep new candidates that perform better than the current one
                if new_performance > current_best_candidate.perf:
                    new_candidate.perf = new_performance
                    current_best_candidate = new_candidate

        schedule_in_candidate = current_best_candidate.schedules.get(self._part_id, None)
        schedule_choice_in_sysconfig = sysconfig.schedule_choices.get(self._part_id, None)

        if schedule_choice_in_sysconfig is None or \
                not np.array_equal(schedule_in_candidate, schedule_choice_in_sysconfig.schedule):
            # update Sysconfig if your schedule in the current sysconf is different to the one in the candidate
            sysconfig.schedule_choices[self._part_id] = ScheduleSelection(
                schedule=schedule_in_candidate, counter=self._counter + 1)
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

        sysconfig_i_schedules: Dict[str, ScheduleSelection] = sysconfig_i.schedule_choices
        sysconfig_j_schedules: Dict[str, ScheduleSelection] = sysconfig_j.schedule_choices
        key_set_i = set(sysconfig_i_schedules.keys())
        key_set_j = set(sysconfig_j_schedules.keys())

        new_sysconfig: Dict[str, ScheduleSelection] = {}
        modified = False

        for i, a in enumerate(sorted(key_set_i | key_set_j)):
            # An a might be in key_set_i, key_set_j or in both!
            if a in key_set_i and \
                    (a not in key_set_j or sysconfig_i_schedules[a].counter >= sysconfig_j_schedules[a].counter):
                # Use data of sysconfig_i
                schedule_selection = sysconfig_i_schedules[a]
            else:
                # Use data of sysconfig_j
                schedule_selection = sysconfig_j_schedules[a]
                modified = True

            new_sysconfig[a] = schedule_selection

        if modified:
            sysconf = SystemConfig(new_sysconfig)
        else:
            sysconf = sysconfig_i

        return sysconf

    @staticmethod
    def _merge_candidates(candidate_i: SolutionCandidate, candidate_j: SolutionCandidate,
                          agent_id: str, perf_func: Callable, target_params):
        """
        Returns a merged Candidate. If the candidate_i remains unchanged, the same instance of candidate_i is
        returned, otherwise a new object is created with agent_id as candidate.agent_id
        :param candidate_i: The first candidate
        :param candidate_j: The second candidate
        :param agent_id: The agent_id that defines who is the creator of a new candidate
        :param perf_func: The performance function
        :param target_params: The current target parameters (e. g. a target schedule)
        :return:  A merged SolutionCandidate. If the candidate_i remains unchanged, the same instance of candidate_i is
        returned, otherwise a new object is created.
        """
        keyset_i = set(candidate_i.schedules.keys())
        keyset_j = set(candidate_j.schedules.keys())
        candidate = candidate_i  # Default candidate is *i*

        if keyset_i < keyset_j:
            # Use *j* if *K_i* is a true subset of *K_j*
            candidate = candidate_j
        elif keyset_i == keyset_j:
            # Compare the performance if the keyset is equal
            if candidate_j.perf > candidate_i.perf:
                # Choose *j* if it performs better
                candidate = candidate_j
            elif candidate_j.perf == candidate_i.perf:
                # If both perform equally well, order them by name
                if candidate_j.agent_id < candidate_i.agent_id:
                    candidate = candidate_j
        elif keyset_j - keyset_i:
            # If *candidate_j* shares some entries with *candidate_i*, update *candidate_i*
            new_schedules: Dict[str, np.array] = {}
            for a in sorted(keyset_i | keyset_j):
                if a in keyset_i:
                    schedule = candidate_i.schedules[a]
                else:
                    schedule = candidate_j.schedules[a]
                new_schedules[a] = schedule

            # create new SolutionCandidate
            candidate = SolutionCandidate(agent_id=agent_id, schedules=new_schedules, perf=None)
            candidate.perf = perf_func(candidate.cluster_schedule, target_params)

        return candidate


class COHDARole(NegotiationParticipantRole):
    """Negotiation role for COHDA.
    """

    def __init__(self, schedules_provider, local_acceptable_func=None, check_inbox_interval: float = 0.1):
        """
        Init of COHDARole
        :param schedules_provider: Function that takes not arguments and returns a list of schedules
        :param local_acceptable_func: Function that takes a schedule as input and returns a boolean indicating,
        if the schedule is locally acceptable or not. Defaults to lambda x: True
        :param check_inbox_interval: Duration of buffering the cohda messages [s]
        """
        super().__init__()

        self._schedules_provider = schedules_provider
        if local_acceptable_func is None:
            self._is_local_acceptable = lambda x: True
        else:
            self._is_local_acceptable = local_acceptable_func
        self._cohda = {}
        self._cohda_msg_queues = {}
        self._cohda_tasks = {}
        self.check_inbox_interval = check_inbox_interval

        self.inactive_counter = 0

    def create_cohda(self, part_id: str):
        """
        Create an instance of COHDA.
        :param part_id: participant id
        :return: COHDA object
        """
        return COHDA(schedule_provider=self._schedules_provider,
                     is_local_acceptable=self._is_local_acceptable,
                     part_id=part_id)


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

    def handle(self,
               message,
               assignment: CoalitionAssignment,
               negotiation: Negotiation,
               meta: Dict[str, Any]):

        if not negotiation.stopped:
            if negotiation.negotiation_id in self._cohda:
                if not negotiation.active:
                    print(f'[{self.context.addr, self.context.aid}] ATTENTION, negotiation was not active and i received a mssage')
                negotiation.active = True
                self._cohda_msg_queues[negotiation.negotiation_id].append(message)
            else:
                self._cohda[negotiation.negotiation_id] = self.create_cohda(assignment.part_id)
                self._cohda_msg_queues[negotiation.negotiation_id] = [message]

                async def process_msg_queue():
                    """
                    Method to evaluate all incoming message of a cohda_message_queue for a certain negotiation
                    """

                    if len(self._cohda_msg_queues[negotiation.negotiation_id]) > 0 and not negotiation.stopped:
                        # get queue
                        cohda_message_queue, self._cohda_msg_queues[negotiation.negotiation_id] = \
                            self._cohda_msg_queues[negotiation.negotiation_id], []

                        message_to_send = self._cohda[negotiation.negotiation_id].handle_cohda_msgs(cohda_message_queue)

                        if message_to_send is not None:
                            await self.send_to_neighbors(assignment, negotiation, message_to_send)

                        # else:
                        #     # set the negotiation as inactive as the incoming information was known already
                        #     print(f'{self.context.addr, self.context.aid} Setting neg active = False -> nothing new in decide')
                        #     negotiation.active = False
                    else:
                        # set the negotiation as inactive as no message has arrived
                        print(f'{self.context.addr, self.context.aid} Setting neg active = False -> no new message')
                        self.inactive_counter += 1
                        if self.inactive_counter > 4:
                            negotiation.active = False

                self._cohda_tasks[negotiation.negotiation_id] = \
                    self.context.schedule_periodic_task(process_msg_queue, delay=self.check_inbox_interval)

    async def handle_neg_stop(self, negotiation: Negotiation, meta: Dict[str, Any]):
        """
        """
        print(f'[{self.context.addr}] handle neg stop')
        if negotiation.negotiation_id in self._cohda_tasks.keys():
            # wait until current iteration is done
            while self._cohda[negotiation.negotiation_id].active:
                await asyncio.sleep(0.05)
            # cancel task
            task = self._cohda_tasks[negotiation.negotiation_id]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        # get current solution
        final_solution = self._cohda[negotiation.negotiation_id]._memory.solution_candidate
        await self.context.send_message(content=CohdaSolution(final_solution), receiver_addr=meta['sender_id'],
                                        receiver_id=meta['receiver_id'], create_acl=True)
