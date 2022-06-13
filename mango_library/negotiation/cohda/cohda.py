"""Module for distributed real power planning with COHDA. Contains roles, which
integrate COHDA in the negotiation system and the core COHDA-decider together with its model.
"""
from typing import List, Dict, Any, Tuple
import asyncio
import numpy as np

from mango.messages.codecs import json_serializable
from mango_library.negotiation.core import NegotiationParticipant, NegotiationStarterRole, Negotiation
from mango_library.coalition.core import CoalitionAssignment
from mango_library.negotiation.cohda.data_classes import \
    SolutionCandidate, SystemConfig, WorkingMemory, ScheduleSelection


@json_serializable
class CohdaMessage:
    """
    Message for a COHDa negotiation.
    Contains the candidate and the working memory of an agent.
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
            def deviation_to_target_schedule(cluster_schedule: np.array, target_parameters):
                if cluster_schedule.size == 0:
                    return float('-inf')
                target_schedule, weights = target_parameters
                sum_cs = cluster_schedule.sum(axis=0)  # sum for each interval
                diff = np.abs(np.array(target_schedule) - sum_cs)  # deviation to the target schedule
                w_diff = diff * np.array(weights)  # multiply with weight vector
                result = -np.sum(w_diff)
                return float(result)
            self._perf_func = deviation_to_target_schedule
        else:
            self._perf_func = perf_func

    def perceive(self, messages: List[CohdaMessage]) -> Tuple[SystemConfig, SolutionCandidate]:
        """

        :param messages: The List of received CohdaMessages
        :return: a tuple of SystemConfig, Candidate as a result of perceive
        """
        current_sysconfig = None
        current_candidate = None
        for message in messages:
            if self._memory.target_params is None:
                self._memory.target_params = message.working_memory.target_params

            if current_sysconfig is None:
                if self._part_id not in self._memory.system_config.schedule_choices:
                    # if you have not yet selected any schedule in the sysconfig, choose any to start with

                    schedule_choices = self._memory.system_config.schedule_choices
                    schedule_choices[self._part_id] = ScheduleSelection(
                        np.array(self._schedule_provider()[0]), self._counter + 1)
                    self._counter += 1
                    # we need to create a new class of Systemconfig so the COHDARole recognizes the update
                    current_sysconfig = SystemConfig(schedule_choices=schedule_choices)
                else:
                    current_sysconfig = self._memory.system_config

            if current_candidate is None:
                if self._part_id not in self._memory.solution_candidate.schedules:
                    # if you have not yet selected any schedule in the sysconfig, choose any to start with
                    schedules = self._memory.solution_candidate.schedules
                    schedules[self._part_id] = self._schedule_provider()[0]
                    # we need to create a new class of SolutionCandidate so the COHDARole recognizes the update
                    current_candidate = SolutionCandidate(agent_id=self._part_id, schedules=schedules, perf=None)
                    current_candidate.perf = self._perf_func(current_candidate.cluster_schedule,
                                                             self._memory.target_params)
                else:
                    current_candidate = self._memory.solution_candidate

            new_sysconf = message.working_memory.system_config
            new_candidate = message.working_memory.solution_candidate

            current_sysconfig = SystemConfig.merge(sysconfig_i=current_sysconfig, sysconfig_j=new_sysconf)
            current_candidate = SolutionCandidate.merge(candidate_i=current_candidate,
                                                        candidate_j=new_candidate,
                                                        agent_id=self._part_id,
                                                        perf_func=self._perf_func,
                                                        target_params=self._memory.target_params)

        return current_sysconfig, current_candidate

    def decide(self, sysconfig: SystemConfig, candidate: SolutionCandidate) -> Tuple[SystemConfig, SolutionCandidate]:
        """

        :param sysconfig:
        :param candidate:
        :return:
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

    def act(self, new_sysconfig: SystemConfig, new_candidate: SolutionCandidate) -> CohdaMessage:
        """
        Stores the new SystemCondig and SolutionCandidate in Memory and returns the COHDA message that should be sent
        :param new_sysconfig: The SystemConfig as a result from perceive and decide
        :param new_candidate: The SolutionCandidate as a result from perceive and decide
        :return: The COHDA message that should be sent
        """
        # update memory
        self._memory.system_config = new_sysconfig
        self._memory.solution_candidate = new_candidate
        # return COHDA message
        return CohdaMessage(working_memory=self._memory)


class COHDARole(NegotiationParticipant):
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
        self._cohda_tasks = []
        self.check_inbox_interval = check_inbox_interval

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
                    # copy queue
                    cohda_message_queue, self._cohda_msg_queues[negotiation.coalition_id] = \
                        self._cohda_msg_queues[negotiation.coalition_id], []
                    # get cohda object
                    current_cohda = self._cohda[negotiation.coalition_id]
                    # copy old memory
                    old_sysconf = current_cohda._memory.system_config
                    old_candidate = current_cohda._memory.solution_candidate

                    # perceive
                    sysconf, candidate = current_cohda.perceive(cohda_message_queue)

                    # decide
                    if sysconf is not old_sysconf or candidate is not old_candidate:
                        sysconf, candidate = current_cohda.decide(sysconfig=sysconf, candidate=candidate)
                        # act
                        message_to_send = current_cohda.act(new_sysconfig=sysconf, new_candidate=candidate)
                        if message_to_send is not None:
                            await self.send_to_neighbors(assignment, negotiation,
                                                         message_to_send)

                    else:
                        # set the negotiation as inactive as the incoming information was known already
                        negotiation.active = False
                else:
                    negotiation.active = False

            self._cohda_tasks.append(self.context.schedule_periodic_task(process_msg_queue,
                                                                         delay=self.check_inbox_interval))
