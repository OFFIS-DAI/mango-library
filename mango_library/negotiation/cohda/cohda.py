"""Module for distributed real power planning with COHDA. Contains roles, which
integrate COHDA in the negotiation system and the core COHDA-decider together with its model.
"""
from typing import List, Dict, Any, Tuple, Optional
import copy
import numpy as np

from mango_library.negotiation.core import NegotiationParticipant, NegotiationStarterRole, Negotiation
from mango_library.coalition.core import CoalitionAssignment
from mango_library.negotiation.cohda.data_classes import \
    SolutionCandidate, SystemConfig, WorkingMemory, ScheduleSelection


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

    # create an empyt Working memory and send
    def __init__(self, target_params, coalition_model_matcher=None, coalition_uuid=None) -> None:
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

    def __init__(self, schedule_provider, is_local_acceptable, part_id, perf_func=None):
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
                return result
            self._perf_func = deviation_to_target_schedule
        else:
            self._perf_func = perf_func

    def perceive(self, messages: List[CohdaMessage]) -> Tuple[SystemConfig, SolutionCandidate]:
        """

        :param content:
        :return: a tuple of
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
                    current_candidate.perf = self._perf_func(current_candidate.cluster_schedule, self._memory.target_params)
                else:
                    current_candidate = self._memory.solution_candidate

            new_sysconf = message.working_memory.system_config
            new_candidate = message.working_memory.solution_candidate

            current_sysconfig = SystemConfig.merge(sysconfig_i=current_sysconfig, sysconfig_j=new_sysconf)
            current_candidate = SolutionCandidate.merge(candidate_i=current_candidate, candidate_j=new_candidate, agent_id=self._part_id,
                                                perf_func=self._perf_func, target_params=self._memory.target_params)

        return current_sysconfig, current_candidate

    def _decide(self, sysconfig: SystemConfig, candidate: SolutionCandidate) -> Tuple[SystemConfig, SolutionCandidate]:
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

    def _act(self, new_sysconfig, new_candidate) -> Optional[CohdaMessage]:
        if new_sysconfig != self._memory.system_config or new_candidate != self._memory.solution_candidate:
            # update memory
            self._memory.system_config = new_sysconfig
            self._memory.solution_candidate = new_candidate

            # send message
            return CohdaMessage(working_memory=self._memory)

        # don't send message
        return None


class COHDARole(NegotiationParticipant):
    """Negotiation role for COHDA.
    """

    def __init__(self, schedules_provider, local_acceptable_func):
        super().__init__()

        self._schedules_provider = schedules_provider
        self._is_local_acceptable = local_acceptable_func
        self._cohda = {}

    def create_cohda(self, part_id: int):
        """Create an instance of the COHDA-decider.

        :param part_id: participant id

        :return: COHDA
        """
        return COHDA(schedule_provider=self._schedules_provider,
                     is_local_acceptable=self._is_local_acceptable,
                     part_id=part_id)

    def handle(self,
               message,
               assignment: CoalitionAssignment,
               negotiation: Negotiation,
               meta: Dict[str, Any]):

        if negotiation.coalition_id not in self._cohda:
            self._cohda[negotiation.coalition_id] = self.create_cohda(assignment.part_id)

        # (old, new) = self._cohda[negotiation.coalition_id].decide(message)

        this_cohda: COHDA = self._cohda[negotiation.coalition_id]
        old_sysconf = this_cohda._memory.system_config
        old_candidate = this_cohda._memory.solution_candidate


        sysconf, candidate = this_cohda.perceive(messages=[message])

        if sysconf is not old_sysconf or candidate is not old_candidate:
            sysconf, candidate = this_cohda._decide(sysconfig=sysconf, candidate=candidate)
            message_to_send = this_cohda._act(new_sysconfig=sysconf, new_candidate=candidate)

            if message_to_send is not None:
                self.send_to_neighbors(assignment, negotiation, message_to_send)

                # set agent as idle
                if self.context.inbox_length() == 0:
                    negotiation.active = False
