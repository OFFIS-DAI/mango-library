"""Module for distributed real power planning with COHDA. Contains roles, which
integrate COHDA in the negotiation system and the core COHDA-decider together with its model.
"""
from typing import Dict, Any, Tuple
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
                                       solution_candidate=SolutionCandidate(assignment.part_id, {}))),
            coalition_model_matcher=coalition_model_matcher, coalition_uuid=coalition_uuid
        )


class COHDA:
    """COHDA-decider
    """

    def __init__(self, schedule_provider, is_local_acceptable, part_id, perf_func=None):
        self._schedule_provider = schedule_provider
        self._is_local_acceptable = is_local_acceptable
        self._memory = WorkingMemory(None, SystemConfig({}), SolutionCandidate(part_id, {}))
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

    def decide(self, content: CohdaMessage) -> Tuple[WorkingMemory, WorkingMemory]:
        """Execute the COHDA decision process.

        :param content: the incoming COHDA message

        :return: old and new working memory
        """
        memory = self._memory
        selection_counter = self._counter

        if memory.target_params is None:
            memory.target_params = content.working_memory.target_params

        old_working_memory = copy.deepcopy(memory)

        if self._part_id not in memory.system_config.schedule_choices:
            memory.system_config.schedule_choices[self._part_id] = ScheduleSelection(None, self._counter)

        own_schedule_selection_wm = memory.system_config.schedule_choices[self._part_id]
        our_solution_cand, objective_our_candidate = self._evaluate_message(content, memory)
        possible_schedules = self._schedule_provider()
        our_selected_schedule = our_solution_cand.schedules[self._part_id] \
            if self._part_id in our_solution_cand.schedules else None
        found_new = False
        for schedule in possible_schedules:
            our_solution_cand.schedules[self._part_id] = schedule
            objective_tryout_candidate = self._perf_func(our_solution_cand.cluster_schedule,
                                                         memory.target_params)
            if objective_tryout_candidate > objective_our_candidate \
               and self._is_local_acceptable(schedule):
                our_selected_schedule = schedule
                objective_our_candidate = objective_tryout_candidate
                found_new = True

        if not found_new:
            our_solution_cand.schedules[self._part_id] = our_selected_schedule

        if not found_new and our_selected_schedule != own_schedule_selection_wm.schedule:
            our_selected_schedule = own_schedule_selection_wm.schedule
            found_new = True

        if found_new:
            memory.system_config.schedule_choices[self._part_id] = \
                ScheduleSelection(our_selected_schedule, selection_counter + 1)
            memory.solution_candidate = our_solution_cand
            our_solution_cand.agent_id = self._part_id
            our_solution_cand.schedules[self._part_id] = our_selected_schedule
            self._counter += 1
        return old_working_memory, memory

    def _evaluate_message(self, content: CohdaMessage, memory: WorkingMemory):
        """Evaluate the incoming message and update our candidate accordingly.

        :param content: the incoming message
        :param memory: our memory

        :return: our new solution candidate and its objective
        """

        msg_solution_cand = content.working_memory.solution_candidate
        our_solution_cand = memory.solution_candidate
        known_part_ids = set(memory.system_config.schedule_choices.keys())
        given_part_ids = set(content.working_memory.system_config.schedule_choices.keys())

        for agent_id, their_selection in content.working_memory.system_config.schedule_choices.items():
            if agent_id in memory.system_config.schedule_choices.keys():
                our_selection = memory.system_config.schedule_choices[agent_id]
                if their_selection.counter > our_selection.counter:
                    memory.system_config.schedule_choices[agent_id] = their_selection
            else:
                memory.system_config.schedule_choices[agent_id] = their_selection

        objective_our_candidate = self._perf_func(our_solution_cand.cluster_schedule, memory.target_params)

        if known_part_ids.issubset(given_part_ids):
            our_solution_cand = msg_solution_cand
        elif len(given_part_ids.union(known_part_ids)) > len(known_part_ids):
            missing_ids = given_part_ids.difference(known_part_ids)
            for missing_id in missing_ids:
                our_solution_cand.schedules[missing_id] = msg_solution_cand.schedules[missing_id]
        else:
            objective_message_candidate = self._perf_func(msg_solution_cand.cluster_schedule, memory.target_params)
            if objective_message_candidate > objective_our_candidate:
                our_solution_cand = msg_solution_cand
            elif objective_message_candidate == objective_our_candidate and \
                    msg_solution_cand.agent_id > our_solution_cand.agent_id:
                our_solution_cand = msg_solution_cand

        return our_solution_cand, objective_our_candidate


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

        (old, new) = self._cohda[negotiation.coalition_id].decide(message)

        if old != new:
            self.send_to_neighbors(assignment, negotiation, CohdaMessage(new))

            # set agent as idle
            if self.context.inbox_length() == 0:
                negotiation.active = False
