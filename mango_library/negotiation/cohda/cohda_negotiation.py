import asyncio
import inspect
import logging
from typing import List, Dict, Optional, Tuple, Callable
from uuid import UUID

import numpy as np

from mango import Role
from mango_library.coalition.core import CoalitionAssignment, CoalitionModel
from mango_library.negotiation.cohda.cohda_messages import (
    CohdaNegotiationMessage,
    CohdaSolutionRequestMessage,
    CohdaProposedSolutionMessage,
    StopNegotiationMessage,
    CohdaFinalSolutionMessage,
    ConfirmCohdaSolutionMessage,
)
from mango_library.negotiation.cohda.data_classes import (
    WorkingMemory,
    SolutionCandidate,
    SystemConfig,
    ScheduleSelection,
)

logger = logging.getLogger(__name__)


class COHDANegotiationRole(Role):
    """Negotiation role for COHDA."""

    def __init__(
        self,
        schedules_provider: Callable,
        local_acceptable_func: Callable = None,
        perf_func: Callable = None,
        check_inbox_interval: float = 0.1,
    ):
        """
        Init of COHDANegotiationRole
        :param schedules_provider: Function that takes either no arguments or "candidate" as an argument or
        "system_config" as an argument or "candidate" and "system_config" as arguments and
        provides a list of possible schedules
        :param local_acceptable_func: Function that takes a schedule as input and returns a boolean indicating,
        if the schedule is locally acceptable or not. Defaults to lambda x: True
        :param perf_func: performance function for the agent. Defaults to deviation_from_target_schedule
        :param check_inbox_interval: Duration of buffering the CohdaNegotiationMessages [s]
        """
        super().__init__()

        self._schedules_provider = schedules_provider
        self._perf_func = (
            perf_func
            if perf_func is not None
            else COHDANegotiation.deviation_to_target_schedule
        )

        if local_acceptable_func is None:
            # accept all schedules if not function is provided
            self._is_local_acceptable = lambda x: True
        else:
            self._is_local_acceptable = local_acceptable_func

        self._cohda_msg_queues: Dict[
            UUID, List[WorkingMemory]
        ] = {}  # stores the message queues
        self._cohda_tasks: Dict[
            UUID, asyncio.Task
        ] = {}  # stores the tasks that process the inbox
        self.check_inbox_interval = check_inbox_interval

    def setup(self) -> None:
        super().setup()

        # negotiation message
        self.context.subscribe_message(
            self,
            self.handle_neg_msg,
            lambda c, _: isinstance(c, CohdaNegotiationMessage),
        )
        # stop negotiation message
        self.context.subscribe_message(
            self,
            self.handle_neg_stop,
            lambda c, _: isinstance(c, StopNegotiationMessage),
        )
        # solution request message
        self.context.subscribe_message(
            self,
            self.handle_solution_request,
            lambda c, _: isinstance(c, CohdaSolutionRequestMessage),
        )
        # final solution message
        self.context.subscribe_message(
            self,
            self.handle_cohda_solution_msg,
            lambda c, _: isinstance(c, CohdaFinalSolutionMessage),
        )

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

    def handle_neg_msg(self, content: CohdaNegotiationMessage, _):
        """
        Will be called when a CohdaNegotiationMessage is received
        :param content: CohdaNegotiationMessage
        :param _: meta dict
        """
        # check if there is a Coalition with the coalition_ID
        if not self.context.get_or_create_model(CoalitionModel).exists(
            content.coalition_id
        ):
            logger.warning(
                f"Received a CohdaNegotiationMessage with the coalition_id {content.coalition_id}"
                f"but there is no such Coalition known."
            )
            return

        # get coalition_assignment
        coalition_assignment: CoalitionAssignment = self.context.get_or_create_model(
            CoalitionModel
        ).by_id(content.coalition_id)

        # get negotiation model
        cohda_negotiation_model: CohdaNegotiationModel = (
            self.context.get_or_create_model(CohdaNegotiationModel)
        )
        # add negotiation if it doesn't exist
        if not cohda_negotiation_model.exists(content.negotiation_id):
            cohda_negotiation_model.add(
                negotiation_id=content.negotiation_id,
                cohda_negotiation=COHDANegotiation(
                    part_id=coalition_assignment.part_id,
                    schedule_provider=self._schedules_provider,
                    is_local_acceptable=self._is_local_acceptable,
                    perf_func=self._perf_func,
                ),
            )
        cohda_negotiation = cohda_negotiation_model.by_id(
            negotiation_id=content.negotiation_id
        )

        # add message to the queue if negotiation is not stopped
        if not cohda_negotiation.stopped:
            if content.negotiation_id in self._cohda_msg_queues.keys():
                cohda_negotiation.active = True
                self._cohda_msg_queues[content.negotiation_id].append(
                    content.working_memory
                )
            else:
                self._cohda_msg_queues[content.negotiation_id] = [
                    content.working_memory
                ]
                self._cohda_tasks[
                    content.negotiation_id
                ] = self.context.schedule_periodic_task(
                    self.get_process_msg_queue_coro(
                        cohda_negotiation=cohda_negotiation,
                        negotiation_id=content.negotiation_id,
                        coalition_assignment=coalition_assignment,
                    ),
                    delay=self.check_inbox_interval,
                )

    def get_process_msg_queue_coro(
        self,
        cohda_negotiation,
        coalition_assignment: CoalitionAssignment,
        negotiation_id: UUID,
    ):
        """
        Method that returns a coroutine that process a message queue of a specific COHDA negotiation.
        :param cohda_negotiation: the COHDANegotiation instance
        :param coalition_assignment: the corresponding coalition assignment
        :param negotiation_id: the corresponding negotiation ID
        :return: a coroutine without arguments that can be scheduled as periodic task
        """

        async def process_msg():
            if (
                len(self._cohda_msg_queues[negotiation_id]) > 0
                and not cohda_negotiation.stopped
            ):
                # get queue
                cohda_message_queue, self._cohda_msg_queues[negotiation_id] = (
                    self._cohda_msg_queues[negotiation_id],
                    [],
                )

                wm_to_send = cohda_negotiation.handle_cohda_msgs(cohda_message_queue)

                if wm_to_send is not None:
                    # send message to all neighbors
                    for neighbor in coalition_assignment.neighbors:
                        self.context.schedule_instant_acl_message(
                            content=CohdaNegotiationMessage(
                                negotiation_id=negotiation_id,
                                coalition_id=coalition_assignment.coalition_id,
                                working_memory=wm_to_send,
                            ),
                            receiver_addr=neighbor[1],
                            receiver_id=neighbor[2],
                            acl_metadata={
                                "sender_addr": self.context.addr,
                                "sender_id": self.context.aid,
                            },
                        )

            else:
                # set the negotiation as inactive as no message has arrived
                cohda_negotiation.active = False

        return process_msg

    def handle_neg_stop(self, content: StopNegotiationMessage, _):
        """Is called once a StopNegotiationMessage arrived"""
        if content.negotiation_id in self._cohda_tasks.keys():
            # get negotiation
            cohda_negotiation_model: CohdaNegotiationModel = (
                self.context.get_or_create_model(CohdaNegotiationModel)
            )
            if not cohda_negotiation_model.exists(content.negotiation_id):
                logger.warning(
                    f"Received a stop message for a negotiation with id {content.negotiation_id} "
                    "but no such negotiation is running."
                )
                return
            cohda_negotiation = cohda_negotiation_model.by_id(content.negotiation_id)

            cohda_negotiation.stopped = True

            # wait until current iteration of negotiation is done (in case it is still running)
            self.context.schedule_conditional_task(
                self.stop_cohda_task(content.negotiation_id),
                condition_func=lambda: not cohda_negotiation.active,
                lookup_delay=0.05,
            )

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

    def handle_solution_request(self, content: CohdaSolutionRequestMessage, meta):
        """
        Handles a solution request for a certain negotiation
        :param content: The CohdaSolutionRequestMessage
        :param meta: meta of the message
        """
        # get negotiation
        cohda_negotiation_model: CohdaNegotiationModel = (
            self.context.get_or_create_model(CohdaNegotiationModel)
        )
        if not cohda_negotiation_model.exists(content.negotiation_id):
            logger.warning(
                f"Received a solution request message for a negotiation with id {content.negotiation_id} "
                "but no such negotiation exists."
            )
            return
        cohda_negotiation = cohda_negotiation_model.by_id(content.negotiation_id)

        # get current solution candidate
        final_solution = cohda_negotiation._memory.solution_candidate
        # send CohdaProposedSolutionMessage
        self.context.schedule_instant_acl_message(
            content=CohdaProposedSolutionMessage(
                solution_candidate=final_solution,
                negotiation_id=content.negotiation_id,
            ),
            receiver_addr=meta["sender_addr"],
            receiver_id=meta["sender_id"],
            acl_metadata={"sender_id": self.context.aid},
        )

    def handle_cohda_solution_msg(self, content: CohdaFinalSolutionMessage, meta):
        """
        Is called once a CohdaFinalSolutionMessage arrives
        :param content: The CohdaFinalSolutionMessage
        :param meta: Meta dict
        :return:
        """
        final_candidate = content.solution_candidate
        neg_id = content.negotiation_id
        # get part id from negotiation
        part_id = (
            self.context.get_or_create_model(CohdaNegotiationModel)
            .by_id(neg_id)
            ._part_id
        )
        # get individual schedule from final candidate
        final_schedule = final_candidate.schedules[part_id]
        # add final schedule to CohdaSolutionModel
        self.context.get_or_create_model(CohdaSolutionModel).add(neg_id, final_schedule)
        # reply with a confirmation
        self.context.schedule_instant_acl_message(
            content=ConfirmCohdaSolutionMessage(neg_id),
            receiver_addr=meta["sender_addr"],
            receiver_id=meta["sender_id"],
            acl_metadata={"sender_id": self.context.aid},
        )


class COHDANegotiation:
    """COHDA-decider"""

    def __init__(
        self,
        schedule_provider: Callable,
        is_local_acceptable: Callable,
        part_id: str,
        perf_func=None,
    ):
        """
        Init of the CohdaNegotiation
        :param schedule_provider: Function that takes either no arguments or "candidate" as an argument or
        "system_config" as an argument or "candidate" and "system_config" as arguments and
        provides a list of possible schedules
        :param is_local_acceptable: Function that defines whether a schedule is locally acceptable or not
        :param part_id: the part_id of the agent
        :param perf_func: The performance function, that takes one numpy array and target_params
        as input and returns a float
        """
        self._part_id = part_id

        def complete_schedule_provider(
            system_config: SystemConfig, candidate: SolutionCandidate
        ):
            schedule_provider_args = inspect.signature(
                schedule_provider
            ).parameters.keys()
            args = {}
            if "candidate" in schedule_provider_args:
                args["candidate"] = candidate
            if "system_config" in schedule_provider_args:
                args["system_config"] = system_config
            return schedule_provider(**args)

        self._schedule_provider = complete_schedule_provider

        self._is_local_acceptable = is_local_acceptable
        # start with an empty WorkingMemory
        self._memory = WorkingMemory(
            None, SystemConfig({}), SolutionCandidate(self._part_id, {}, float("-inf"))
        )
        self._counter = 0  # counter for ScheduleSelections

        self._stopped = False  # is Ture once a StopNegotiationMessage is received for this negotiation
        # self._active is False once an iteration of the process_message_queue function has not received any message
        self._active = True

        if perf_func is None:
            # as a default performance function we take the deviation to a target schedule, which is defined in
            # target_params
            self._perf_func = self.deviation_to_target_schedule
        else:
            self._perf_func = perf_func

    @staticmethod
    def deviation_to_target_schedule(
        cluster_schedule: np.array, target_parameters
    ) -> float:
        """Default Performance function for a cohda negotiation
        :param cluster_schedule: The cluster schedule to be evaluated
        :param target_parameters: The target parameters. In this case we expect a Tuple of target_schedule, weights
        :return: The performance
        """
        if cluster_schedule.size == 0:
            # we have no schedules, so return worst possible performance
            return float("-inf")
        # we expect this input here
        target_schedule, weights = target_parameters
        sum_cs = cluster_schedule.sum(axis=0)  # sum for each interval
        diff = np.abs(
            np.array(target_schedule) - sum_cs
        )  # deviation to the target schedule
        w_diff = diff * np.array(weights)  # multiply with weight vector
        result = -np.sum(w_diff)
        return float(result)  # cast to float, otherwise a numpy object is returned

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

    def handle_cohda_msgs(
        self, messages: List[WorkingMemory]
    ) -> Optional[WorkingMemory]:
        """
        This is called by the COHDARole.
        It takes a List of WorkingMemories, executes perceive, decide, act and returns
        a new WorkingMemory in case the working memory has changed and None otherwise
        :param messages: The list of received WorkingMemories
        :return: The message to be sent to the neighbors, None if no message has to be sent
        """
        # store old wm
        old_sysconf = self._memory.system_config
        old_candidate = self._memory.solution_candidate

        # perceive
        sysconf, candidate = self._perceive(messages)

        # decide
        if sysconf is not old_sysconf or candidate is not old_candidate:
            # something changed in perceive so we do decide and act
            sysconf, candidate = self._decide(sysconfig=sysconf, candidate=candidate)
            # act
            return self._act(new_sysconfig=sysconf, new_candidate=candidate)
        else:
            # nothing has changed in perceive, so we return None
            return None

    def _perceive(
        self, working_memories: List[WorkingMemory]
    ) -> Tuple[SystemConfig, SolutionCandidate]:
        """
        Updates the current knowledge
        :param working_memories: The List of received WorkingMemories
        :return: a tuple of SystemConfig, Candidate as a result of perceive
        """
        current_sysconfig = None
        current_candidate = None
        for new_wm in working_memories:
            if self._memory.target_params is None:
                # get target parameters if not known
                self._memory.target_params = new_wm.target_params
            if current_sysconfig is None:
                if self._part_id not in self._memory.system_config.schedule_choices:
                    # if you have not yet selected any schedule in the sysconfig, choose any to start with
                    schedule_choices = self._memory.system_config.schedule_choices
                    schedule_choices[self._part_id] = ScheduleSelection(
                        np.array(
                            self._schedule_provider(
                                candidate=self._memory.solution_candidate,
                                system_config=self._memory.system_config,
                            )[0]
                        ),
                        self._counter + 1,
                    )
                    self._counter += 1
                    # we need to create a new instance of SystemConfig so the updates are
                    # recognized in handle_cohda_msgs()
                    current_sysconfig = SystemConfig(schedule_choices=schedule_choices)
                else:
                    current_sysconfig = self._memory.system_config

            if current_candidate is None:
                if self._part_id not in self._memory.solution_candidate.schedules:
                    # if you have not yet selected any schedule in the sysconfig, choose any to start with
                    schedules = self._memory.solution_candidate.schedules
                    schedules[self._part_id] = self._schedule_provider(
                        candidate=self._memory.solution_candidate,
                        system_config=self._memory.system_config,
                    )[0]
                    # we need to create a new class of SolutionCandidate so the updates are
                    # recognized in handle_cohda_msgs()
                    current_candidate = SolutionCandidate(
                        agent_id=self._part_id, schedules=schedules, perf=None
                    )
                    current_candidate.perf = self._perf_func(
                        current_candidate.cluster_schedule, self._memory.target_params
                    )
                else:
                    current_candidate = self._memory.solution_candidate

            new_sysconf = new_wm.system_config
            new_candidate = new_wm.solution_candidate

            # Merge new information into current_sysconfig and current_candidate
            current_sysconfig = self._merge_sysconfigs(
                sysconfig_i=current_sysconfig, sysconfig_j=new_sysconf
            )
            current_candidate = self._merge_candidates(
                candidate_i=current_candidate,
                candidate_j=new_candidate,
                agent_id=self._part_id,
                perf_func=self._perf_func,
                target_params=self._memory.target_params,
            )

        return current_sysconfig, current_candidate

    def _decide(
        self, sysconfig: SystemConfig, candidate: SolutionCandidate
    ) -> Tuple[SystemConfig, SolutionCandidate]:
        """
        Check whether a better SolutionCandidate can be created based on the current state of the negotiation
        :param sysconfig: Current SystemConfig
        :param candidate: Current SolutionCandidate
        :return: Tuple of SystemConfig, SolutionCandidate. Unchanged to parameters if no new SolutionCandidate was
        found. Else it consists of the new SolutionCandidate and an updated SystemConfig
        """
        possible_schedules = self._schedule_provider(
            candidate=candidate, system_config=sysconfig
        )
        current_best_candidate = candidate
        for schedule in possible_schedules:
            if self._is_local_acceptable(schedule):
                # create new candidate from sysconfig
                new_candidate = SolutionCandidate.create_from_updated_sysconf(
                    agent_id=self._part_id,
                    sysconfig=sysconfig,
                    new_schedule=np.array(schedule),
                )
                new_performance = self._perf_func(
                    new_candidate.cluster_schedule, self._memory.target_params
                )
                # only keep new candidates that perform better than the current one
                if new_performance > current_best_candidate.perf:
                    new_candidate.perf = new_performance
                    current_best_candidate = new_candidate

        schedule_in_candidate = current_best_candidate.schedules.get(
            self._part_id, None
        )
        schedule_choice_in_sysconfig = sysconfig.schedule_choices.get(
            self._part_id, None
        )

        if schedule_choice_in_sysconfig is None or not np.array_equal(
            schedule_in_candidate, schedule_choice_in_sysconfig.schedule
        ):
            # update Sysconfig if your schedule in the current sysconf is different to the one in the candidate
            sysconfig.schedule_choices[self._part_id] = ScheduleSelection(
                schedule=schedule_in_candidate, counter=self._counter + 1
            )
            # update counter
            self._counter += 1

        return sysconfig, current_best_candidate

    def _act(
        self, new_sysconfig: SystemConfig, new_candidate: SolutionCandidate
    ) -> WorkingMemory:
        """
        Stores the new SystemConfig and SolutionCandidate in Memory and returns the new working Memory
        :param new_sysconfig: The SystemConfig as a result from perceive and decide
        :param new_candidate: The SolutionCandidate as a result from perceive and decide
        :return: The COHDA message that should be sent
        """
        # update memory
        self._memory.system_config = new_sysconfig
        self._memory.solution_candidate = new_candidate
        # return new Working Memory
        return self._memory

    @staticmethod
    def _merge_sysconfigs(sysconfig_i: SystemConfig, sysconfig_j: SystemConfig):
        """
        Merge *sysconf_i* and *sysconf_j* and return the result.

        Returns a merged SystemConfig. If the sysconfig_i remains unchanged, the same instance of sysconfig_i is
        returned, otherwise a new object is created.
        """

        sysconfig_i_schedules: Dict[
            str, ScheduleSelection
        ] = sysconfig_i.schedule_choices
        sysconfig_j_schedules: Dict[
            str, ScheduleSelection
        ] = sysconfig_j.schedule_choices
        key_set_i = set(sysconfig_i_schedules.keys())
        key_set_j = set(sysconfig_j_schedules.keys())

        new_sysconfig: Dict[str, ScheduleSelection] = {}
        modified = False

        for i, a in enumerate(sorted(key_set_i | key_set_j)):
            # An "a" might be in key_set_i, key_set_j or in both!
            if a in key_set_i and (
                a not in key_set_j
                or sysconfig_i_schedules[a].counter >= sysconfig_j_schedules[a].counter
            ):
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
    def _merge_candidates(
        candidate_i: SolutionCandidate,
        candidate_j: SolutionCandidate,
        agent_id: str,
        perf_func: Callable,
        target_params,
    ):
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
            candidate = SolutionCandidate(
                agent_id=agent_id, schedules=new_schedules, perf=None
            )
            candidate.perf = perf_func(candidate.cluster_schedule, target_params)

        return candidate


class CohdaNegotiationModel:
    """Model for storing all metadata regarding negotiations"""

    def __init__(self) -> None:
        self._negotiations: Dict[UUID, COHDANegotiation] = {}

    def by_id(self, negotiation_id: UUID) -> COHDANegotiation:
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

    def add(self, negotiation_id: UUID, cohda_negotiation: COHDANegotiation):
        """Add a concrete negotiation
        :param negotiation_id: the UUID of the negotiation
        :param cohda_negotiation: the cohda negotiation object
        """
        self._negotiations[negotiation_id] = cohda_negotiation


class CohdaSolutionModel:
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
