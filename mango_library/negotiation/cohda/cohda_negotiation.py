import asyncio
import logging
from typing import List, Dict, Optional, Tuple, Callable
from uuid import UUID
import numpy as np

from mango_library.coalition.core import CoalitionAssignment, CoalitionModel
from mango_library.negotiation.cohda.cohda_messages import CohdaNegotiationMessage, CohdaStopNegotiationMessage
from mango_library.negotiation.cohda.data_classes import WorkingMemory, SolutionCandidate, SystemConfig, \
    ScheduleSelection
from mango.role.api import Role

from mango_library.negotiation.core import NegotiationModel

logger = logging.getLogger(__name__)


class COHDANegotiationRole(Role):
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

    def setup(self) -> None:
        super().setup()
        self.context.subscribe_message(self, self.handle_neg_msg, lambda c, _: isinstance(c, CohdaNegotiationMessage))
        self.context.subscribe_message(self, self.handle_neg_stop,
                                       lambda c, _: isinstance(c, CohdaStopNegotiationMessage))
        # self.context.subscribe_message(self, self.handle_neg_stop,
        #                                lambda c, _: isinstance(c, CohdaRequestSolutionMessage))

    def handle_msg(self, content, meta: Dict[str, Any]) -> None:
        """Handles NegotiationMessage, StopNegotiationMessage, RequestSolutionMessage

          :param content: the message
          :param meta: meta
          """
        if isinstance(content, NegotiationMessage):
            if not self.context.get_or_create_model(CoalitionModel).exists(content.coalition_id):
                return

            assignment = self.context.get_or_create_model(
                CoalitionModel).by_id(content.coalition_id)
            negotiation_model = self.context.get_or_create_model(NegotiationModel)

            if not negotiation_model.exists(content.negotiation_id):
                negotiation_model.add(content.negotiation_id, Negotiation(
                    content.coalition_id, content.negotiation_id))

            self.handle_neg_msg(content.message, assignment,
                        negotiation_model.by_id(content.negotiation_id), meta)

        elif isinstance(content, StopNegotiationMessage):

            # set stopped
            negotiation_model = self.context.get_or_create_model(NegotiationModel)
            if not negotiation_model.exists(content.negotiation_id):
                negotiation_model.add(content.negotiation_id, Negotiation(
                    content.coalition_id, content.negotiation_id))
            negotiation_model.by_id(content.negotiation_id).stopped = True
            self.context.schedule_instant_task(
                self.handle_neg_stop(negotiation=negotiation_model.by_id(content.negotiation_id), meta=meta)
            )

        else:
            logger.warning(f'NegotiationParticipantRole received unexpected Message of type {type(content)}')


    def create_cohda(self, part_id: str):
        """
        Create an instance of COHDA.
        :param part_id: participant id
        :return: COHDA object
        """
        return COHDANegotiation(schedule_provider=self._schedules_provider,
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

    def handle_neg_msg(self, content: CohdaNegotiationMessage, meta: Dict):
        """
        Will be called when a CohdaNegotiationMessage is received
        :param content: CohdaNegotiationMessage
        :param meta: meta dict
        """

        # check if there is a Coalition with the coalition_ID
        if not self.context.get_or_create_model(CoalitionModel).exists(content.coalition_id):
            logger.warning(f'Received a CohdaNegotiationMessage with the coalition_id {content.coalition_id}'
                           f'but there is no such Coalition known.')
            return

        # get coalition_assignment
        coalition_assignment: CoalitionAssignment = self.context.get_or_create_model(
            CoalitionModel).by_id(content.coalition_id)
        #get negotiation
        cohda_negotiation_model: CohdaNegotiationModel = self.context.get_or_create_model(CohdaNegotiationModel)

        if not cohda_negotiation_model.exists(content.negotiation_id):
            cohda_negotiation_model.add(
                negotiation_id = content.negotiation_id,
                assignment=coalition_assignment,
                COHDANegotiation(
                content.coalition_id, content.negotiation_id))

        self.handle_neg_msg(content.message, coalition_assignment,
                            negotiation_model.by_id(content.negotiation_id), meta)




        if not negotiation.stopped:
            if negotiation.negotiation_id in self._cohda:
                if not negotiation.active:
                    print(f'[{self.context.addr, self.context.aid}] ATTENTION, negotiation was not active and i received a mssage')
                negotiation.active = True
                self._cohda_msg_queues[negotiation.negotiation_id].append(message)
            else:
                self._cohda[negotiation.negotiation_id] = self.create_cohda(coalition_assignment.part_id)
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
                            await self.send_to_neighbors(coalition_assignment, negotiation, message_to_send)

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

    async def send_to_neighbors(self, assignment: CoalitionAssignment, negotation: Negotiation, message):
        """Send a message to all neighbors

        :param assignment: the coalition you want to use the neighbors of
        :param negotation: the negotiation message
        :param message: the message you want to send
        """
        for neighbor in assignment.neighbors:
            await self.context.send_message(
                content=NegotiationMessage(negotation.coalition_id, negotation.negotiation_id, message),
                receiver_addr=neighbor[1], receiver_id=neighbor[2],
                acl_metadata={'sender_addr': self.context.addr, 'sender_id': self.context.aid},
                create_acl=True)

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

        async def handle_solution_request(negotiation, meta):
            pass


class COHDANegotiation:
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

    def handle_cohda_msgs(self, messages: List[CohdaNegotiationMessage]) -> Optional[CohdaNegotiationMessage]:
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

    def _perceive(self, messages: List[CohdaNegotiationMessage]) -> Tuple[SystemConfig, SolutionCandidate]:
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
            current_candidate = self._merge_candidates(
                candidate_i=current_candidate,
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

    def _act(self, new_sysconfig: SystemConfig, new_candidate: SolutionCandidate) -> CohdaNegotiationMessage:
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
        return CohdaNegotiationMessage(working_memory=self._memory)

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


class CohdaNegotiationModel:
    """Model for storing all metadata regarding negotiations
    """

    def __init__(self) -> None:
        self._negotiations = {}

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
        return negotiation_id in self._negotiations

    def add(self, negotiation_id: UUID, assignment: Negotiation):
        """Add a concrete negotiation

        :param negotiation_id: the UUID of the negotiation
        :param assignment: the assignment for the negotiation
        """
        self._negotiations[negotiation_id] = assignment
