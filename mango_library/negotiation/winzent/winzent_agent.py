import asyncio
import logging
import math
import uuid
from copy import deepcopy

from mango.core.agent import Agent

from mango_library.negotiation.winzent import xboole
from mango_library.negotiation.winzent.winzent_message_pb2 import WinzentMessage

logger = logging.getLogger(__name__)


class WinzentAgent(Agent):
    def __init__(self, container, ttl, time_to_sleep=2, send_message_paths=False):
        super().__init__(container)
        print("Version from 26.07.2022 18:05")

        # PGASC: if true stores the message path in the message
        self.send_message_paths = send_message_paths
        self.negotiation_connections = {}  # message paths of all agents that have established a connection in
        # the negotiation

        self.messages_sent = 0

        # store flexibility as interval with maximum and minimum value per time
        self.flex = {}
        self.original_flex = {}
        self._adapted_flex_according_to_msgs = []

        # create Governor with PowerBalance
        self.governor = xboole.Governor()
        self.governor.power_balance = xboole.PowerBalance()
        self.governor.power_balance_strategy = \
            xboole.XboolePowerBalanceSolverStrategy()

        # in final, the result for a disequilibrium is stored
        self.final = {}

        # PGASC in result, the final negotiated (accepted and acknowledged) result is saved
        self.result = {}
        self.result_sum = 0
        # store other agents as neighbors in a list
        self.neighbors = {}

        # some parameters necessary for a negotiation
        self._solution_found = False  # True if there is already a solution for the current problem
        self._negotiation_running = False  # is currently a negotiation running
        self._current_inquiries_from_agents = {}  # the inquiries the agent received from others for the current problem
        self._curr_sent_acceptances = []  # the acceptance notifications sent for the current problem
        self._acknowledgements_sent = []  # the id of the acceptance notifications to with an acknowledgement was sent
        self._waiting_for_acknowledgements = False  # True if the agent is still waiting for acknowledgements
        self.negotiation_done = None  # True if the negotiation is done
        self._own_request = None  # the agent stores its own request when it starts a negotiation
        self._current_ttl = ttl  # the current time to live for messages, indicates how far messages will be forwarded
        self._time_to_sleep = time_to_sleep  # time to sleep between regular tasks
        self._unsuccessful_negotiations = []  # negotiations for which no result at all was found

        # tasks which should be triggered regularly
        self.tasks = []
        task_settings = [
            (self.trigger_solver),
        ]
        for trigger_fkt in task_settings:
            t = asyncio.create_task(trigger_fkt())
            t.add_done_callback(self.raise_exceptions)
            self.tasks.append(t)

    @property
    def solution_found(self):
        """
        True, if a complete solution for a negotiation was found.
        """
        return self._solution_found

    @property
    def unsuccessful_negotiations(self):
        """
        Return negotiations without results. Can be used to restart those.
        """
        return self._unsuccessful_negotiations

    @property
    def time_to_sleep(self):
        """
        Return time_to_sleep.
        """
        return self._time_to_sleep

    @time_to_sleep.setter
    def time_to_sleep(self, time_to_sleep):
        """
        Adapt value for time_to_sleep.
        """
        self._time_to_sleep = time_to_sleep

    def update_time_to_live(self, ttl):
        """
        Method to update time to live which is sent in messages. This can be useful when the
        number of agents in the network decreases or increases.
        :param ttl: time to live, integer
        """
        self._current_ttl = ttl

    def add_neighbor(self, aid, addr):
        """
        Add another agent to list of neighbors with agent id (aid) and the
        address of the agent.
        """
        self.neighbors[aid] = addr

    def delete_neighbor(self, aid):
        """
        Delete an agent from the list of neighbors with agent id (aid)
        """
        self.neighbors.pop(aid, None)

    def update_flexibility(self, t_start, min_p, max_p):
        """
        Update the own flexibility. Flexibility is a range from power from
        min_p to max_p for a given time interval beginning with t_start.
        """
        self.flex[t_start] = [min_p, max_p]

    async def start_negotiation(self, ts, value):
        """
        Start a negotiation with other agents for the given timestamp and 
        value. The negotiation is started by calling handle_internal_request.
        :param ts: timespan for the negotiation
        :param value: power value to negotiate about
        """
        if not isinstance(value, int):
            value = value[1]

        requirement = xboole.Requirement(
            xboole.Forecast((ts, math.ceil(value))), ttl=self._current_ttl)
        requirement.from_target = True
        requirement.message.sender = self._aid

        message = requirement.message
        message.sender = self._aid
        self.governor.message_journal.add(message)
        self.governor.curr_requirement_value = value
        self.negotiation_done = asyncio.Future()

        await self.handle_internal_request(requirement)
        self.governor.diff_to_real_value = 1 - (message.value[0] % 1)

    async def trigger_solver(self):
        """
        The agent always sleeps for a certain amount of time and then triggers
        the solver.
        The same happens for the acknowledgements. If the solution is found,
        the agent waits for the acknowledgements of the other agents to
        finalize the negotiation. If after a certain amount of time the
        acknowledgements are not there, the agent declares the negotiation
        result as no longer valid and does not wait for acknowledgements anymore.
        The time_to_sleep needs to be set according to the network size of the
        agents.
        """
        while not self.stopped.done():
            await asyncio.sleep(0.1)
            if self._negotiation_running:
                await asyncio.sleep(self._time_to_sleep)
                # After sleeping, the solver is triggered. This is necessary
                # in case when not the complete negotiation problem can be
                # solved. The solver is triggered after the timeout to
                # determine the solution according to the power that
                # is available.
                self.governor.triggered_due_to_timeout = True
                await self.solve()
                self._negotiation_running = False
            if self._waiting_for_acknowledgements:
                await asyncio.sleep(self._time_to_sleep)
                # Time for waiting for acknowledgements is done, therefore
                # do not wait for acknowledgements anymore
                logger.debug(
                    f"*** {self._aid} did not receive all acknowledgements. Negotiation was not successful."
                )
                # PGASC add logging
                logger.debug(
                    "trigger_solver: stopped waiting for acknowledgments"
                )
                # PGASC adaption: reset final so unsuccessful negotiations are removed from final
                # self.final = {}

                self._waiting_for_acknowledgements = False
                for acc_msg in self._curr_sent_acceptances:
                    withdrawal = WinzentMessage(time_span=acc_msg.time_span,
                                                is_answer=True, answer_to=acc_msg.id,
                                                msg_type=xboole.MessageType.WithdrawalNotification,
                                                ttl=self._current_ttl, receiver=acc_msg.receiver,
                                                value=[acc_msg.value[0]],
                                                id=str(uuid.uuid4()),
                                                sender=self._aid
                                                )
                    if self.send_message_paths:
                        await self.send_message(withdrawal, message_path=self.negotiation_connections[acc_msg.receiver])
                    else:
                        await self.send_message(withdrawal, receiver=acc_msg.receiver)
                await asyncio.sleep(self._time_to_sleep)
                await self.reset()
                self._unsuccessful_negotiations.append([self._own_request.time_span, self._own_request.value])
                self.flex[self._own_request.time_span[0]] = self.original_flex[self._own_request.time_span[0]]
                self._curr_sent_acceptances = []

    async def handle_internal_request(self, requirement):
        """
        The negotiation request is for this agents. Therefore, it handles an
        internal request and not a request from other agents. This is the
        beginning of a negotiation, because messages to the neighboring agents
        are sent regarding the negotiation information in the given
        requirement.
        """
        message = requirement.message
        self.original_flex = deepcopy(self.flex)
        value = self.get_flexibility_for_interval(t_start=message.time_span[0], msg_type=message.msg_type)

        if abs(message.value[0]) - abs(value) <= 0:
            # PGASC add logging
            logger.debug(
                f"handle_internal_request: {self.aid} has sufficient flexibility to solve own requirements"
            )
            # If the own forecast is sufficient to completely solve the
            # problem, a solution is found and no other agents are informed.
            self.final[self._aid] = abs(message.value[0])
            self._solution_found = True
            if abs(message.value[0]) - abs(value) == 0:
                new_flex = 0
            else:
                new_flex = value - message.value[0]

            if message.msg_type == xboole.MessageType.DemandNotification:
                self.flex[message.time_span[0]][0] = new_flex
            else:
                self.flex[message.time_span[0]][1] = new_flex
            return

        # PGASC add logging
        logger.debug(
            f"handle_internal_request: {self.aid} own forecast not sufficient, needs help"
        )
        if message.msg_type == xboole.MessageType.DemandNotification:
            self.flex[message.time_span[0]][0] = 0
        else:
            self.flex[message.time_span[0]][1] = 0

        # In this case, there is still a value to negotiate about. Therefore,
        # add the message regarding the request to the own message journal
        # and store the problem in the power balance.
        message.value[:] = [message.value[0] - value]
        requirement.message = message
        requirement.forecast.second = message.value[0]
        self.governor.message_journal.add(message)
        requirement.from_target = True
        self.governor.power_balance.add(requirement)

        self.governor.power_balance.help_requested(
            requirement.time_span, requirement)

        # create a negotiation request to send it to other agents
        neg_msg = WinzentMessage(is_answer=False,
                                 msg_type=requirement.message.
                                 msg_type,
                                 ttl=self._current_ttl,
                                 value=message.value,
                                 id=message.id,
                                 time_span=requirement.time_span,
                                 sender=self._aid,
                                 )
        # PGASC add logging
        logger.debug(
            f"handle_internal_request: {self.aid} sends negotiation Request"
            f"with value: {neg_msg.value[0]} and type: {neg_msg.msg_type}"
        )
        self._own_request = requirement.message
        self._negotiation_running = True
        logger.debug(f"{self.aid} sends negotiation start notification")
        await self.send_message(neg_msg)

    def get_flexibility_for_interval(self, t_start, msg_type):
        """
        Returns the flexibility for the given time interval according
        to the msg type.
        """
        flexibility = self.flex[t_start]
        if msg_type == xboole.MessageType.OfferNotification:
            # in this case, the upper part of the flexibility interval
            # is considered
            return flexibility[1]
        elif msg_type == xboole.MessageType.DemandNotification:
            # in this case, the lower part of the flexibility interval
            # is considered
            return flexibility[0]

    def exists_flexibility(self, t_start):
        """
        Returns whether there exists flexibility for the given time interval.
        """
        return t_start in self.flex.keys()

    async def stop_agent(self):
        """
        Method to stop the agent externally.
        """
        for task in self.tasks:
            try:
                task.remove_done_callback(self.raise_exceptions)
                task.cancel()
                await task
            except asyncio.CancelledError:
                pass

    def should_withdraw(self, message):
        """
        Check whether the agent should withdraw its own negotiation request according to the given method. True, if
        for the same interval the opposite request with the same amount was sent.
        """
        if message.msg_type == xboole.MessageType.OfferNotification and self._own_request.msg_type == \
                xboole.MessageType.DemandNotification or message.msg_type == xboole.MessageType.DemandNotification \
                and self._own_request.msg_type == xboole.MessageType.OfferNotification:
            if message.value[0] >= self._own_request.value[0]:
                return True
        return False

    async def handle_external_request(self, requirement, message_path=None):
        """
        The agent received a negotiation request from another agent.
        """
        if message_path is None:
            message_path = []

        message = requirement.message
        # There is already a negotiation running
        if self._negotiation_running and self._own_request.time_span == message.time_span:
            # PGASC add logging
            logger.debug(
                f"handle_external_request: {self.aid} negotiation is already running"
            )
            # first, check whether the value fulfills the own request
            if self.should_withdraw(message):
                logger.debug(
                    f"handle_external_request: {self.aid} should withdrawal"
                )
                withdrawal = WinzentMessage(time_span=self._own_request.time_span,
                                            is_answer=True, answer_to=self._own_request.id,
                                            msg_type=xboole.MessageType.WithdrawalNotification,
                                            ttl=self._current_ttl, receiver=message.sender,  # PGASC: added sender
                                            # because this message will be sent endlessly otherwise
                                            value=self._own_request.value,
                                            id=str(uuid.uuid4()),
                                            sender=self._aid
                                            )
                if self.send_message_paths:
                    await self.send_message(withdrawal, message_path=self.negotiation_connections[withdrawal.receiver])
                else:
                    await self.send_message(withdrawal)
                await self.reset()
            else:
                # no withdrawal, still keep the negotiation running
                return
        # PGASC add logging
        logger.debug(
            f"{self.aid} received message with python object id={id(message)} and ttl={message.ttl}"
        )
        # PGASC add logging
        logger.debug(
            f"message content: {message.msg_type}, {message.value[0]}, {message.sender}, {message.receiver}, "
            f"{message.is_answer} "
        )
        value = 0
        if self.exists_flexibility(
                message.time_span[0]):
            logger.debug(f"{self.aid} has flexibility")
            # If the agent has flexibility for the requested time, it replies
            # to the requesting agent
            value = self.get_flexibility_for_interval(
                t_start=message.time_span[0],
                msg_type=message.msg_type)
            if value != 0:
                # PGASC add logging
                msg_type = xboole.MessageType.Null
                # send message reply
                if message.msg_type == xboole.MessageType.OfferNotification:
                    msg_type = xboole.MessageType.DemandNotification
                elif message.msg_type == xboole.MessageType. \
                        DemandNotification:
                    msg_type = xboole.MessageType.OfferNotification
                reply = WinzentMessage(msg_type=msg_type,
                                       sender=self._aid,
                                       is_answer=True,
                                       receiver=message.sender,
                                       time_span=message.time_span,
                                       value=[value], ttl=self._current_ttl,
                                       id=str(uuid.uuid4()))
                self.governor.message_journal.add(reply)
                self._current_inquiries_from_agents[reply.id] = reply
                if self.send_message_paths:
                    message_path.append(self.aid)
                    message_path.reverse()
                    if message_path is not None:
                        demander_index = message_path[-1]
                        self.negotiation_connections[
                            demander_index] = message_path
                        # send offer and save established connection demander:[self.aid/supplier, ..., demander]
                    else:
                        logger.error("message path none")
                    logger.debug(f"{self.aid} sends Reply to Request to {reply.receiver} on path: {message_path}")
                    await self.send_message(reply, message_path=message_path)
                else:
                    await self.send_message(reply)
        message.ttl -= 1
        if message.ttl <= 0:
            # PGASC add logging
            logger.debug(
                f"handle_external_request: {self.aid} does not forward the message to other agents because ttl<=0"
            )
            # do not forward the message to other agents
            return
        # PGASC add logging
        logger.debug(
            f"handle_external_request: {self.aid} forward request to other agents ttl={message.ttl}"
        )

        if abs(message.value[0]) - abs(value) == 0:
            # PGASC add logging
            logger.debug(
                f"handle_external_request: {self.aid} does not forward the message to other agents "
                f"because value is completely fulfilled"
            )
            # The value in the negotiation request is completely fulfilled.
            # Therefore, the message is not forwarded to other agents.
            return
        # In this case, the power value of the request cannot be completely
        # fulfilled yet. Therefore, the remaining power of the request is
        # forwarded to other agents.
        val = message.value[0]
        del message.value[:]
        message.value.append(val - value)
        message.is_answer = False
        message.receiver = ''
        logger.debug(
            f"demand not fulfilled yet, send {message.msg_type} request further through neighbors, path {message_path}")
        if self.send_message_paths:
            await self.send_message(message, message_path=message_path)
        else:
            await self.send_message(message)

    async def flexibility_valid(self, reply):
        """
        Checks whether the requested flexibility value in reply is valid (less than or equal to the stored
        flexibility value for the given interval).
        """
        message_type = self._current_inquiries_from_agents[reply.answer_to].msg_type

        if message_type == xboole.MessageType.DemandNotification:
            valid = abs(self.flex[reply.time_span[0]][1]) >= abs(reply.value[0])
            if valid:
                self.original_flex = deepcopy(self.flex)
                self.flex[reply.time_span[0]][1] = self.flex[reply.time_span[0]][1] - reply.value[0]
        else:
            valid = abs(self.flex[reply.time_span[0]][0]) >= abs(reply.value[0])
            if valid:
                self.original_flex = deepcopy(self.flex)
                self.flex[reply.time_span[0]][0] = self.flex[reply.time_span[0]][0] - reply.value[0]
        return valid

    async def handle_external_reply(self, requirement, message_path=None):
        """
        Handle a reply from other agents. Reply may be from types:
        DemandNotification, OfferNotification, AcceptanceNotification,
        AcceptanceAcknowledgementNotification
        """
        if message_path is None:
            message_path = []

        reply = requirement.message
        # PGASC add logging
        logger.debug(f"receiver of this reply is {reply.receiver}")
        if reply.receiver != self._aid:
            # The agent is not the receiver of the reply, therefore it needs
            # to forward it if the time to live is above 0.

            # PGASC add logging
            logger.debug(
                f"handle_external_reply: {self.aid} is not the receiver; "
                f"received type={reply.msg_type} with value={reply.value} "
                f"from {reply.sender} to {reply.receiver} with ttl={reply.ttl}"
            )
            reply.ttl = reply.ttl - 1
            if reply.ttl > 0:
                logger.debug(
                    f"{self.aid} is not receiver {reply.receiver} of reply {reply.msg_type}, forward; "
                    f"path {message_path}")
                if self.send_message_paths:
                    await self.send_message(reply, message_path=message_path)
                else:
                    await self.send_message(reply)
            return

        if reply.msg_type == xboole.MessageType.DemandNotification \
                or reply.msg_type == xboole.MessageType.OfferNotification:
            # The agent received an offer or demand notification as reply.
            # If the power_balance is empty, the reply is not considered
            # because the negotiation is already done.
            if self.governor.power_balance.empty():
                return
            # If there is no solution found already, the reply is considered
            # to find a new solution. Therefore, trigger solver.
            if not self._solution_found:
                self.governor.power_balance.add(requirement)
                if not self.governor.solver_triggered:
                    self.governor.triggered_due_to_timeout = False
                # Save the established connection
                if self.send_message_paths:
                    message_path.reverse()
                    self.negotiation_connections[
                        message_path[-1]] = message_path  # received offer; establish connection
                    # supplier:[self.aid/demander, ..., supplier]
                await self.solve()

        elif reply.msg_type == xboole.MessageType.AcceptanceNotification:
            # First, check whether the AcceptanceNotification is still valid
            if self.acceptance_valid(reply):
                # Send an AcceptanceAcknowledgementNotification for the
                # acceptance
                if await self.flexibility_valid(reply):
                    answer = WinzentMessage(
                        msg_type=xboole.MessageType.AcceptanceAcknowledgementNotification,
                        is_answer=True, answer_to=reply.id,
                        sender=self._aid, receiver=reply.sender,
                        value=reply.value,  # PGASC added value to AAN messages to confirm the results
                        ttl=self._current_ttl, id=str(uuid.uuid4()))
                    if self.send_message_paths:
                        await self.send_message(answer,
                                                message_path=self.negotiation_connections[answer.receiver])
                    else:
                        await self.send_message(answer)
                    self._adapted_flex_according_to_msgs.append(reply.id)
                    self._acknowledgements_sent.append(reply.id)

                    del self._current_inquiries_from_agents[reply.answer_to]
            return
        elif reply.msg_type == \
                xboole.MessageType.AcceptanceAcknowledgementNotification:
            # If there is no message in solution journal or
            # the solution journal does not contain this message, it is
            # irrelevant
            if self.governor.solution_journal.is_empty():
                return
            if not self.governor.solution_journal.contains_message(
                    reply.answer_to):
                return
            # Remove the Acknowledgement from the solution journal
            self.governor.solution_journal.remove_message(reply.answer_to)

            # PGASC: Save the acknowledged value in result
            if self.acknowledgement_valid(reply):
                if not self.solution_overshoots_requirement(reply):
                    self.save_accepted_values(reply)
                else:
                    logger.info(f"{self._aid} has thrown out reply {reply.value}")
                    withdrawal = WinzentMessage(time_span=self._own_request.time_span,
                                                is_answer=True, answer_to=self._own_request.id,
                                                msg_type=xboole.MessageType.WithdrawalNotification,
                                                ttl=self._current_ttl, receiver=reply.sender,  # PGASC: added sender
                                                # because this message will be sent endlessly otherwise
                                                value=self._own_request.value,
                                                id=str(uuid.uuid4()),
                                                sender=self._aid
                                                )
                    if self.send_message_paths:
                        await self.send_message(withdrawal,
                                                message_path=self.negotiation_connections[withdrawal.receiver])
                    else:
                        await self.send_message(withdrawal)
            else:
                logger.debug(
                    f"{self.aid} received an AcceptanceAcknowledgement (from {reply.sender} with value {reply.value}) "
                    f"was not valid "
                )

            # if the solution journal is empty afterwards, the agent does not
            # wait for any further acknowledgments and can stop the negotiation
            if self.governor.solution_journal.is_empty():
                # PGASC changed logger.info to logging
                logger.debug(f'\n*** {self._aid} received all Acknowledgements. ***')
                await self.reset()

        elif reply.msg_type == xboole.MessageType.WithdrawalNotification:
            # if the id is not saved, the agent already handled this
            # WithdrawalNotification
            if reply.answer_to in self._acknowledgements_sent:
                # Withdraw flexibility for this interval, therefore
                # it is possible to participate in a negotiation for
                # this time span
                if reply.answer_to in self._adapted_flex_according_to_msgs:
                    self._acknowledgements_sent.remove(reply.answer_to)
                    self.flex[reply.time_span[0]] = self.original_flex[reply.time_span[0]]
                    self._adapted_flex_according_to_msgs.remove(reply.answer_to)

                    # PGASC add logging
                    logger.debug(
                        f"{self.aid}/{reply.receiver} gets withdrawal message from {reply.sender} with value "
                        f"{reply.value} "
                    )
                else:
                    pass
                    logger.debug(
                        f"Withdrawal received: {reply.answer_to} not in {self._adapted_flex_according_to_msgs}"
                    )
            else:
                logger.debug(
                    f"{self.aid} received Withdrawal from {reply.sender} with answer_to {reply.answer_to} and id "
                    f"{reply.id} which is not in {self._acknowledgements_sent} "
                )

    def solution_overshoots_requirement(self, reply) -> bool:
        if (self.result_sum + reply.value[0]) > self.governor.curr_requirement_value:
            return True
        return False

    def acknowledgement_valid(self, reply) -> bool:
        """
        Checks if the Acknowledgment is a reply to a current AcceptanceNotification
        :param reply:
        :return:
        """
        acceptance_msg = None
        for acc_msg in self._curr_sent_acceptances:
            if acc_msg.id == reply.answer_to:
                acceptance_msg = acc_msg

        if acceptance_msg is not None:
            self._curr_sent_acceptances.remove(acceptance_msg)
            return True
        else:
            logger.debug(
                "AcceptanceAcknowledgement was sent without there being a current Acceptance"
            )
            return False

    def save_accepted_values(self, message):
        # PGASC add logging
        logger.debug(
            f"AcceptanceAcknowledgeNotification: load {message.receiver} ({self.aid}) gets {message.value[0]} "
            f"from sgen {message.sender}"
        )
        if message.sender not in self.result.keys():
            self.result[message.sender] = 0
        self.result[message.sender] += message.value[0]
        self.result_sum += message.value[0]

    async def reset(self):
        """
        After a negotiation, reset the negotiation parameters.
        """
        self._negotiation_running = False
        self._solution_found = False
        self._waiting_for_acknowledgements = False
        self.governor.power_balance.clear()
        self._curr_sent_acceptances = []
        if not self.negotiation_done.done():
            self.negotiation_done.set_result(True)
        self.result_sum = 0
        self._acknowledgements_sent = []
        self.negotiation_connections = {}

    def acceptance_valid(self, msg):
        """
        Returns whether the message is still valid by checking whether it is
        in the current inquiries the agent received from others
        """
        return msg.answer_to in self._current_inquiries_from_agents.keys()

    async def answer_requirements(self, solution, gcd, initial_req):
        """
        Method to send out AcceptanceNotifications for the agents being
        part of the solution.
        """
        answer_objects = []
        initial_value = initial_req.forecast.second
        logger.info(f"initial value is {initial_value}")
        initial_msg_type = initial_req.message.msg_type
        # determine flexibility sign according to msg type
        positive = False if initial_msg_type == xboole.MessageType.DemandNotification else True
        afforded_value = 0

        # create dict with agents and used power value per agent
        for j in range(len(solution)):
            answer_objects.append(solution[j].split(':', 1)[0])
        sol = DictList()
        for j in range(len(answer_objects)):
            sol.add((answer_objects[j], gcd[int(solution[j][-1])]))
        self.final = {}
        for k, v in sol.values.copy().items():
            if abs(afforded_value) + abs(v) >= abs(initial_value):
                diff = abs(initial_value) - abs(afforded_value)
                self.final[self.governor.message_journal.get_message_for_id(
                    k).sender] = diff
                afforded_value += diff
                break
            self.final[self.governor.message_journal.get_message_for_id(
                k).sender] = v
            afforded_value += v
        if positive:
            act_value = afforded_value
        else:
            act_value = -afforded_value
        # the problem was not solved completely
        if abs(afforded_value) < abs(initial_value):
            # problem couldn't be solved, but the timer is still running:
            # we didn't receive the flexibility from every
            # agent
            logger.debug(
                f'*** {self._aid} has not enough flexibility. Timeout? '
                f'{self.governor.triggered_due_to_timeout} ***')
            if not self.governor.triggered_due_to_timeout:
                # Solver is not triggered currently and can be triggered again
                self.governor.solver_triggered = False
                return
            else:
                # In this case, the problem could not be solved completely,
                # but the timer stopped and the agent would not receive
                # more flexibility. Therefore, take afforded flexibility
                # and send acknowledgements.
                if act_value == 0:
                    await self.no_solution_after_timeout()
                    self.governor.triggered_due_to_timeout = False
                    return
        i = 0
        zero_indeces = []

        for k, v in self.final.items():
            value = v
            if not positive:
                value = -v
            if v == 0:
                zero_indeces.append(k)
                continue
            self.final[k] = value
            i += 1
            if k == self._aid:
                if len(self.final) == 1:
                    # Only the agent itself is part of the solution
                    self.governor.solver_triggered = False
                    if self.governor.triggered_due_to_timeout:
                        self._negotiation_running = False
                        self.governor.triggered_due_to_timeout = False
                        await self.no_solution_after_timeout()
                    return
                continue

            self._solution_found = True

            # id of original negotiation request
            answer_to = self.find_id_for_sender(
                time_span=initial_req.forecast.first,
                receiver=k)
            if answer_to == '':
                if len(self.final) == 1:
                    self.governor.solver_triggered = False
                    if self.governor.triggered_due_to_timeout:
                        self.governor.triggered_due_to_timeout = False
                        await self.no_solution_after_timeout()

                else:
                    continue
            # create AcceptanceNotification
            msg = WinzentMessage(
                msg_type=xboole.MessageType.AcceptanceNotification,
                sender=self._aid,
                is_answer=True,
                receiver=k,
                time_span=initial_req.forecast.first,
                value=[self.final[k]], ttl=self._current_ttl,
                id=str(uuid.uuid4()),
                answer_to=answer_to)
            self._curr_sent_acceptances.append(msg)

            # store acceptance message
            self.governor.solution_journal.add(msg)
            if self.send_message_paths:
                logger.debug(f"receiver {msg.receiver} in connections {self.negotiation_connections}?")
                await self.send_message(msg, message_path=self.negotiation_connections[msg.receiver])
            else:
                await self.send_message(msg)
        for key in zero_indeces:
            del self.final[key]
        self._waiting_for_acknowledgements = True
        self.governor.solver_triggered = False
        self.governor.triggered_due_to_timeout = False

    def find_id_for_sender(self, time_span, receiver):
        """
        Returns the id of the original reply to the negotiation request for
         the given agent.
        """
        if self.governor.power_balance:
            for entry in self.governor.power_balance[1]:
                if entry.message.time_span == time_span \
                        and entry.message.sender == receiver:
                    return entry.message.id
        # default, if no entry is found
        return ''

    async def solve(self):
        """
        Trigger the solver and try to solve the problem.
        """
        # First, check whether the solver is currently triggered and if there
        # is already a solution
        if not self._negotiation_running:
            return
        if self.governor.solver_triggered or self._solution_found:
            return
        self.governor.solver_triggered = True
        # PGASC changed logger.info to logging
        logger.debug(f'\n*** {self._aid} starts solver now. ***')
        result = self.governor.try_balance()
        if result is None:
            self.governor.solver_triggered = False
            if self.governor.triggered_due_to_timeout:
                # solver was triggered after the timeout and yet there was
                # still no solution
                self.governor.triggered_due_to_timeout = False
                await self.no_solution_after_timeout()
            return
        solution = result[0]
        if len(solution) > 0:
            # There was actually a solution. Split solution values according
            # to agents taking part in it
            answers = []
            for k, v in solution.vv.items():
                if v[0] == xboole.Tval(1):
                    answers.append(k)
            gcd_p = result[1]
            if len(answers) > 0:
                # PGASC changed logger.info to logging
                logger.debug(f'\n*** {self._aid} found solution. ***')
                logger.info(f'\n*** {self._aid} found solution. ***')
                logger.info(f'\n*** {self._aid} found solution.{answers} is something. {gcd_p} is something else ***')
                await self.answer_requirements(answers, gcd_p, result[2])
                return

        if self.governor.triggered_due_to_timeout:
            self.governor.triggered_due_to_timeout = False
            await self.no_solution_after_timeout()

    async def no_solution_after_timeout(self):
        """
        No solution was found after the timeout. Negotiation is invalid
        and stopped.
        """
        if self._solution_found:
            return
        # PGASC changed logger.info to logging
        logger.debug(
            f'*** {self._aid} has no solution after timeout. ***')
        self._unsuccessful_negotiations.append([self._own_request.time_span, self._own_request.value])
        self.flex[self._own_request.time_span[0]] = self.original_flex[self._own_request.time_span[0]]
        self._negotiation_running = False
        self.governor.solver_triggered = False
        self.governor.triggered_due_to_timeout = False
        self._solution_found = False
        self.governor.power_balance.clear()
        self.governor.solution_journal.clear()
        self._waiting_for_acknowledgements = False
        await self.reset()

    def handle_msg(self, content, meta):
        """
        Handle message object (content) from other agents.
        """
        if content.msg_type == xboole.MessageType. \
                WithdrawalNotification:
            # withdraw the message the content refers to
            self.governor.message_journal.remove_message(
                content.answer_to)
        if not self.governor.message_journal.contains_message(
                content.id):
            self.governor.message_journal.add(content)
            if content.is_answer:
                req = xboole.Requirement(content,
                                         content.sender, ttl=self._current_ttl)
                asyncio.create_task(self.handle_external_reply(req, message_path=meta["ontology"]))
            else:
                req = xboole.Requirement(content,
                                         content.sender, ttl=self._current_ttl)
                asyncio.create_task(self.handle_external_request(req, message_path=meta["ontology"]))

    async def send_message(self, winzent_message, receiver=None, message_path=None):
        """
        Sends the given message to all neighbors unless the receiver is given.
        """
        if message_path is None:
            message_path = []

        logger.debug(
            f"*** {self._aid} sends message with type {winzent_message.msg_type}. ***")

        if self.send_message_paths:
            # for first connection establishment (demand notifications) append aid of this agent to the message path
            if not winzent_message.is_answer:
                if self.aid in message_path:
                    # remove old message loops
                    index_of_self = message_path.index(self.aid)
                    message_path = message_path[0:index_of_self - 1]
                message_path.append(self.aid)

            # sending over the message path, so next receiver in the neighborhood is known
            else:
                logger.debug(
                    f"{self.aid} sends {winzent_message.msg_type} on the message path in message_path: {message_path}")
                if len(message_path) == 0:
                    logger.error("message_path has length zero")
                    return
                index_of_next_on_path = message_path.index(self.aid) + 1
                receiver = message_path[index_of_next_on_path]
                if receiver not in self.neighbors.keys():
                    logger.error(
                        f"message_path at {self.aid} with message_path {message_path} failed because "
                        f"receiver {receiver} not in {self.neighbors.keys()}")
                    return

        if receiver is not None and receiver in self.neighbors.keys():
            # receiver is a neighbor
            message = copy_winzent_message(winzent_message)
            self.messages_sent += 1
            await self._container.send_message(
                content=message, receiver_addr=self.neighbors[receiver],
                receiver_id=receiver,
                acl_metadata={'sender_addr': self._container.addr,
                              'sender_id': self._aid,
                              'ontology': message_path.copy()}, create_acl=True)
        else:
            # send message to every neighbor
            for neighbor in self.neighbors.keys():
                # PGASC added deep copy of messages - every neighbor gets own message object
                message = copy_winzent_message(winzent_message)
                if message.sender == neighbor:
                    continue
                if message.receiver is None:
                    message.receiver = ''
                self.messages_sent += 1
                await self._container.send_message(
                    content=message, receiver_addr=self.neighbors[neighbor],
                    receiver_id=neighbor,
                    acl_metadata={'sender_addr': self._container.addr,
                                  'sender_id': self._aid,
                                  'ontology': message_path.copy()},
                    # copy to avoid neighbors working on the same object
                    create_acl=True
                )


def copy_winzent_message(message: WinzentMessage) -> WinzentMessage:
    """PGASC fix: deep copy of winzent3 message object (otherwise two agents manipulate the same object and its ttl)"""
    return WinzentMessage(
        msg_type=message.msg_type,
        sender=message.sender,
        is_answer=message.is_answer,
        receiver=message.receiver,
        time_span=message.time_span,
        value=message.value[:],
        ttl=message.ttl,
        id=message.id,
        answer_to=message.answer_to,
    )


class DictList:
    def __init__(self):
        """
        DictList is a helper object.
        """
        self._values = {}

    def add(self, answer_tuple):
        if len(self._values.items()) == 0:
            self._values[answer_tuple[0]] = answer_tuple[1]
            return

        for k, v in self._values.copy().items():
            if k == answer_tuple[0]:
                if answer_tuple[1] > v:
                    self._values[k] = answer_tuple[1]
                return
            self._values[answer_tuple[0]] = answer_tuple[1]
        return self

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, values):
        self._values = values
