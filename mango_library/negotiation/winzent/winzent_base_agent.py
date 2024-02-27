import asyncio
import logging
import math
import uuid
from abc import ABC
from datetime import datetime

import numpy as np
from mango.agent.core import Agent

from mango_library.negotiation.winzent import xboole
from mango_library.negotiation.winzent.winzent_message_pb2 import WinzentMessage


logger = logging.getLogger(__name__)


class WinzentBaseAgent(Agent, ABC):
    """
    The base Winzent version with all its core functions.
    :param ttl: The time to live for a message. Defines the amount of times a message can be forwarded
                before being discarded.
    :param time_to_sleep The time agent gets to finish its negotiation
    :param send_message_paths Controls the activation of message path tracking
    :param ethics_score The ethics score of the agent, used in the child class WinzentEthicalAgent
    :param elem_type The type of grid component the agent manages
    :param index The index of the agent used for identification inside the agent network
    """
    def __init__(self, container, ttl, time_to_sleep=3, send_message_paths=False, ethics_score=1.0,
                 elem_type=None, index=-1):
        super().__init__(container)

        self.send_message_paths = send_message_paths
        self.negotiation_connections = {}  # message paths of all agents that have established a connection in
        # the negotiation

        self.messages_sent = 0

        # store flexibility as interval with maximum and minimum value per time
        self.flex = {}
        self.original_flex = {}
        self._adapted_flex_according_to_msgs = []
        self.ethics_score = ethics_score
        # create Governor with PowerBalance
        self.governor = xboole.Governor()
        self.governor.power_balance = xboole.PowerBalance()
        # The base agent is using the not-ethical solving strategy as a default.
        # This can be overriden by agent subclasses.
        self.governor.power_balance_strategy = \
            xboole.XboolePowerBalanceSolverStrategy()

        # in final, the result for a disequilibrium is stored
        self.final = {}

        self.elem_type = elem_type
        self.index = index

        # in result, the final negotiated (accepted and acknowledged) result is saved
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
        self._list_of_acknowledgements_sent = []
        self._waiting_for_acknowledgements = False  # True if the agent is still waiting for acknowledgements
        self.negotiation_done = None  # True if the negotiation is done
        self._own_request = None  # the agent stores its own request when it starts a negotiation
        self._current_ttl = ttl  # the current time to live for messages, indicates how far messages will be forwarded
        self._time_to_sleep = time_to_sleep  # time to sleep between regular tasks
        self._lock = asyncio.Lock()
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
        True if a complete solution for a negotiation was found.
        """
        return self._solution_found

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
        :param t_start The point in time of the flexibility
        :param min_p The minimum possible flexibility
        :param max_p The maximum possible flexibility
        """
        self.flex[t_start] = [min_p, max_p]
        self.original_flex[t_start] = [min_p, max_p]
        self._list_of_acknowledgements_sent.clear()
        self._current_inquiries_from_agents.clear()

    async def start_negotiation(self, start_dates, values):
        """
        Start a negotiation with other agents for the given timestamp and
        value. The negotiation is started by calling handle_internal_request.
        :param start_dates: timespan for the negotiation
        :param values: power value to negotiate about
        """
        values = [math.ceil(value) for value in values]
        self._solution_found = False
        requirement = xboole.Requirement(
            xboole.Forecast((start_dates, values)), ttl=self._current_ttl)
        requirement.from_target = True
        requirement.message.sender = self.aid
        message = requirement.message
        message.sender = self.aid
        self.governor.message_journal.add(message)
        self.governor.curr_requirement_values = values
        self.governor.solution_journal.clear()
        self.negotiation_done = asyncio.Future()
        for val in message.value:
            self.governor.diff_to_real_value.append(1 - (val % 1))
        await self.handle_internal_request(requirement)

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
        while not self._stopped.done():
            await asyncio.sleep(0.1)
            if self._negotiation_running:
                await asyncio.sleep(self._time_to_sleep)
                now = datetime.now()

                current_time = now.strftime("%H:%M:%S")
                logger.debug(f"{self.aid}: Timer ran out at =", current_time)
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
                    f"*** {self.aid} did not receive all acknowledgements. Negotiation was not successful."
                )
                # self.final = {}
                self._waiting_for_acknowledgements = False
                for acc_msg in self._curr_sent_acceptances:
                    withdrawal = WinzentMessage(time_span=acc_msg.time_span,
                                                is_answer=True, answer_to=acc_msg.id,
                                                msg_type=xboole.MessageType.WithdrawalNotification,
                                                ttl=self._current_ttl, receiver=acc_msg.receiver,
                                                value=[acc_msg.value[0]],
                                                id=str(uuid.uuid4()),
                                                sender=self.aid
                                                )
                    if self.send_message_paths:
                        await self.send_message(withdrawal, msg_path=self.negotiation_connections[acc_msg.receiver])
                    else:
                        await self.send_message(withdrawal, receiver=acc_msg.receiver)
                logger.info(f"{self.aid} reset because the waiting time for the remaining acknowledgements"
                            f" is over.")
                for acc in self._curr_sent_acceptances:
                    logger.info(f"{self.aid}: {acc.value[0]} from {acc.receiver} not received.")
                self._curr_sent_acceptances = []
                await self.reset()

    async def handle_internal_request(self, requirement):
        """
        The negotiation request is from this agents. Therefore, it handles an
        internal request and not a request from other agents. This is the
        beginning of a negotiation, because messages to the neighboring agents
        are sent regarding the negotiation information in the given
        requirement.
        :param requirement The internal requirement that is handled
        """
        message = requirement.message
        values = self.get_flexibility_for_interval(time_span=message.time_span, msg_type=message.msg_type)

        # for each value to negotiate about, check whether the request could be fulfilled internally completely.
        for idx in range(len(values)):
            if abs(message.value[idx]) - abs(values[idx]) <= 0:
                # If the own forecast is sufficient to completely solve the
                # problem, a solution is found and no other agents are informed.
                self.final[self.aid][idx] = [abs(val) for val in message.value]
                self._solution_found = True
                new_flex = []
                if abs(message.value[idx][0]) - abs(values[idx][0]) == 0:
                    new_flex.append(0)
                else:
                    new_flex.append(values[idx][0] - message.value[idx][0])

                if abs(message.value[idx][1]) - abs(values[idx][1]) == 0:
                    new_flex.append(0)
                else:
                    new_flex.append(values[idx][1] - message.value[idx][1])
            else:
                self._solution_found = False

            if message.msg_type == xboole.MessageType.DemandNotification:
                self.flex[message.time_span[idx]][0] = 0
            else:
                self.flex[message.time_span[idx]][1] = 0

        if self._solution_found:
            return

        # In this case, there is still a value to negotiate about. Therefore,
        # add the message regarding the request to the own message journal
        # and store the problem in the power balance.
        for idx in range(len(message.time_span)):
            message.value[idx] = message.value[idx] - values[idx]

        # In this case, there is still a value to negotiate about. Therefore,
        # add the message regarding the request to the own message journal
        # and store the problem in the power balance.
        requirement.message = message
        requirement.forecast.second = message.value
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
                                 sender=self.aid,
                                 ethics_score=self.ethics_score
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

    def get_flexibility_for_interval(self, time_span, msg_type=6):
        """
        Returns the flexibility for the given time interval according
        to the msg type.
        :param time_span The time span the flexibility is looked up for
        :param msg_type The type of message the flexibility is looked up for. Possible
                        message type values or 5 for negative flexibility (min_p) and
                        6 for positive flexibility. These numbers stand for OfferNotification
                        and DemandNotification
        """
        flex = []
        if not is_iterable(time_span):
            time_span = [time_span]
        for idx in range(len(time_span)):
            t_start = time_span[idx]
            if t_start in self.flex:
                flexibility = self.flex[t_start]
            else:
                flexibility = [0, 0]
            if msg_type == xboole.MessageType.OfferNotification:
                # in this case, the upper part of the flexibility interval
                # is considered
                flex.append(flexibility[1])
            elif msg_type == xboole.MessageType.DemandNotification:
                # in this case, the lower part of the flexibility interval
                # is considered
                flex.append(flexibility[0])
            else:
                flex.append(0)
        return flex

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

    async def answer_external_request(self, message, message_path, values, msg_type):
        """

        :param message: The message that is answered.
        :param message_path: The message path of the message to send the answer back
               on that exact same path.
        :param values: The power values that this message is answered with.
        :param msg_type: The type of the answer sent.
        :return:
        """
        reply = WinzentMessage(
            msg_type=msg_type,
            sender=self.aid,
            is_answer=True,
            receiver=message.sender,
            time_span=message.time_span,
            value=values,
            ttl=self._current_ttl,
            id=str(uuid.uuid4()),
            ethics_score=self.ethics_score
        )
        self.governor.message_journal.add(reply)
        self._current_inquiries_from_agents[reply.id] = reply
        if self.send_message_paths:
            message_path_copy = message_path.copy()
            message_path_copy.append(self.aid)
            message_path_copy.reverse()

            if message_path_copy:
                demander_index = message_path_copy[-1]
                self.negotiation_connections[demander_index] = message_path_copy
                # send offer and save established connection demander:[self.aid/supplier, ..., demander]
            else:
                logger.error("message path is empty")

            logger.debug(f"{self.aid} sends Reply to Request to {reply.receiver} on path: {message_path_copy}")
            await self.send_message(reply, msg_path=message_path_copy)
        else:
            await self.send_message(reply)
        return

    async def handle_external_request(self, requirement, message_path=None):
        """
        The method to handle incoming requests and see if they can be answered.
        :param requirement: The requirement that arrived.
        :param message_path: The message path of the message to send the answer back
               on that exact same path.
        """
        if message_path is None:
            message_path = []

        message = requirement.message
        logger.debug(
            f"{self.aid} received message with object id={id(message)} and ttl={message.ttl}\n"
            f"message content: {message.msg_type}, {message.value[0]}, {message.sender}, {message.receiver}, "
            f"{message.is_answer} "
        )
        if self._negotiation_running:
            await self.send_message(msg=message, forwarding=True)
            return
        # If the agent has flexibility for the requested time, it replies
        # to the requesting agent
        value_array = self.get_flexibility_for_interval(
            message.time_span,
            msg_type=message.msg_type
        )

        if not all(element == 0 for element in value_array):
            msg_type = xboole.MessageType.Null
            # send message reply
            if message.msg_type == xboole.MessageType.OfferNotification:
                msg_type = xboole.MessageType.DemandNotification
            elif message.msg_type == xboole.MessageType. \
                    DemandNotification:
                msg_type = xboole.MessageType.OfferNotification
            await self.answer_external_request(message, message_path, value_array, msg_type)
            # if there are still values remaining, forward them to other agents
            remaining_values = np.array(list(message.value)) - np.array(value_array)
            if not all(element == 0 for element in remaining_values):
                del message.value[:]
                for value in remaining_values:
                    message.value.append(value)
                await self.send_message(message, msg_path=message_path, forwarding=True)
        else:
            await self.send_message(message, msg_path=message_path, forwarding=True)

    def get_ethics_score(self, message):
        """
        Returns the ethics score of a message.
        :param message: The message containing the ethics score.
        :return: message.ethics_score: The ethics score of the message.
        """
        return message.ethics_score

    async def check_flex(self, reply, flex_to_pick, it):
        """
        Checks if the promised flexibility is still valid and corrects the flexibility if necessary.
        :param reply: The reply of the agent that wants to claim the promised flexibility.
        :param flex_to_pick: This value is either 0 if the lower flex is needed or 1 if the higher flex is needed.
        :param it: The index of the time span that is checked.
        :return: Boolean that is True when the promised flexibility is still valid.
                 It is False when the flexibility is not valid anymore.
        """
        distributed_value = 0
        for ack in self._list_of_acknowledgements_sent:
            if reply.time_span[it] in ack.time_span:
                value_index = ack.time_span.index(reply.time_span[it])
                distributed_value += ack.value[value_index]
                logger.debug(f"{self.aid} promised {ack.value[0]} to {ack.receiver}")
        if self.original_flex[reply.time_span[it]][flex_to_pick] - distributed_value == self.flex[reply.time_span[it]][
            flex_to_pick]:
            return True
        else:
            logger.info(
                f"{self.aid}: Current flex is not consistent with the values already distributed."
                f"Distributed value is {distributed_value} and original flex is "
                f"{self.original_flex[reply.time_span[it]][flex_to_pick]}."
                f"Current flex is {self.flex[reply.time_span[it]][flex_to_pick]}")
            logger.info(f"Attempting to fix flex...")
            self.flex[reply.time_span[it]][flex_to_pick] = self.original_flex[reply.time_span[it]][
                                                               flex_to_pick] - distributed_value
            if self.flex[reply.time_span[it]][flex_to_pick] < reply.value[it]:
                logger.info(f"{self.aid}: Acknowledgement to {reply.sender} cannot be sent, "
                            f"flex is {self.flex[reply.time_span[it]][flex_to_pick]}.")
                return False
            logger.info("Flex has been fixed and Acknowledgement can be sent.")
        return True

    async def flexibility_valid(self, reply):
        """
        Checks whether the requested flexibility value in reply is valid (less than or equal to the stored
        flexibility value for the given interval).
        :param reply: The reply that the validity of the flexibility is checked for.
        """
        valid_array = []
        for it in range(len(reply.time_span)):
            if reply.value[it] > 0:
                flex_to_pick = 1
            else:
                flex_to_pick = 0
            valid_array.append(
                abs(self.flex[reply.time_span[it]][flex_to_pick]) >= abs(reply.value[it]) and await self.check_flex(
                    reply, flex_to_pick, it))
            if valid_array[it]:
                self.flex[reply.time_span[it]][flex_to_pick] = \
                    self.flex[reply.time_span[it]][flex_to_pick] - reply.value[it]
        return True if all(valid_array) else False

    async def handle_initial_reply(self, requirement, message_path):
        """
        Adds the reply to the power balance and attempts a solution creation.
        :param requirement: The requirement that is added to the power balance.
        :param message_path: The path of the message containing the requirement.
        :return:
        """
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
                    message_path[-1]] = message_path
            await self.solve()

    async def handle_acceptance_reply(self, reply):
        """
        Checks if an AcceptanceNotification is still valid. If so, then it answers it with
        an AcceptanceAcknowledgementNotification.
        :param reply: The reply that the validity is checked for.
        :return:
        """
        # First, check whether the AcceptanceNotification is still valid
        if self.acceptance_valid(reply):
            async with self._lock:
                flex_valid = await self.flexibility_valid(reply)
            # after checking for the validity of the reply, delete it to make duplicates invalid
            del self._current_inquiries_from_agents[reply.answer_to]
            if flex_valid:
                # Send an AcceptanceAcknowledgementNotification for the acceptance
                answer = WinzentMessage(
                    msg_type=xboole.MessageType.AcceptanceAcknowledgementNotification,
                    is_answer=True, answer_to=reply.id,
                    sender=self.aid, receiver=reply.sender,
                    value=reply.value,
                    ttl=self._current_ttl, id=str(uuid.uuid4()))
                await self.send_message(answer)
                self._adapted_flex_according_to_msgs.append(reply.id)
                self._acknowledgements_sent.append(reply.id)
                self._list_of_acknowledgements_sent.append(answer)
                return
            else:
                logger.info(f"{self.aid}: Flex is not valid. Current flex for requested time span:"
                            f" {self.flex[reply.time_span[0]][1]}."
                            f"Wanted flex by {reply.sender}: {reply.value[0]}")
        else:
            logger.info(f"{self.aid}: Acceptance from {reply.sender} invalid.")
        logger.debug(f"{self.aid}: Sending withdrawal to {reply.sender}")
        print(type(list(reply.time_span)))
        print(reply.value)
        withdrawal = WinzentMessage(time_span=list(reply.time_span),
                                    is_answer=True, answer_to=reply.id,
                                    msg_type=xboole.MessageType.WithdrawalNotification,
                                    ttl=self._current_ttl, receiver=reply.sender,
                                    value=reply.value,
                                    id=str(uuid.uuid4()),
                                    sender=self.aid
                                    )
        await self.send_message(withdrawal)

    async def handle_acceptance_acknowledgement_reply(self, reply):
        """
        Checks if an AcceptanceAcknowledgementNotification is still valid. If so, then saves it as part of
        the solution. If not, the agent sends a WithdrawalNotification.
        :param reply: The reply that is validated.
        :return:
        """
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
        if self.acknowledgement_valid(reply):
            self.save_accepted_values(reply)
        else:
            logger.debug(
                f"{self.aid} received an AcceptanceAcknowledgement (from {reply.sender} with value {reply.value}) "
                f"was not valid "
            )
            withdrawal = WinzentMessage(time_span=self._own_request.time_span,
                                        is_answer=True, answer_to=self._own_request.id,
                                        msg_type=xboole.MessageType.WithdrawalNotification,
                                        ttl=self._current_ttl, receiver=reply.sender,  # PGASC: added sender
                                        # because this message will be sent endlessly otherwise
                                        value=self._own_request.value,
                                        id=str(uuid.uuid4()),
                                        sender=self.aid
                                        )
            await self.send_message(withdrawal)
        # if the solution journal is empty afterwards, the agent does not
        # wait for any further acknowledgments and can stop the negotiation
        if self.governor.solution_journal.is_empty():
            # PGASC changed logger.info to logging
            logger.debug(f'\n*** {self.aid} received all Acknowledgements. ***')
            await self.reset()

    async def handle_withdrawal_reply(self, reply):
        """
        After receiving a WithdrawalNotification, this agent removes the other agent's messages from
        the self._acknowledgements_sent and adds that flexibility back onto his own available
        flexibility.
        :param reply: The reply containing the WithdrawalNotification
        :return:
        """
        # if the id is not saved, the agent already handled this
        # WithdrawalNotification
        if reply.answer_to in self._acknowledgements_sent:
            # Withdraw flexibility for this interval, therefore
            # it is possible to participate in a negotiation for
            # this time span
            if reply.answer_to in self._adapted_flex_according_to_msgs:
                self._acknowledgements_sent.remove(reply.answer_to)
                async with self._lock:
                    self.flex[reply.time_span[0]][1] = self.flex[reply.time_span[0]][1] + reply.value[0]
                self._adapted_flex_according_to_msgs.remove(reply.answer_to)

                # Create a new list containing the acknowledgements to keep
                new_ack_list = [ack for ack in self._list_of_acknowledgements_sent if ack.receiver != reply.sender]
                if len(self._list_of_acknowledgements_sent) != len(new_ack_list) + 1:
                    logger.error(f"{self.aid}: WARNING! Could not remove acknowledgement from list even though it "
                                 f"is in ack_sent_list."
                                 f"List of acks: {self._list_of_acknowledgements_sent}\n"
                                 f"new List: {new_ack_list}"
                                 f"Sender: {reply.sender}")
                # Replace the original list with the new list
                self._list_of_acknowledgements_sent = new_ack_list

                logger.info(
                    f"{self.aid} gets withdrawal message from consumer {reply.sender} with value "
                    f"{reply.value} "
                )
            else:
                pass
                logger.debug(
                    f"Withdrawal received: {reply.answer_to} not in {self._adapted_flex_according_to_msgs}"
                )
        elif reply.answer_to in self._curr_sent_acceptances:
            # An agent who was part of the solution withdrew its offer and will be removed from the solution.
            logger.info(
                f"{self.aid} gets withdrawal message from generator {reply.sender} with value "
                f"{reply.value}."
            )
            if reply.answer_to in self.governor.solution_journal:
                self.governor.solution_journal.remove_message(reply.answer_to)
                del self.result[reply.sender]
                self.result_sum -= reply.value[0]
                self._curr_sent_acceptances.remove(reply.answer_to)
                logger.info(
                    f"{self.aid}: Sucessfully removed offer from {reply.sender}."
                )
        else:
            logger.debug(
                f"{self.aid} received Withdrawal from {reply.sender} with answer_to {reply.answer_to} and id "
                f"{reply.id} which is not in {self._acknowledgements_sent} "
            )

    async def handle_external_reply(self, requirement, message_path=None):
        """
        Handle a reply from other agents. Reply may be from types:
        DemandNotification, OfferNotification, AcceptanceNotification,
        AcceptanceAcknowledgementNotification
        :param requirement: The requirement being handled.
        :param message_path: The message path of the message containing the requirement.
        """
        if message_path is None:
            message_path = []

        reply = requirement.message
        logger.debug(f"receiver of this reply is {reply.receiver}")
        if reply.receiver != self.aid:
            await self.send_message(reply, msg_path=message_path, forwarding=True)
            return

        if reply.msg_type == xboole.MessageType.DemandNotification \
                or reply.msg_type == xboole.MessageType.OfferNotification:
            await self.handle_initial_reply(requirement, message_path)

        elif reply.msg_type == xboole.MessageType.AcceptanceNotification:
            await self.handle_acceptance_reply(reply)

        elif reply.msg_type == \
                xboole.MessageType.AcceptanceAcknowledgementNotification:
            await self.handle_acceptance_acknowledgement_reply(reply)

        elif reply.msg_type == xboole.MessageType.WithdrawalNotification:
            await self.handle_withdrawal_reply(reply)

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
        After a negotiation, reset the negotiation parameters and the negotiation_done - Future to True.
        """
        logger.debug("the result for " + self.aid + " is " + str(self.result))
        # print("the result for " + self.aid + " is " + str(self.result))
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

    async def answer_requirements(self, final, afforded_values, initial_req):
        """
        Method to send out AcceptanceNotifications for the agents being
        part of the solution.
        The solution variable includes all the agents contributing to the solution plus their position in
        the gcd array (name and position are separated by a ":").
        The gcd includes the values that the agents in the solution variable are able to contribute.
        """
        initial_values = initial_req.forecast.second
        if isinstance(initial_values, int):
            initial_values = [initial_values]
        initial_msg_type = initial_req.message.msg_type
        # determine flexibility sign according to msg type
        positive = False if initial_msg_type == xboole.MessageType.DemandNotification else True
        self.final = {}
        for key in final.keys():
            sender = self.governor.message_journal.get_message_for_id(
                key).sender
            self.final[sender] = final[key]
        # the problem was not solved completely
        for k in afforded_values.keys():
            afforded_value = afforded_values[k]
            if positive:
                act_value = afforded_value
            else:
                act_value = -afforded_value
            if abs(afforded_values[k]) < abs(initial_values[k]):
                # problem couldn't be solved, but the timer is still running:
                # we didn't receive the flexibility from every
                # agent
                print(
                    f'*** {self.aid} has not enough flexibility. Timeout? '
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
        for k, idx_v in self.final.items():
            for idx, v in idx_v.items():
                value = v
                if not positive:
                    value = [-e for e in v]
                if v == 0:
                    zero_indeces.append(k)
                    continue
                self.final[k][idx] = value
                i += 1
                if k == self.aid:
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
            entries = self.final[k]
            time_span = []
            power_val = []
            for time, power in entries.items():
                time_span.append(initial_req.time_span[time])
                power_val.extend(power)

            msg = WinzentMessage(
                msg_type=xboole.MessageType.AcceptanceNotification,
                sender=self.aid,
                is_answer=True,
                receiver=k,
                time_span=time_span,
                value=power_val, ttl=self._current_ttl,
                id=str(uuid.uuid4()),
                answer_to=answer_to)
            self._curr_sent_acceptances.append(msg)

            # store acceptance message
            self.governor.solution_journal.add(msg)
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
        :param time_span: the time span of the original reply.
        :param receiver: The name of the agent whose reply id is searched for.
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
        logger.debug(f'\n*** {self.aid} starts solver now. ***')
        final, afforded_values, initial_req = self.governor.try_balance()
        if final:
            logger.debug(f'\n*** {self.aid} found solution. ***')
            await self.answer_requirements(final, afforded_values, initial_req)
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
        logger.info(
            f'*** {self.aid} has no solution after timeout. ***')
        self.flex[self._own_request.time_span[0]] = self.original_flex[self._own_request.time_span[0]]
        self._negotiation_running = False
        self.governor.solver_triggered = False
        self.governor.triggered_due_to_timeout = False
        self._solution_found = False
        self.governor.power_balance.clear()
        self.governor.solution_journal.clear()
        self._waiting_for_acknowledgements = False
        await self.reset()

    def handle_message(self, content, meta):
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
                asyncio.create_task(self.handle_external_reply(req,
                                                               # message_path=meta["ontology"]
                                                               ))
            else:
                req = xboole.Requirement(content,
                                         content.sender, ttl=self._current_ttl)
                asyncio.create_task(self.handle_external_request(req,
                                                                 # message_path=meta["ontology"]
                                                                 ))

    async def send_message(self, msg, receiver=None, msg_path=None, forwarding=False):
        """
        Sends the given message to all neighbors unless the receiver is given.
        """
        if msg_path is None:
            msg_path = []

        if forwarding:
            msg.ttl -= 1
            if msg.ttl <= 0:
                # PGASC add logging
                logger.debug(
                    f"handle_external_request: {self.aid} does not forward the message to other agents because ttl<=0"
                )
                # do not forward the message to other agents
                return
            logger.debug(
                f"handle_external_request: {self.aid} forward request to other agents ttl={msg.ttl}"
            )

        if self.send_message_paths:
            # for first connection establishment (demand notifications) append aid of this agent to the message path
            if not msg.is_answer:
                if self.aid in msg_path:
                    # remove old message loops
                    index_of_self = msg_path.index(self.aid)
                    msg_path = msg_path[0:index_of_self - 1]
                msg_path.append(self.aid)

            # sending over the message path, so next receiver in the neighborhood is known
            else:
                logger.debug(
                    f"{self.aid} sends {msg.msg_type} on the message path in message_path: {msg_path}")
                if len(msg_path) == 0:
                    logger.error("message_path has length zero")
                    return
                index_of_next_on_path = msg_path.index(self.aid) + 1
                receiver = msg_path[index_of_next_on_path]
                if receiver not in self.neighbors.keys():
                    logger.error(
                        f"message_path at {self.aid} with message_path {msg_path} failed because "
                        f"receiver {receiver} not in {self.neighbors.keys()}")
                    return

        if receiver is not None and receiver in self.neighbors.keys():
            # receiver is a neighbor
            message = copy_winzent_message(msg)
            self.messages_sent += 1
            await super().send_message(
                content=message, receiver_addr=self.neighbors[receiver],
                receiver_id=receiver,
                acl_metadata={'sender_addr': super().addr,
                              'sender_id': self.aid,
                              'ontology': msg_path.copy()}, create_acl=True)
        else:
            # send message to every neighbor
            for neighbor in self.neighbors.keys():
                message = copy_winzent_message(msg)
                if message.sender == neighbor:
                    continue
                if message.receiver is None:
                    message.receiver = ''
                self.messages_sent += 1
                await super().send_message(
                    content=message, receiver_addr=self.neighbors[neighbor],
                    receiver_id=neighbor,
                    acl_metadata={'sender_addr': super().addr,
                                  'sender_id': self.aid,
                                  'ontology': msg_path.copy()},
                    # copy to avoid neighbors working on the same object
                    create_acl=True
                )


def copy_winzent_message(message: WinzentMessage) -> WinzentMessage:
    """
    This method creates a deep copy of a Winzent message to avoid manipulations
    when the message is sent to multiple agents.

    param message: a Winzent message to be copied
    return: the copied Winzent message
    """
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
        ethics_score=message.ethics_score
    )


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False