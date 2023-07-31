import asyncio
import logging
import math
import uuid
from abc import ABC
from datetime import datetime

from mango.core.agent import Agent

from mango_library.negotiation.winzent import xboole
from mango_library.negotiation.winzent.winzent_message_pb2 import WinzentMessage

logger = logging.getLogger(__name__)


class WinzentBaseAgent(Agent, ABC):
    def __init__(self, container, ttl, time_to_sleep=3, send_message_paths=False, ethics_score=1.0):
        super().__init__(container)

        # PGASC: if true stores the message path in the message
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
        """
        self.flex[t_start] = [min_p, max_p]
        self.original_flex[t_start] = [min_p, max_p]
        self._list_of_acknowledgements_sent.clear()
        self._current_inquiries_from_agents.clear()

    async def start_negotiation(self, ts, value):
        """
        Start a negotiation with other agents for the given timestamp and
        value. The negotiation is started by calling handle_internal_request.
        :param ts: timespan for the negotiation
        :param value: power value to negotiate about
        """
        if not isinstance(value, int):
            value = value[1]
        self._solution_found = False
        requirement = xboole.Requirement(
            xboole.Forecast((ts, math.ceil(value))), ttl=self._current_ttl)
        requirement.from_target = True
        requirement.message.sender = self._aid
        message = requirement.message
        message.sender = self._aid
        self.governor.message_journal.add(message)
        self.governor.curr_requirement_value = value
        self.governor.solution_journal.clear()
        self.negotiation_done = asyncio.Future()
        logger.debug(f"{self.aid} starts negotiation, needs {value}")
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
                now = datetime.now()

                current_time = now.strftime("%H:%M:%S")
                logger.debug(f"{self.aid}: Timer ran out at =", current_time)
                # After sleeping, the solver is triggered. This is necessary
                # in case when not the complete negotiation problem can be
                # solved. The solver is triggered after the timeout to
                # determine the solution according to the power that
                # is available.
                print("timer timed out")
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
        The negotiation request is for this agents. Therefore, it handles an
        internal request and not a request from other agents. This is the
        beginning of a negotiation, because messages to the neighboring agents
        are sent regarding the negotiation information in the given
        requirement.
        """
        message = requirement.message
        value = self.get_flexibility_for_interval(t_start=message.time_span[0], msg_type=message.msg_type)

        if abs(message.value[0]) - abs(value) <= 0:
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

    def get_flexibility_for_interval(self, t_start, msg_type=6):
        """
        Returns the flexibility for the given time interval according
        to the msg type.
        """
        if t_start in self.flex.keys():
            flexibility = self.flex[t_start]
            if msg_type == xboole.MessageType.OfferNotification:
                # in this case, the upper part of the flexibility interval
                # is considered
                return flexibility[1]
            elif msg_type == xboole.MessageType.DemandNotification:
                # in this case, the lower part of the flexibility interval
                # is considered
                return flexibility[0]
        else:
            return 0

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

    async def answer_external_request(self, message, message_path, value):
        msg_type = xboole.MessageType.Null
        # send message reply
        if message.msg_type == xboole.MessageType.OfferNotification:
            msg_type = xboole.MessageType.DemandNotification
        elif message.msg_type == xboole.MessageType.DemandNotification:
            msg_type = xboole.MessageType.OfferNotification

        reply = WinzentMessage(
            msg_type=msg_type,
            sender=self._aid,
            is_answer=True,
            receiver=message.sender,
            time_span=message.time_span,
            value=[value],
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
        The agent received a negotiation request from another agent.
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
        try:
            value = self.get_flexibility_for_interval(
                t_start=message.time_span[0],
                msg_type=message.msg_type
            )
        except:
            print(f" ERROR! {message}")
            value = 0
        if value != 0:
            await self.answer_external_request(message, message_path, value)
        else:
            # if self.aid == "agent14":
            #    print(f"current flex is {self.flex}")
            await self.send_message(message, msg_path=message_path, forwarding=True)

    def get_ethics_score(self, message):
        return message.ethics_score

    async def check_flex(self, reply):
        distributed_value = 0
        for ack in self._list_of_acknowledgements_sent:
            distributed_value += ack.value[0]
            logger.info(f"{self.aid} promised {ack.value[0]} to {ack.receiver}")
        if self.original_flex[reply.time_span[0]][1] - distributed_value == self.flex[reply.time_span[0]][1]:
            return True
        else:
            for ack in self._list_of_acknowledgements_sent:
                distributed_value += ack.value[0]
                logger.info(f"{self.aid} promised {ack.value[0]} to {ack.receiver}")
            logger.info(
                f"{self.aid}: Current flex is not consistent with the values already distributed."
                f"Distributed value is {distributed_value} and original flex is "
                f"{self.original_flex[reply.time_span[0]][1]}."
                f"Current flex is {self.flex[reply.time_span[0]][1]}")
            logger.info(f"Attempting to fix flex...")
            self.flex[reply.time_span[0]][1] = self.original_flex[reply.time_span[0]][1] - distributed_value
            if self.flex[reply.time_span[0]][1] < reply.value[0]:
                logger.info(f"{self.aid}: Acknowledgement to {reply.sender} cannot be sent, "
                            f"flex is {self.flex[reply.time_span[0]][1]}.")
                return False
            logger.info("Flex has been fixed and Acknowledgement can be sent.")
        return True

    async def flexibility_valid(self, reply):
        """
        Checks whether the requested flexibility value in reply is valid (less than or equal to the stored
        flexibility value for the given interval).
        """
        valid = abs(self.flex[reply.time_span[0]][1]) >= abs(reply.value[0]) and await self.check_flex(reply)
        if valid:
            self.flex[reply.time_span[0]][1] = self.flex[reply.time_span[0]][1] - reply.value[0]
        return valid

    async def handle_demand_or_offer_reply(self, requirement, message_path):
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
                    sender=self._aid, receiver=reply.sender,
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
        withdrawal = WinzentMessage(time_span=self._own_request.time_span,
                                    is_answer=True, answer_to=reply.id,
                                    msg_type=xboole.MessageType.WithdrawalNotification,
                                    ttl=self._current_ttl, receiver=reply.sender,
                                    value=reply.value[0],
                                    id=str(uuid.uuid4()),
                                    sender=self._aid
                                    )
        await self.send_message(withdrawal)

    async def handle_acceptance_acknowledgement_reply(self, reply):
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

    async def handle_withdrawal_reply(self, reply):
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
        """
        if message_path is None:
            message_path = []

        reply = requirement.message
        logger.debug(f"receiver of this reply is {reply.receiver}")
        if reply.receiver != self._aid:
            await self.send_message(reply, msg_path=message_path, forwarding=True)
            return

        if reply.msg_type == xboole.MessageType.DemandNotification \
                or reply.msg_type == xboole.MessageType.OfferNotification:
            await self.handle_demand_or_offer_reply(requirement, message_path)

        elif reply.msg_type == xboole.MessageType.AcceptanceNotification:
            await self.handle_acceptance_reply(reply)

        elif reply.msg_type == \
                xboole.MessageType.AcceptanceAcknowledgementNotification:
            await self.handle_acceptance_acknowledgement_reply(reply)

        elif reply.msg_type == xboole.MessageType.WithdrawalNotification:
            await self.handle_withdrawal_reply(reply)

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
        logger.info("the result for " + self.aid + " is " + str(self.result))
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
        The solution variable includes all the agents contributing to the solution plus their position in
        the gcd array (name and position are separated by a ":").
        The gcd includes the values that the agents in the solution variable are able to contribute.
        """
        initial_value = initial_req.forecast.second
        answer_objects = [solution[j].split(':', 1)[0] for j in range(len(solution))]
        positive = False if initial_req.message.msg_type == xboole.MessageType.DemandNotification else True
        sol = DictList()
        for j in range(len(answer_objects)):
            sol.add((answer_objects[j], gcd[int(solution[j][-1])]))

        self.final = {}
        afforded_value = 0
        for k, v in sol.values.copy().items():
            diff = min(abs(initial_value) - abs(afforded_value), abs(v))
            self.final[self.governor.message_journal.get_message_for_id(k).sender] = diff
            afforded_value += diff

        act_value = afforded_value if positive else -afforded_value

        # the problem was not solved completely
        if abs(afforded_value) < abs(initial_value):
            print("afforded value " + str(abs(afforded_value)) + " and inital value " + str(abs(initial_value)))
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
                        print("option 1")
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
                        print("option 2")
                        await self.no_solution_after_timeout()

                else:
                    continue
            # create AcceptanceNotification
            reply = WinzentMessage(
                msg_type=xboole.MessageType.AcceptanceNotification,
                sender=self._aid,
                is_answer=True,
                receiver=k,
                time_span=initial_req.forecast.first,
                value=[self.final[k]], ttl=self._current_ttl,
                id=str(uuid.uuid4()),
                answer_to=answer_to)
            self._curr_sent_acceptances.append(reply)

            # store acceptance message
            self.governor.solution_journal.add(reply)
            if self.send_message_paths:
                logger.debug(f"receiver {reply.receiver} in connections {self.negotiation_connections}?")
                await self.send_message(reply, receiver=reply.receiver,
                                        msg_path=self.negotiation_connections[reply.receiver])
            else:
                await self.send_message(reply, receiver=reply.receiver)
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
        logger.debug(f'\n*** {self._aid} starts solver now. ***')
        result = self.governor.try_balance()
        if result is None:
            self.governor.solver_triggered = False
            if self.governor.triggered_due_to_timeout:
                # solver was triggered after the timeout and yet there was
                # still no solution
                self.governor.triggered_due_to_timeout = False
                print("option 3")
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
                await self.answer_requirements(answers, gcd_p, result[2])
                return

        if self.governor.triggered_due_to_timeout:
            self.governor.triggered_due_to_timeout = False
            print("option 4")
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
            f'*** {self._aid} has no solution after timeout. ***')
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
                asyncio.create_task(self.handle_external_reply(req, message_path=meta["ontology"]))
            else:
                req = xboole.Requirement(content,
                                         content.sender, ttl=self._current_ttl)
                asyncio.create_task(self.handle_external_request(req, message_path=meta["ontology"]))

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
            await self._container.send_message(
                content=message, receiver_addr=self.neighbors[receiver],
                receiver_id=receiver,
                acl_metadata={'sender_addr': self._container.addr,
                              'sender_id': self._aid,
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
                await self._container.send_message(
                    content=message, receiver_addr=self.neighbors[neighbor],
                    receiver_id=neighbor,
                    acl_metadata={'sender_addr': self._container.addr,
                                  'sender_id': self._aid,
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
