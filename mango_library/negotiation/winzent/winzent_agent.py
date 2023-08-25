import asyncio
import math
import uuid
from copy import deepcopy

from mango import Agent

from mango_library.negotiation.winzent import xboole
from mango_library.negotiation.winzent.winzent_message_pb2 import WinzentMessage


class WinzentAgent(Agent):
    def __init__(self, container, ttl, time_to_sleep=2):
        super().__init__(container)

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
        self._time_to_sleep = 10# time_to_sleep  # time to sleep between regular tasks
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

    def update_flexibility(self, t_start, min_p, max_p):
        """
        Update the own flexibility. Flexibility is a range from power from
        min_p to max_p for a given time interval beginning with t_start.
        """
        self.flex[t_start] = [min_p, max_p]

    async def start_negotiation(self, ts, values):
        """
        Start a negotiation with other agents for the given timestamp and
        value. The negotiation is started by calling handle_internal_request.
        :param ts: timespan for the negotiation
        :param value: power value to negotiate about
        """
        values = [math.ceil(value) for value in values]
        requirement = xboole.Requirement(
            xboole.Forecast((ts, values)), ttl=self._current_ttl)
        requirement.from_target = True
        requirement.message.sender = self.aid
        message = requirement.message
        message.sender = self.aid
        self.governor.message_journal.add(message)
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
                print(f'*** {self.aid} did not receive all acknowledgements. Negotiation was not successful.')
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
                    await self.send_msg(withdrawal, receiver=acc_msg.receiver)

                self._unsuccessful_negotiations.append([self._own_request.time_span, self._own_request.value])
                for idx in range(len(self._own_request.time_span)):
                    self.flex[self._own_request.time_span[idx]] = self.original_flex[self._own_request.time_span[idx]]

                await asyncio.sleep(self._time_to_sleep)
                await self.reset()

                self._curr_sent_acceptances = []

    async def handle_internal_request(self, requirement):
        """
        The negotiation request is for this agent. Therefore, it handles an
        internal request and not a request from other agents. This is the
        beginning of a negotiation, because messages to the neighboring agents
        are sent regarding the negotiation information in the given
        requirement.
        """
        message = requirement.message
        self.original_flex = deepcopy(self.flex)
        values = self.get_flexibility_for_interval(time_span=message.time_span, msg_type=message.msg_type)

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
        for idx in range(len(message.time_span) - 1):
            message.value[idx] = message.value[idx] - values[0]
        requirement.message = message
        requirement.forecast.second = message.value
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
                                 sender=self.aid,
                                 )
        self._own_request = requirement.message
        self._negotiation_running = True
        await self.send_msg(neg_msg)

    def get_flexibility_for_interval(self, time_span, msg_type):
        """
        Returns the flexibility for the given time interval according
        to the msg type.
        """
        flex = []
        for idx in range(len(time_span)):
            t_start = time_span[idx]
            flexibility = self.flex[t_start]
            if msg_type == xboole.MessageType.OfferNotification:
                # in this case, the upper part of the flexibility interval
                # is considered
                flex.append(flexibility[1])
            elif msg_type == xboole.MessageType.DemandNotification:
                # in this case, the lower part of the flexibility interval
                # is considered
                flex.append(flexibility[0])
        return flex

    def exists_flexibility(self, ts):
        """
        Returns whether there exists flexibility for the given time interval.
        """
        ts = deepcopy(ts)
        return all(t_start in self.flex.keys() for t_start in ts)

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

    async def handle_external_request(self, requirement):
        """
        The agent received a negotiation request from another agent.
        """
        message = requirement.message
        # There is already a negotiation running
        if self._negotiation_running and self._own_request.time_span == message.time_span:
            # first, check whether the value fulfills the own request
            if self.should_withdraw(message):
                withdrawal = WinzentMessage(time_span=self._own_request.time_span,
                                            is_answer=True, answer_to=self._own_request.id,
                                            msg_type=xboole.MessageType.WithdrawalNotification,
                                            ttl=self._current_ttl, receiver='',
                                            value=self._own_request.value,
                                            id=str(uuid.uuid4()),
                                            sender=self.aid
                                            )
                await self.send_msg(withdrawal)
                await self.reset()
            else:
                # no withdrawal, still keep the negotiation running
                return
        values = []
        if self.exists_flexibility(
                message.time_span):
            # If the agent has flexibility for the requested time, it replies
            # to the requesting agent
            values = self.get_flexibility_for_interval(
                time_span=message.time_span,
                msg_type=message.msg_type)
            if not all(element == 0 for element in values):
                msg_type = xboole.MessageType.Null
                # send message reply
                if message.msg_type == xboole.MessageType.OfferNotification:
                    msg_type = xboole.MessageType.DemandNotification
                elif message.msg_type == xboole.MessageType. \
                        DemandNotification:
                    msg_type = xboole.MessageType.OfferNotification
                reply = WinzentMessage(msg_type=msg_type,
                                       sender=self.aid,
                                       is_answer=True,
                                       receiver=message.sender,
                                       time_span=message.time_span,
                                       value=values, ttl=self._current_ttl,
                                       id=str(uuid.uuid4()))
                self.governor.message_journal.add(reply)
                self._current_inquiries_from_agents[reply.id] = reply
                await self.send_msg(reply)
        message.ttl -= 1
        if message.ttl <= 0:
            # do not forward the message to other agents
            return
        a = self.forward_msg(message.value, values)
        if not self.forward_msg(message.value, values):
            # The value in the negotiation request is completely fulfilled.
            # Therefore, the message is not forwarded to other agents.
            return
        # In this case, the power value of the request cannot be completely
        # fulfilled yet. Therefore, the remaining power of the request is
        # forwarded to other agents.
        val = deepcopy(message.value)
        del message.value[:]
        for idx in range(len(values)):
            message.value.append(val[idx] - values[idx])
        message.is_answer = False
        message.receiver = ''
        await self.send_msg(message)

    def forward_msg(self, msg_values, values):
        return any(abs(x) - abs(y) != 0 for x, y in zip(msg_values, values))

    async def flexibility_valid(self, reply):
        """
        Checks whether the requested flexibility value in reply is valid (less than or equal to the stored
        flexibility value for the given interval).
        """
        message_type = self._current_inquiries_from_agents[reply.answer_to].msg_type

        for idx, value in enumerate(reply.value):
            if message_type == xboole.MessageType.DemandNotification:
                valid = abs(self.flex[reply.time_span[idx]][1]) >= abs(value)
                if valid:
                    self.original_flex = deepcopy(self.flex)
                    self.flex[reply.time_span[idx]][1] = self.flex[reply.time_span[idx]][1] - value
            else:
                valid = abs(self.flex[reply.time_span[idx]][0]) >= abs(value)
                if valid:
                    self.original_flex = deepcopy(self.flex)
                    self.flex[reply.time_span[idx]][0] = self.flex[reply.time_span[idx]][0] - value
        return valid

    async def handle_external_reply(self, requirement):
        """
        Handle a reply from other agents. Reply may be from types:
        DemandNotification, OfferNotification, AcceptanceNotification,
        AcceptanceAcknowledgementNotification
        """
        reply = requirement.message
        if reply.receiver != self.aid:
            # The agent is not the receiver of the reply, therefore it needs
            # to forward it if the time to live is above 0.
            reply.ttl = reply.ttl - 1
            if reply.ttl > 0:
                await self.send_msg(reply)
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
                await self.solve()
        elif reply.msg_type == xboole.MessageType.AcceptanceNotification:
            # First, check whether the AcceptanceNotification is still valid
            print('check valid')
            if self.acknowledgement_valid(reply):
                print('ack valid')
                # Send an AcceptanceAcknowledgementNotification for the
                # acceptance
                if await self.flexibility_valid(reply):
                    print('flex also valid')
                    answer = WinzentMessage(
                        msg_type=xboole.MessageType.AcceptanceAcknowledgementNotification,
                        is_answer=True, answer_to=reply.id,
                        sender=self.aid, receiver=reply.sender,
                        ttl=self._current_ttl, id=str(uuid.uuid4()))
                    await self.send_msg(answer)
                    self._adapted_flex_according_to_msgs.append(reply.id)
                    self._acknowledgements_sent.append(reply.id)
                    del self._current_inquiries_from_agents[reply.answer_to]
            return
        elif reply.msg_type == \
                xboole.MessageType.AcceptanceAcknowledgementNotification:
            # If there is no message in solution journal or
            # the solution journal does not contain this message, it is
            # irrelevant
            if self.governor.solution_journal.is_empty() or not self.governor.solution_journal.contains_message(
                    reply.answer_to) or not self._waiting_for_acknowledgements:
                return
            # Remove the Acknowledgement from the solution journal
            self.governor.solution_journal.remove_message(reply.answer_to)
            # if the solution journal is empty now, the agent does not
            # wait for any further acknowledgments and can stop the negotiation
            if self.governor.solution_journal.is_empty():
                print(f'\n*** {self.aid} received all Acknowledgements. ***')
                await self.reset()

        elif reply.msg_type == xboole.MessageType.WithdrawalNotification:
            # if the id is not saved, the agent already handled this
            # WithdrawalNotification
            if reply.id in self._acknowledgements_sent:
                # Withdraw flexibility for this interval, therefore
                # it is possible to participate in a negotiation for
                # this time span
                if reply.answer_to in self._adapted_flex_according_to_msgs:
                    del self._acknowledgements_sent[reply.id]
                    for t_start in reply.time_span:
                        self.flex[t_start] = self.original_flex[t_start]
                    self._adapted_flex_according_to_msgs.remove(reply.answer_to)

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

        self._acknowledgements_sent = []

    def acknowledgement_valid(self, msg):
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
        initial_values = initial_req.forecast.second
        initial_msg_type = initial_req.message.msg_type
        # determine flexibility sign according to msg type
        positive = False if initial_msg_type == xboole.MessageType.DemandNotification else True
        self.final = {}

        for key, value in gcd.items():
            initial_value = initial_values[key]
            afforded_value = 0
            answer_objects = []
            # create dict with agents and used power value per agent
            for j in range(len(solution[key])):
                answer_objects.append(solution[key][j].split(':', 1)[0])
            sol = DictList()
            # from the solution, store the actual values for each agent in a dict (sol)
            for j in range(len(answer_objects)):
                sol.add((answer_objects[j], gcd[key][int(solution[key][j][-1])]))

            for k, v in sol.values.copy().items():
                if abs(afforded_value) + abs(v) >= abs(initial_value):
                    diff = abs(initial_value) - abs(afforded_value)
                    sender = self.governor.message_journal.get_message_for_id(
                        k).sender
                    if sender not in self.final.keys():
                        self.final[sender] = {}
                    if key not in self.final[sender].keys():
                        self.final[sender][key] = []
                    self.final[sender][key].append(diff)
                    afforded_value += diff
                    break
                sender = self.governor.message_journal.get_message_for_id(
                    k).sender
                if sender not in self.final.keys():
                    self.final[sender] = {}
                if key not in self.final[sender].keys():
                    self.final[sender][key] = []
                self.final[sender][key].append(v)
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
                await self.send_msg(msg)
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
        print(f'\n*** {self.aid} starts solver now. ***')

        result = await self.governor.try_balance()

        if result is None:
            self.governor.solver_triggered = False
            if self.governor.triggered_due_to_timeout:
                # solver was triggered after the timeout and yet there was
                # still no solution
                self.governor.triggered_due_to_timeout = False
                await self.no_solution_after_timeout()
            return
        if isinstance(result, list):
            answers = {}
            gcd_p = {}
            for idx, entry in enumerate(result):
                solution = entry[0]
                if len(solution) > 0:
                    # There was actually a solution. Split solution values according
                    # to agents taking part in it
                    for k, v in solution.vv.items():
                        if v[0] == xboole.Tval(1):
                            if idx not in answers.keys():
                                answers[idx] = []
                            answers[idx].append(k)
                    gcd_p[idx] = entry[1]
            if len(answers) >= len(result):
                print(f'\n*** {self.aid} found solution. ***')
                await self.answer_requirements(answers, gcd_p, result[0][2])
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
        print(
            f'*** {self.aid} has no solution after timeout. ***')
        self._unsuccessful_negotiations.append([self._own_request.time_span, self._own_request.value])
        for idx in range(len(self._own_request.time_span)):
            self.flex[self._own_request.time_span[idx]] = self.original_flex[self._own_request.time_span[idx]]
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
                asyncio.create_task(self.handle_external_reply(req))
                return
            req = xboole.Requirement(content,
                                     content.sender, ttl=self._current_ttl)
            asyncio.create_task(self.handle_external_request(req))

    async def send_msg(self, message, receiver=None):
        """
        Sends the given message to all neighbors unless the receiver is given.
        """
        print(
            f'*** {self.aid} sends message with type {message.msg_type}. Is answer: {message.is_answer} ***')

        if receiver is not None:
            if receiver in self.neighbors.keys():
                await self._context.send_message(
                    content=message, receiver_addr=self.neighbors[receiver],
                    receiver_id=receiver,
                    acl_metadata={'sender_addr': self._context.addr,
                                  'sender_id': self.aid}, create_acl=True)
        else:
            for neighbor in self.neighbors.keys():
                if message.sender == neighbor:
                    continue
                if message.receiver is None:
                    message.receiver = ''
                if message.sender == neighbor:
                    continue
                await self._context.send_message(
                    content=message, receiver_addr=self.neighbors[neighbor],
                    receiver_id=neighbor,
                    acl_metadata={'sender_addr': self._context.addr,
                                  'sender_id': self.aid},
                    create_acl=True
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
