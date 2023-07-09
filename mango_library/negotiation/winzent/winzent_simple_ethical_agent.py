import asyncio
import logging
import math
import uuid
from abc import ABC
from copy import deepcopy
from datetime import datetime
from time import sleep

from mango.core.agent import Agent

from mango_library.negotiation.winzent import xboole
from mango_library.negotiation.winzent.winzent_message_pb2 import WinzentMessage
from negotiation.winzent.winzent_base_agent import WinzentBaseAgent

logger = logging.getLogger(__name__)


class WinzentSimpleEthicalAgent(WinzentBaseAgent):
    def __init__(self, container, ttl, time_to_sleep=3, send_message_paths=False, ethics_score=1):
        super().__init__(container, ttl, time_to_sleep, send_message_paths, ethics_score)

        self.current_time_span = 0
        self.stored_offers_and_demands = {}
        # store flexibility as interval with maximum and minimum value per time
        self.offer_list = []
        self.first_offer_received = False
        self.first_demand_received = False
        # override base agent power balance strategy
        self.governor.power_balance_strategy = \
            xboole.XbooleEthicalPowerBalanceSolverStrategy()

    def update_flexibility(self, t_start, min_p, max_p):
        super().update_flexibility(t_start, min_p, max_p)
        self.current_time_span = t_start
        self.first_demand_received = False

    async def handle_forwarding(self, reply, message_path):
        await self.forward_message(reply, message_path)
        if reply.is_answer:
            return

    async def handle_forwarding_request(self, value, message, message_path, request_completed):
        await self.forward_message(message, message_path=None)

    async def answer_external_request(self, message, message_path, value):
        msg_type = xboole.MessageType.Null
        # send message reply
        if message.msg_type == xboole.MessageType.OfferNotification:
            msg_type = xboole.MessageType.DemandNotification
        elif message.msg_type == xboole.MessageType. \
                DemandNotification:
            msg_type = xboole.MessageType.OfferNotification
        self.governor.message_journal.add(message)
        self.stored_offers_and_demands[message.sender] = message
        self.offer_list.append(message)
        if not self.first_demand_received:
            # print(self.aid + " received its first demand.")
            self.first_demand_received = True
            await asyncio.sleep(0.5)
            offers = deepcopy(self.offer_list)
            self.offer_list.clear()
            offers.sort(key=self.get_ethics_score, reverse=True)
            for offer in offers:
                if offer.value[0] >= value:
                    value_to_offer = value
                    value = 0
                else:
                    value = value - offer.value[0]
                    value_to_offer = offer.value[0]
                #print(self.aid + " sending offer to " + offer.sender)
                reply = WinzentMessage(msg_type=msg_type,
                                       sender=self._aid,
                                       is_answer=True,
                                       receiver=offer.sender,
                                       time_span=offer.time_span,
                                       value=[value_to_offer], ttl=self._current_ttl,
                                       id=str(uuid.uuid4()),
                                       ethics_score=self.ethics_score)
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
                    logger.debug(
                        f"{self.aid} sends Reply to Request to {reply.receiver} on path: {message_path}")
                    await self.send_message(reply, message_path=message_path)
                else:
                    await self.send_message(reply)
                    # print(f"{self.aid} sends message to {offer.sender}")
                if value <= 0:
                    # self.first_offer_received = False
                    break
            self.first_demand_received = False
        else:
            return

    async def flexibility_valid(self, reply):
        #TODO: vereinheitlichen
        """
        Checks whether the requested flexibility value in reply is valid (less than or equal to the stored
        flexibility value for the given interval).
        """
        message_type = self.stored_offers_and_demands[reply.sender].msg_type

        if message_type == xboole.MessageType.OfferNotification:
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
            self.stored_offers_and_demands[requirement.message.sender] = requirement.message
            if not self.governor.solver_triggered:
                self.governor.triggered_due_to_timeout = False
            # Save the established connection
            if self.send_message_paths:
                message_path.reverse()
                self.negotiation_connections[
                    message_path[-1]] = message_path  # received offer; establish connection
                # supplier:[self.aid/demander, ..., supplier]
            if not self.first_offer_received:
                self.first_offer_received = True
                await asyncio.sleep(0.5)
                print(
                    f"slept enough. power ledger len is {len(self.governor.power_balance._ledger[self.current_time_span])}")
                await self.solve()
                self.first_offer_received = False

    async def handle_acceptance_reply(self, reply):
        # First, check whether the AcceptanceNotification is still valid
        if self.acceptance_valid(reply):
            # Send an AcceptanceAcknowledgementNotification for the
            # acceptance
            # if await self.flexibility_valid(reply):
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
            current_flex = self.flex[self.current_time_span][1]
            self.flex[self.current_time_span] = [0, current_flex - reply.value[0]]
            del self.stored_offers_and_demands[reply.sender]
        return

    async def reset(self):
        """
        After a negotiation, reset the negotiation parameters.
        """
        flex = self.get_flexibility_for_interval(self.current_time_span)
        if flex > 0:
            self.result[self.aid] = flex
            self.result_sum += flex
        print("the result for " + self.aid + " is " + str(self.result))
        self.first_offer_received = False
        self.first_demand_received = False
        await super().reset()

    def acceptance_valid(self, msg):
        #TODO: vereinheitlichen
        """
        Returns whether the message is still valid by checking whether it is
        in the current inquiries the agent received from others
        """
        offer_message = self.stored_offers_and_demands[msg.sender]
        if offer_message is not None:
            if offer_message.value[0] >= msg.value[0]:
                return True
        return False

    def calc_result_sum(self):
        self.result_sum = 0
        for res in self.result.values():
            self.result_sum += res
