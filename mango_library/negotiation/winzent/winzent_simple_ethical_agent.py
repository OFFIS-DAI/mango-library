import asyncio
import logging
import uuid
from abc import ABC
from copy import deepcopy

from mango_library.negotiation.winzent import xboole
from mango_library.negotiation.winzent.winzent_message_pb2 import WinzentMessage
from mango_library.negotiation.winzent.winzent_base_agent import WinzentBaseAgent

logger = logging.getLogger(__name__)


class WinzentSimpleEthicalAgent(WinzentBaseAgent, ABC):
    def __init__(self, container, ttl, time_to_sleep=3, send_message_paths=False, ethics_score=1,
                 use_producer_ethics_score=True,
                 use_consumer_ethics_score=True):
        super().__init__(container, ttl, time_to_sleep, send_message_paths, ethics_score)

        self.current_time_span = 0
        # store flexibility as interval with maximum and minimum value per time
        self.offer_list = []
        self.first_offer_received = False
        self.first_demand_received = False
        self.use_producer_ethics_score = use_producer_ethics_score
        self.use_consumer_ethics_score = use_consumer_ethics_score
        # override base agent power balance strategy
        self.governor.power_balance_strategy = \
            xboole.XbooleEthicalPowerBalanceSolverStrategy()
        self.lock = asyncio.Lock()

    def update_flexibility(self, t_start, min_p, max_p):
        super().update_flexibility(t_start, min_p, max_p)
        self.current_time_span = t_start
        self.first_demand_received = False

    async def handle_forwarding(self, reply, message_path):
        await self.forward_message(reply, message_path)
        # Don't forward the message if it is already an answer to one of this agent's previous messages.
        if reply.is_answer:
            return

    async def handle_forwarding_request(self, value, message, message_path, request_completed):
        await self.forward_message(message, message_path=None)

    async def answer_external_request(self, message, message_path, value):
        if self.use_consumer_ethics_score:
            await self.forward_message(message, message_path)
            msg_type = xboole.MessageType.Null
            # send message reply
            if message.msg_type == xboole.MessageType.OfferNotification:
                msg_type = xboole.MessageType.DemandNotification
            elif message.msg_type == xboole.MessageType. \
                    DemandNotification:
                msg_type = xboole.MessageType.OfferNotification
            self.governor.message_journal.add(message)
            self.offer_list.append(message)
            if not self.first_demand_received:
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
                    # print(f"{self.aid} sends offer to {reply.receiver}. Offer list is {len(offers)}")
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
                        break
                self.first_demand_received = False
            else:
                return
        else:
            await super().answer_external_request(self, message, message_path)

    async def handle_demand_or_offer_reply(self, requirement, message_path):
        if self.use_producer_ethics_score:
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
                if not self.first_offer_received:
                    self.first_offer_received = True
                    await asyncio.sleep(0.5)
                    print(
                        f"{self.aid}: slept enough. power ledger len is {len(self.governor.power_balance._ledger[self.current_time_span])}")
                    await self.solve()
                    self.first_offer_received = False
        else:
            await super().handle_demand_or_offer_reply(requirement, message_path)

    async def reset(self):
        """
        After a negotiation, reset the negotiation parameters.
        """
        flex = self.get_flexibility_for_interval(self.current_time_span)
        if flex > 0:
            self.result[self.aid] = flex
            self.result_sum += flex
        self.first_offer_received = False
        self.first_demand_received = False
        await super().reset()

    def calc_result_sum(self):
        self.result_sum = 0
        for res in self.result.values():
            self.result_sum += res
