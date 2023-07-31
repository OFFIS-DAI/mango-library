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
                 request_processing_waiting_time=0.4,
                 reply_processing_waiting_time=0.4,
                 use_producer_ethics_score=True,
                 use_consumer_ethics_score=True):
        super().__init__(container, ttl, time_to_sleep, send_message_paths, ethics_score)
        print(time_to_sleep)
        self.current_time_span = 0
        # store flexibility as interval with maximum and minimum value per time
        self.offer_list = []
        self.first_offer_received = False
        self.first_demand_received = False
        self.use_producer_ethics_score = use_producer_ethics_score
        self.use_consumer_ethics_score = use_consumer_ethics_score
        print(f"{self.use_producer_ethics_score} and con: {self.use_consumer_ethics_score}")
        self.request_processing_waiting_time = request_processing_waiting_time
        self.reply_processing_waiting_time = reply_processing_waiting_time
        # override base agent power balance strategy
        self.governor.power_balance_strategy = \
            xboole.XbooleEthicalPowerBalanceSolverStrategy()
        self.lock = asyncio.Lock()

    def update_flexibility(self, t_start, min_p, max_p):
        super().update_flexibility(t_start, min_p, max_p)
        self.current_time_span = t_start
        self.first_demand_received = False

    async def answer_external_request(self, message, message_path, value):
        if self.use_consumer_ethics_score:
            await self.send_message(message, msg_path=message_path, forwarding=True)
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
                print("first demand received")
                await asyncio.sleep(self.request_processing_waiting_time)
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
                    reply = WinzentMessage(msg_type=msg_type,
                                           sender=self._aid,
                                           is_answer=True,
                                           receiver=offer.sender,
                                           time_span=offer.time_span,
                                           value=[value_to_offer], ttl=self._current_ttl,
                                           id=str(uuid.uuid4()),
                                           ethics_score=self.ethics_score)
                    self._current_inquiries_from_agents[reply.id] = reply
                    await self.send_message(reply)
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
                if not self.first_offer_received:
                    self.first_offer_received = True
                    await asyncio.sleep(self.reply_processing_waiting_time )
                    # print( f"{self.aid}: slept enough. power ledger len is {len(
                    # self.governor.power_balance._ledger[self.current_time_span])}")
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
