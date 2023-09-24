import asyncio
import logging
import uuid
from abc import ABC
from copy import deepcopy

from mango_library.negotiation.winzent import xboole
from mango_library.negotiation.winzent.winzent_message_pb2 import WinzentMessage
from mango_library.negotiation.winzent.winzent_base_agent import WinzentBaseAgent

logger = logging.getLogger(__name__)


class WinzentEthicalAgent(WinzentBaseAgent, ABC):
    def __init__(self, container, ttl, time_to_sleep=3, send_message_paths=False, ethics_score=1,
                 request_processing_waiting_time=0.2,
                 reply_processing_waiting_time=0.4,
                 use_producer_ethics_score=True,
                 use_consumer_ethics_score=True,
                 min_coverage=0.9,
                 coverage_weight=0.4,
                 ethics_score_weight=0.6,
                 elem_type=None,
                 index=-1,
                 ):
        super().__init__(container, ttl, time_to_sleep, send_message_paths, ethics_score, elem_type, index)
        self.current_time_spans = []
        self.offer_list = []
        self.first_offer_received = False
        self.first_demand_received = False
        self.use_producer_ethics_score = use_producer_ethics_score
        self.use_consumer_ethics_score = use_consumer_ethics_score
        self.request_processing_waiting_time = request_processing_waiting_time
        self.reply_processing_waiting_time = reply_processing_waiting_time
        # override base agent power balance strategy
        self.governor.power_balance_strategy = \
            xboole.XbooleEthicalPowerBalanceSolverStrategy(min_coverage=min_coverage,
                                                           coverage_weight=coverage_weight,
                                                           ethics_score_weight=ethics_score_weight
                                                           )
        self.lock = asyncio.Lock()

    def update_flexibility(self, t_start, min_p, max_p):
        """
        This method calls the super method of the Winzent base agent to update its flexibility.
        Additionally, it resets the boolean responsible for allowing to process demands.
        :param t_start The point in time of the flexibility
        :param min_p The minimum possible flexibility
        :param max_p The maximum possible flexibility
        """
        super().update_flexibility(t_start, min_p, max_p)
        self.current_time_spans.append(t_start)
        self.first_demand_received = False

    async def answer_external_request(self, message, message_path, value, msg_type):
        """
        This method answers the external requests received by other agents.
        It gathers them in a list (offers), sorts them by ethics score and sends
        them to the agents.
        :param message The requirement representing the reply
        :param message_path The path of message from its sender to this agent
        :param value The flexibility value set for this agent
        :param msg_type The type of the message received
        """
        if self.use_consumer_ethics_score:
            await self.send_message(message, msg_path=message_path, forwarding=True)
            self.governor.message_journal.add(message)
            self.offer_list.append(message)
            if not self.first_demand_received:
                self.first_demand_received = True
                await asyncio.sleep(self.request_processing_waiting_time)
                offers = deepcopy(self.offer_list)
                self.offer_list.clear()
                offers.sort(key=self.get_ethics_score, reverse=True)
                await self.send_offers_to_most_ethical_agents(offer_list=offers)
                self.first_demand_received = False
            else:
                return
        else:
            try:
                await super().answer_external_request(message, message_path, value, msg_type)
            except Exception as e:
                print(e)

    async def handle_demand_or_offer_reply(self, requirement, message_path):
        """
        This method uses the reply_processing_waiting_time to wait for replies after
        the first one has been received. It then sends over the replies to the solver and
        repeats the process.
        :param requirement The requirement representing the reply
        :param message_path The path of message from its sender to this agent
        """
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
                    await asyncio.sleep(self.reply_processing_waiting_time)
                    await self.solve()
                    self.first_offer_received = False
        else:
            try:
                await super().handle_demand_or_offer_reply(requirement, message_path)
            except Exception as e:
                print(e)

    async def reset(self):
        """
        After a negotiation, reset the booleans responsible for allowing the processing of
        offers and demands and add your own flex to the solution.
        Afterwards, call the reset method of the super class.
        """
        self.add_own_flex_to_solution()
        self.first_offer_received = False
        self.first_demand_received = False
        await super().reset()

    def add_own_flex_to_solution(self):
        flex = self.get_flexibility_for_interval(self.current_time_spans)
        own_request = self.governor.get_from_power_balance(self.aid, self.current_time_spans[0])
        for flex_value in flex:
            for time_span in own_request.time_span:
                if flex_value > 0 and time_span in own_request.time_span:
                    self.result[self.aid] = flex_value
                    self.result_sum += flex_value

    def calc_result_sum(self):
        self.result_sum = 0
        for res in self.result.values():
            self.result_sum += res

    async def send_offers_to_most_ethical_agents(self, offer_list):
        """
        Offers are sent to the requesting agents until either all requests have been answered
        or the agent's flexibility is depleted for the requested time frames.
        :param offer_list the list of offers collected and sorted by ethics score
        """
        temp_flex = {}
        specific_offer_values = []
        for offer in offer_list:
            if offer.msg_type == 5:
                msg_to_answer_with = 6
                flex_to_choose = 0
            else:
                msg_to_answer_with = 5
                flex_to_choose = 0
            for time_slot in offer.time_span:
                if time_slot not in temp_flex:
                    temp_flex[time_slot] = self.get_flexibility_for_interval(time_slot)
                try:
                    if abs(offer.value[len(specific_offer_values)]) >= abs(temp_flex[time_slot][flex_to_choose]):
                        specific_offer_values.append(temp_flex[time_slot][flex_to_choose])
                        temp_flex[time_slot][flex_to_choose] = 0
                    else:
                        temp_flex[time_slot][flex_to_choose] = temp_flex[time_slot][flex_to_choose] - \
                                                               offer.value[len(specific_offer_values)]
                        specific_offer_values.append(offer.value[len(specific_offer_values)])
                except Exception as e:
                    print(e)
            if not all(value == 0 for value in specific_offer_values):
                reply = WinzentMessage(msg_type=msg_to_answer_with,
                                       sender=self._aid,
                                       is_answer=True,
                                       receiver=offer.sender,
                                       time_span=offer.time_span,
                                       value=specific_offer_values, ttl=self._current_ttl,
                                       id=str(uuid.uuid4()),
                                       ethics_score=self.ethics_score)
                self._current_inquiries_from_agents[reply.id] = reply
                await self.send_message(reply)
            specific_offer_values.clear()
