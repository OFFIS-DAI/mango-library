import asyncio
import logging
import math
import uuid
from abc import ABC
from copy import deepcopy

from mango_library.negotiation.winzent import xboole
from mango_library.negotiation.winzent.winzent_message_pb2 import WinzentMessage
from mango_library.negotiation.winzent.winzent_base_agent import WinzentBaseAgent


logger = logging.getLogger(__name__)


class WinzentEthicalAgent(WinzentBaseAgent, ABC):
    """
    The ethical Winzent version. Overrides message handling and uses a different solving
    strategy, the XbooleEthicalPowerBalanceSolverStrategy.
    :param request_processing_waiting_time: The time the agent waits for additional requests
           after the first one arrived.
    :param reply_processing_waiting_time: The time the agent waits for additional initial replies
           after the first one arrived.
    :param use_ethics_score_as_negotiator: If true, activates the use of the ethics score for the agent
           who initiates the negotiation.
    :param use_ethics_score_as_contributor: If true, activates the use of the ethics score for the agent
           who received a request from a negotiating agent.
    :param min_coverage: The minimum coverage of solution to be accepted as a valid solution. Ranges from
           0.0 to 1.0.
    :param coverage_weight: The weight of the coverage when calculating a solution. Ranges from 0.0 to 1.0.
           The other component influencing the solution evaluation is the combined ethics score of the solution.
           It consequently has a weight of: 1-coverage_weight.
    """

    def __init__(self, container, ttl, time_to_sleep=3, send_message_paths=False, ethics_score=1,
                 request_processing_waiting_time=0.2,
                 reply_processing_waiting_time=0.4,
                 use_ethics_score_as_negotiator=True,
                 use_ethics_score_as_contributor=True,
                 min_coverage=0.9,
                 coverage_weight=0.4,
                 elem_type=None,
                 index=-1,

                 ):
        super().__init__(container, ttl, time_to_sleep, send_message_paths, ethics_score, elem_type, index)
        # the current time spans that are negotiated about
        self.current_time_spans = []
        # the list of initial replies that stores the offer for the length of 'reply_processing_waiting_time'.
        self.initial_reply_list = []
        self.first_initial_reply_received = False
        self.first_request_received = False
        self.use_ethics_score_as_negotiator = use_ethics_score_as_negotiator
        self.use_ethics_score_as_contributor = use_ethics_score_as_contributor
        self.request_processing_waiting_time = request_processing_waiting_time
        self.reply_processing_waiting_time = reply_processing_waiting_time
        # override base agent power balance strategy
        self.governor.power_balance_strategy = \
            xboole.XbooleEthicalPowerBalanceSolverStrategy(min_coverage=min_coverage,
                                                           coverage_weight=coverage_weight,
                                                           ethics_score_weight=1 - coverage_weight
                                                           )
        self.decay_rate = 0,
        self.sub_tier_size = 0,
        self.calc_ethics_score_params()
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
        self.first_request_received = False

    async def answer_external_request(self, message, message_path, values, msg_type):
        """
        This method answers the external requests received by other agents.
        It gathers them in a list (offers), sorts them by ethics score and sends
        them to the agents.
        :param message The requirement representing the reply
        :param message_path The path of message from its sender to this agent
        :param values The flexibility value set for this agent
        :param msg_type The type of the message received
        """
        if self.use_ethics_score_as_contributor:
            await self.send_message(message, msg_path=message_path, forwarding=True)
            self.governor.message_journal.add(message)
            self.initial_reply_list.append(message)
            if not self.first_request_received:
                self.first_request_received = True
                await asyncio.sleep(self.request_processing_waiting_time)
                offers = deepcopy(self.initial_reply_list)
                self.initial_reply_list.clear()
                offers.sort(key=self.get_ethics_score, reverse=True)
                await self.send_initial_replies_to_highest_ethics_scores(request_list=offers)
                self.first_request_received = False
            else:
                return
        else:
            try:
                await super().answer_external_request(message, message_path, values, msg_type)
            except Exception as e:
                print(e)

    async def handle_initial_reply(self, requirement, message_path):
        """
        This method uses the reply_processing_waiting_time to wait for replies after
        the first one has been received. It then sends over the replies to the solver and
        repeats the process.
        :param requirement The requirement representing the reply
        :param message_path The path of message from its sender to this agent
        """
        if self.use_ethics_score_as_negotiator:
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
                if not self.first_initial_reply_received:
                    self.first_initial_reply_received = True
                    await asyncio.sleep(self.reply_processing_waiting_time)
                    await self.solve()
                    self.first_initial_reply_received = False
        else:
            try:
                await super().handle_initial_reply(requirement, message_path)
            except Exception as e:
                print(e)

    async def reset(self):
        """
        After a negotiation, reset the booleans responsible for allowing the processing of
        offers and demands and add your own flex to the solution.
        Afterwards, call the reset method of the super class.
        """
        self.add_own_flex_to_solution()
        self.first_initial_reply_received = False
        self.first_request_received = False
        await super().reset()

    def add_own_flex_to_solution(self):
        """
        This method adds the flex the agent has himself to the results of his negotiation.
        """
        flex = self.get_flexibility_for_interval(self.current_time_spans)
        own_request = self.governor.get_from_power_balance(self.aid, self.current_time_spans[0])
        if own_request is not None:
            for flex_value in flex:
                for time_span in own_request.time_span:
                    if flex_value > 0 and time_span in own_request.time_span:
                        self.result[self.aid] = flex_value
                        self.result_sum += flex_value

    def calc_ethics_score_params(self, end=10, step_size=1):
        """
        This method calculates the parameters that dictate the change of the ethics score in case
        of an unsucessful negotiation. This parameter are self.sub_tier_size and self.decay_rate.
        :param end: The value/number of the last step of simulation. Can be certain time or
                    a step number, depending on the simulation. If there is no set last step,
                    this number can be altered as desired.
        :param step_size: The size of the steps towards the end.
        """
        total_amount_of_steps = end / step_size
        self.sub_tier_size = 1.0 / total_amount_of_steps
        self.decay_rate = self.sub_tier_size / total_amount_of_steps

    def calculate_new_ethics_score(self, success):
        """
        After a negotiation, this method calculates the new ethics score based on the success of
        the previous negotiation. For that, it uses the variables self.sub_tier_size and
        self.decay_rate.
        :param success: A boolean determining if the negotiation was successful or not.
        :return self.ethics_score: The ethics score is returned for logging and analysis purposes.
        """
        max_len_of_ethics_score = "{:." + str(len(str(self.decay_rate).replace('.', ''))) + "f}"
        initial_ethics_score = float(math.floor(self.ethics_score))
        str_eth_score = list(str(self.ethics_score))
        str_eth_score[0] = "0"
        str_eth_score = float("".join(str_eth_score))
        amount_of_outages = int(str_eth_score / self.sub_tier_size)
        current_tier_low = max(float(str(self.ethics_score)[0]) + (amount_of_outages * (self.sub_tier_size)),
                               initial_ethics_score)
        current_tier_high = max(float(str(self.ethics_score)[0]) + ((amount_of_outages + 1) * self.sub_tier_size),
                                initial_ethics_score)
        if not success:
            temp = math.floor(self.ethics_score * 10) / 10
            if (math.floor(float(temp)) + 1) > (float(temp) + self.sub_tier_size):
                if self.ethics_score == initial_ethics_score:
                    self.ethics_score = float(
                        max_len_of_ethics_score.format(initial_ethics_score + self.sub_tier_size - self.decay_rate))
                    return self.ethics_score
                self.ethics_score = float(
                    max_len_of_ethics_score.format(current_tier_high + self.sub_tier_size - self.decay_rate))
                return self.ethics_score
            else:
                self.ethics_score = float(
                    max_len_of_ethics_score.format((math.floor(float(self.ethics_score)) + 1) - self.decay_rate))
                return self.ethics_score
        else:
            temp_ethics_score = float(max_len_of_ethics_score.format(self.ethics_score - self.decay_rate))
            if temp_ethics_score <= current_tier_low:
                self.ethics_score = current_tier_low
                return self.ethics_score
            else:
                self.ethics_score = temp_ethics_score
                return self.ethics_score

    async def send_initial_replies_to_highest_ethics_scores(self, request_list):
        """
        Offers are sent to the requesting agents until either all requests have been answered
        or the agent's flexibility is depleted for the requested time frames.
        :param request_list: the list of requests collected and sorted by ethics score
        """
        temp_flex = {}
        specific_request_values = []
        for initial_request in request_list:
            flex_to_choose = 0
            if initial_request.msg_type == 5:
                # If the request is of the type 5, meaning it offers power, the agent answers with
                # his ability to consume power, which is a negative score.
                msg_to_answer_with = 6
            else:
                # If the request is of the type 6, meaning it wants power, the agent answers with
                # his ability to generate power, which is a positive score.
                msg_to_answer_with = 5
            for time_slot in initial_request.time_span:
                if time_slot not in temp_flex:
                    temp_flex[time_slot] = self.get_flexibility_for_interval(time_slot)
                try:
                    if abs(initial_request.value[len(specific_request_values)]) >= abs(
                            temp_flex[time_slot][flex_to_choose]):
                        specific_request_values.append(temp_flex[time_slot][flex_to_choose])
                        temp_flex[time_slot][flex_to_choose] = 0
                    else:
                        temp_flex[time_slot][flex_to_choose] = temp_flex[time_slot][flex_to_choose] - \
                                                               initial_request.value[len(specific_request_values)]
                        specific_request_values.append(initial_request.value[len(specific_request_values)])
                except Exception as e:
                    print(e)
            if not all(value == 0 for value in specific_request_values):
                logger.debug(f"{self.aid}: sending reply with offered values to {initial_request.sender}")
                reply = WinzentMessage(msg_type=msg_to_answer_with,
                                       sender=self.aid,
                                       is_answer=True,
                                       receiver=initial_request.sender,
                                       time_span=initial_request.time_span,
                                       value=specific_request_values, ttl=self._current_ttl,
                                       id=str(uuid.uuid4()),
                                       ethics_score=self.ethics_score)
                self._current_inquiries_from_agents[reply.id] = reply
                await self.send_message(reply)
            specific_request_values.clear()