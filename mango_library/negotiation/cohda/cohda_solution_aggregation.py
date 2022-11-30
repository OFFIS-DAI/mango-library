import logging
from uuid import UUID
from typing import Dict, Tuple, Optional, List
from mango.role.api import Role

from mango_library.negotiation.cohda.cohda_messages import CohdaSolutionRequestMessage, CohdaProposedSolutionMessage, \
    CohdaFinalSolutionMessage
from mango_library.negotiation.cohda.data_classes import SolutionCandidate
from mango_library.negotiation.cohda.cohda_negotiation import COHDANegotiation
from mango_library.negotiation.termination import InformAboutTerminationMessage

logger = logging.getLogger(__name__)


class CohdaSolutionAggregationRole(Role):
    """

    """
    def __init__(self, ):
        """

        """
        super().__init__()
        self.cohda_solutions: Dict[UUID, SolutionCandidate] = {}
        self.open_solution_requests: Dict[UUID, Dict[Tuple, Optional[SolutionCandidate]]] = {}

    def setup(self):
        super().setup()
        self.context.subscribe_message(self, self.handle_inform_about_term,
                                       message_condition=lambda c, _: isinstance(c, InformAboutTerminationMessage))
        self.context.subscribe_message(self, self.handle_cohda_solution,
                                       message_condition=lambda c, _: isinstance(c, CohdaProposedSolutionMessage))

    def handle_inform_about_term(self, content: InformAboutTerminationMessage, meta):
        """

        :param content:
        :param meta:
        :return:
        """
        # we have a new terminated COHDA negotiation.
        # check if it is really new
        if content.negotiation_id in self.cohda_solutions.keys():
            logger.warning(f'Received a InformAboutTerminationMessage, however there is already a solution for the '
                           f'negotiation id {content.negotiation_id}.')
            return

        if content.negotiation_id in self.open_solution_requests.keys():
            logger.warning(f'Received a InformAboutTerminationMessage, however there is already an attempt to get a'
                           f'aggregated solution for negotiation id {content.negotiation_id}.')
            return

        # We will create an open solution request
        self.open_solution_requests[content.negotiation_id] = {}

        futs = []
        # We will now add all agents as keys with None as Values (we don't know their solution candidate yet)
        for agent_addr, agent_id in content.participants:
            if isinstance(agent_addr, list):
                agent_addr = tuple(agent_addr)
            self.open_solution_requests[content.negotiation_id][(agent_addr, agent_id)] = None

        # Let's ask for all solutions from the individual agents
        for agent_addr, agent_id in content.participants:
            self.context.schedule_instant_task(self.context.send_acl_message(
                content=CohdaSolutionRequestMessage(negotiation_id=content.negotiation_id),
                receiver_addr=agent_addr,
                receiver_id=agent_id
            ))

    def handle_cohda_solution(self, content: CohdaProposedSolutionMessage, meta):
        """

        :param content:
        :param meta:
        :return:
        """
        sender_addr = meta['sender_addr']
        sender_id = meta['sender_id']
        negotiation_id = content.negotiation_id
        if isinstance(sender_addr, list):
            sender_addr = tuple(sender_addr)

        if negotiation_id not in self.open_solution_requests:
            logger.warning(f'Received a COHDA Solution for negotiation_id {negotiation_id}, but there is no'
                           f'open solution request for this id.')
            return

        if (sender_addr, sender_id) not in self.open_solution_requests[negotiation_id].keys():
            logger.warning(f'Received a COHDA Solution for negotiation_id {negotiation_id}'
                           f' from {(sender_addr, sender_id)}, but there is no negotiation '
                           f'participant with this address.')
            return

        self.open_solution_requests[negotiation_id][(sender_addr, sender_id)] = content.solution_candidate

        if None not in self.open_solution_requests[negotiation_id].values():
            # we have received all individual solutions, so we can aggregate
            final_solution = self.aggregate_solution(negotiation_id=negotiation_id)
            # stor the result
            self.cohda_solutions[negotiation_id] = final_solution
            # inform all participants
            for agent_addr, agent_id in self.open_solution_requests[negotiation_id].keys():
                self.context.schedule_instant_task(self.context.send_acl_message(
                    content=CohdaFinalSolutionMessage(solution_candidate=final_solution, negotiation_id=negotiation_id),
                    receiver_addr=agent_addr, receiver_id=agent_id
                ))
            # delete from open requests dict
            del self.open_solution_requests[negotiation_id]

    def aggregate_solution(self, negotiation_id: UUID) -> SolutionCandidate:
        """

        :param negotiation_id:
        :return:
        """
        current_best_candidate = None
        for candidate in self.open_solution_requests[negotiation_id].values():
            if current_best_candidate is None:
                current_best_candidate = candidate
            else:
                current_best_candidate = COHDANegotiation._merge_candidates(
                    current_best_candidate, candidate, agent_id=self.context.aid,
                    perf_func=lambda *_: float('-inf'), target_params=None)

        return current_best_candidate







