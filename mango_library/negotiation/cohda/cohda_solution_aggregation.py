import logging
from uuid import UUID
from typing import Dict, Tuple, Optional, List

from mango import Role

from mango_library.negotiation.cohda.cohda_messages import (
    CohdaSolutionRequestMessage,
    CohdaProposedSolutionMessage,
    CohdaFinalSolutionMessage,
    ConfirmCohdaSolutionMessage,
)
from mango_library.negotiation.cohda.data_classes import SolutionCandidate
from mango_library.negotiation.cohda.cohda_negotiation import COHDANegotiation
from mango_library.negotiation.termination import InformAboutTerminationMessage

logger = logging.getLogger(__name__)


class CohdaSolutionAggregationRole(Role):
    """
    This Role aggregates the CohdaSolutions of single agents.
    It reacts on an incoming InformAboutTerminationMessage, sends a SolutionRequest to all participants.
    Once all replies have arrived, it aggregates the solutions to one final solution and then sends a
    CohdaFinalSolutionMessage to all participants with the final candidate
    """

    def __init__(self):
        """
        Init the CohdaSolutionAggregationRole
        """
        super().__init__()
        self.cohda_solutions: Dict[
            UUID, SolutionCandidate
        ] = {}  # holds teh final solutions
        self._open_solution_requests: Dict[
            UUID, Dict[Tuple, Optional[SolutionCandidate]]
        ] = {}  # holds open requests
        self._open_confirmations: Dict[UUID, List[Tuple]] = {}
        self._confirmed_cohda_solutions: List[UUID] = []

    def setup(self):
        super().setup()
        self.context.subscribe_message(
            self,
            self.handle_inform_about_term,
            message_condition=lambda c, _: isinstance(c, InformAboutTerminationMessage),
        )
        self.context.subscribe_message(
            self,
            self.handle_cohda_solution,
            message_condition=lambda c, _: isinstance(c, CohdaProposedSolutionMessage),
        )
        self.context.subscribe_message(
            self,
            self.handle_solution_confirmation,
            message_condition=lambda c, _: isinstance(c, ConfirmCohdaSolutionMessage),
        )

    def handle_inform_about_term(self, content: InformAboutTerminationMessage, _):
        """
        Is called, once a InformAboutTerminationMessage has arrived.
        Here, a new open_solution_requests will be created and a CohdaSolutionRequestMessage will be sent to all
        participants
        :param content: The InformAboutTerminationMessage
        :param _: Meta dict
        """
        # we have a new terminated COHDA negotiation.
        # check if it is really new
        if content.negotiation_id in self.cohda_solutions.keys():
            logger.warning(
                f"Received a InformAboutTerminationMessage, however there is already a solution for the "
                f"negotiation id {content.negotiation_id}."
            )
            return

        if content.negotiation_id in self._open_solution_requests.keys():
            logger.warning(
                f"Received a InformAboutTerminationMessage, however there is already an attempt to get a"
                f"aggregated solution for negotiation id {content.negotiation_id}."
            )
            return

        # We will create an open solution request
        self._open_solution_requests[content.negotiation_id] = {}

        # We will now add all agents as keys with None as Values (we don't know their solution candidate yet)
        for agent_addr, agent_id in content.participants:
            if isinstance(agent_addr, list):
                agent_addr = tuple(agent_addr)  # so we can add them as keys in our list
            self._open_solution_requests[content.negotiation_id][
                (agent_addr, agent_id)
            ] = None

        # Ask for all solutions from the participating agents
        for agent_addr, agent_id in content.participants:
            self.context.schedule_instant_acl_message(
                content=CohdaSolutionRequestMessage(
                    negotiation_id=content.negotiation_id
                ),
                receiver_addr=agent_addr,
                receiver_id=agent_id,
                acl_metadata={"sender_id": self.context.aid},
            )

    def handle_cohda_solution(self, content: CohdaProposedSolutionMessage, meta):
        """Is called, once a CohdaProposedSolutionMessage arrives.
        It will first add the proposed solution to the open_solution_requests.
        In case all proposed solutions have arrived, it will start the aggregation
        :param content: The CohdaProposedSolutionMessage
        :param meta: The meta dict
        """

        # get sender and id
        sender_addr = meta["sender_addr"]
        sender_id = meta["sender_id"]
        negotiation_id = content.negotiation_id
        if isinstance(sender_addr, list):
            sender_addr = tuple(
                sender_addr
            )  # so we can check if it is equal to our keys in the dict

        if negotiation_id not in self._open_solution_requests:
            logger.warning(
                f"Received a COHDA Solution for negotiation_id {negotiation_id}, but there is no"
                f"open solution request for this id."
            )
            return

        if (sender_addr, sender_id) not in self._open_solution_requests[
            negotiation_id
        ].keys():
            logger.warning(
                f"Received a COHDA Solution for negotiation_id {negotiation_id}"
                f" from {(sender_addr, sender_id)}, but there is no negotiation "
                f"participant with this address."
            )
            return

        # add solution to the open_solution_requests dict
        self._open_solution_requests[negotiation_id][
            (sender_addr, sender_id)
        ] = content.solution_candidate

        if None not in self._open_solution_requests[negotiation_id].values():
            # we have received all individual solutions, so we can aggregate
            final_solution = self.aggregate_solution(
                list(self._open_solution_requests[negotiation_id].values())
            )
            # store the result
            self.cohda_solutions[negotiation_id] = final_solution
            self._open_confirmations[negotiation_id] = list(
                self._open_solution_requests[negotiation_id].keys()
            )
            # inform all participants
            for agent_addr, agent_id in self._open_solution_requests[
                negotiation_id
            ].keys():
                self.context.schedule_instant_acl_message(
                    content=CohdaFinalSolutionMessage(
                        solution_candidate=final_solution,
                        negotiation_id=negotiation_id,
                    ),
                    receiver_addr=agent_addr,
                    receiver_id=agent_id,
                    acl_metadata={"sender_id": self.context.aid},
                )

            # delete negotiation_id from open requests dict
            del self._open_solution_requests[negotiation_id]

    @staticmethod
    def aggregate_solution(candidates: List[SolutionCandidate]) -> SolutionCandidate:
        """
        Aggregates potentially different SolutionCandidates to one SolutionCandidate
        :param candidates: List of proposed Solution Candidates
        :return: The final SolutionCandidate
        """
        current_best_candidate = None

        for candidate in candidates:
            if current_best_candidate is None:
                # take current as best, since there is no other yet
                current_best_candidate = candidate
            else:
                # run merge candidates.
                # If participant list differs, create a new SolutionCandidate with all participants and set performance
                # to float('-inf').
                # This is necessary, as we don't know the performance function at this place
                current_best_candidate = COHDANegotiation._merge_candidates(
                    current_best_candidate,
                    candidate,
                    agent_id="Aggregation",
                    perf_func=lambda *_: float("-inf"),
                    target_params=None,
                )

        return current_best_candidate

    def handle_solution_confirmation(self, content: ConfirmCohdaSolutionMessage, meta):
        neg_id = content.negotiation_id

        # check if neg_id is expected and if you can extract sender information
        if neg_id not in self._open_confirmations.keys():
            logger.warning(
                f"Received a solution confirmation with strange neg_id {neg_id}"
            )
            return
        if "sender_addr" not in meta or "sender_id" not in meta:
            logger.warning("Cannot identify of sender of a ConfirmCohdaSolutionMessage")
            return

        # get sender information
        sender_addr = meta["sender_addr"]
        if isinstance(sender_addr, list):
            sender_addr = tuple(sender_addr)
        agent_key = sender_addr, meta["sender_id"]

        # check if you expect a rply from this agent
        if agent_key not in self._open_confirmations[neg_id]:
            logger.warning(
                f"Received a ConfirmCohdaSolutionMessage from {agent_key}, but don't"
                f"expect any from this agent"
            )
            return

        self._open_confirmations[neg_id].remove(agent_key)

        if len(self._open_confirmations[neg_id]) == 0:
            # all confirmations received
            del self._open_confirmations[neg_id]
            self._confirmed_cohda_solutions.append(neg_id)
