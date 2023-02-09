from uuid import UUID
from fractions import Fraction

from mango.messages.codecs import json_serializable

from mango_library.negotiation.cohda.data_classes import WorkingMemory, SolutionCandidate


@json_serializable
class StartCohdaNegotiationMessage:
    """
        Message to start a COHDA negotiation.
        Contains the coalition_id of the associated coalition
        and the target parameters that are necessary for the agents to calculate the performance.
        """
    def __init__(self, coalition_id, send_weight, target_params = None):
        self.coalition_id = coalition_id
        self._send_weight = send_weight
        self._target_params = target_params

    @property
    def send_weight(self):
        return self._send_weight

    @property
    def target_params(self):
        return self._target_params

@json_serializable
class CohdaNegotiationMessage:
    """
    Message for a COHDA negotiation.
    Contains the working memory of an agent.
    """

    def __init__(self, working_memory: WorkingMemory, negotiation_id: UUID, coalition_id: UUID,
                 message_weight: Fraction = None):
        self._working_memory = working_memory
        self._negotiation_id = negotiation_id
        self._coalition_id = coalition_id
        self._message_weight = message_weight   # for termination, can be set after initialization of the message

    @property
    def working_memory(self) -> WorkingMemory:
        """Return the working memory of the sender agent

        :return: the working memory of the sender
        """
        return self._working_memory

    @property
    def negotiation_id(self) -> UUID:
        """Return the negotiation_id of the COHDA negotiation

        :return: the negotiation_id of the COHDA negotiation
        """
        return self._negotiation_id

    @property
    def coalition_id(self) -> UUID:
        """Return the coalition_id of the COHDA negotiation

        :return: the coalition_id of the COHDA negotiation
        """
        return self._coalition_id

    @property
    def message_weight(self) -> Fraction:
        """Return the message weight of the message

        :return: the message weight
        """
        return self._message_weight

    @message_weight.setter
    def message_weight(self, new_weight: Fraction) -> None:
        """
        Sets the message_weight of the message
        :param new_weight: The new message
        """
        self._message_weight = new_weight


@json_serializable
class StopNegotiationMessage:
    """
    Message that informs an agent that a negotiation should be stopped.
    """
    def __init__(self, negotiation_id: UUID) -> None:
        self._negotiation_id = negotiation_id

    @property
    def negotiation_id(self) -> UUID:
        """Return the negotiation id

        :return: the negotiation id
        """
        return self._negotiation_id


@json_serializable
class CohdaSolutionRequestMessage:
    """
    Message that asks the agent for its current SolutionCandidate regarding a specific negotiation
    """
    def __init__(self, negotiation_id: UUID) -> None:
        self._negotiation_id = negotiation_id

    @property
    def negotiation_id(self) -> UUID:
        """Return the negotiation id

        :return: the negotiation id
        """
        return self._negotiation_id


@json_serializable
class CohdaProposedSolutionMessage:
    """
    Message for a COHDA solution.
    Contains the candidate of an agent.
    """

    def __init__(self, solution_candidate: SolutionCandidate, negotiation_id: UUID):
        self._solution_candidate = solution_candidate
        self._negotiation_id = negotiation_id

    @property
    def solution_candidate(self) -> SolutionCandidate:
        """Return the solution candidate of the sender agent

        :return: the solution_candidate of the sender
        """
        return self._solution_candidate

    @property
    def negotiation_id(self) -> UUID:
        """Return the negotiation_id of the corresponding solution

        :return: the negotiation_id
        """
        return self._negotiation_id


@json_serializable
class CohdaFinalSolutionMessage:
    """
    Message for a final COHDA solution.
    Contains the final candidate after aggregation.
    """

    def __init__(self, solution_candidate: SolutionCandidate, negotiation_id: UUID):
        self._solution_candidate = solution_candidate
        self._negotiation_id = negotiation_id

    @property
    def solution_candidate(self) -> SolutionCandidate:
        """Return the solution candidate of the sender agent

        :return: the solution_candidate of the sender
        """
        return self._solution_candidate

    @property
    def negotiation_id(self) -> UUID:
        """Return the negotiation_id of the corresponding solution

        :return: the negotiation_id
        """
        return self._negotiation_id


@json_serializable
class ConfirmCohdaSolutionMessage:
    """
    Message that is sent by unit Agents to the Aggregator Agent to confirm that they have received the final solution.
    """

    def __init__(self, negotiation_id: UUID):
        self._negotiation_id = negotiation_id

    @property
    def negotiation_id(self) -> UUID:
        """Return the negotiation_id of the solution that was received

        :return: the negotiation_id
        """
        return self._negotiation_id
