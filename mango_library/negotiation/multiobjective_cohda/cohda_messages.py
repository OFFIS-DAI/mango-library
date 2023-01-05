from uuid import UUID
from fractions import Fraction

from mango.messages.codecs import json_serializable

from mango_library.negotiation.multiobjective_cohda.data_classes import WorkingMemory, SolutionCandidate


@json_serializable
class MoCohdaNegotiationMessage:
    """
    Message for a COHDa negotiation.
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