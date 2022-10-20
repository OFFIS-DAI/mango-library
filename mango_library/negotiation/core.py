"""This module implements some reacurring elements of negotiation processes. Therefore, a negotiation
wrapper for some meta data is provided. Furthermore, this module implements roles which can handle
this meta data models and store them as central role data.

Role-Models:
* :class:`NegotiationModel`: Stores information for all currently known negotiations

Messages:
* :class:`NeogtationMessage`: Wrapper message for negotiation oriented content

Roles:
* :class:`NegotiationStarterRole`: Starts a negotiation
* :class:`NegotiationParticipant`: (abstract) Participates in a negotiation, stores the meta data
                                   and wraps the real message in the negotiation wrapper message
"""
import uuid
from typing import Dict, Any, Union, Optional, Callable
from abc import ABC, abstractmethod
from fractions import Fraction
import logging

from mango.messages.codecs import json_serializable
from mango.role.api import ProactiveRole, SimpleReactiveRole
from mango.util.scheduling import ConditionalTask
from ..coalition.core import CoalitionAssignment, CoalitionModel

logger = logging.getLogger(__name__)


class Negotiation:
    """
    Model for storing the data regarding a concrete negotiation
    """

    def __init__(self, coalition_id: uuid.UUID, negotiation_id: uuid.UUID,
                 active: bool = True) -> None:
        self._negotiation_id = negotiation_id
        self._coalition_id = coalition_id
        self._active = active
        self._stopped = False

    @property
    def negotiation_id(self) -> uuid.UUID:
        """Return the negotiation id

        :return: the UUID
        """
        return self._negotiation_id

    @property
    def coalition_id(self) -> uuid.UUID:
        """Return the coalition id

        :return: the UUID
        """
        return self._coalition_id

    @property
    def active(self) -> bool:
        """Is seen as active

        :return: True if active, False otherwise
        """
        return self._active

    @active.setter
    def active(self, is_active) -> None:
        """Set is active

        :param is_active: active
        """
        self._active = is_active

    @property
    def stopped(self) -> bool:
        """
        Is seen as stopped
        :return: True if stopped, False otherwise
        """
        return self._active

    @stopped.setter
    def stopped(self, is_stopped) -> None:
        """
        Set is stopped
        :param is_stopped: stopped
        """
        self._stopped = is_stopped


class NegotiationModel:
    """Model for storing all metadata regarding negotiations
    """

    def __init__(self) -> None:
        self._negotiations = {}

    def by_id(self, negotiation_id: uuid.UUID) -> Negotiation:
        """Get a negotiation by id

        :param negotiation_id: id of the negotiation

        :return: the negotiation
        """
        return self._negotiations[negotiation_id]

    def exists(self, negotiation_id: uuid.UUID) -> bool:
        """Checks whether a negotiation exists

        :param negotiation_id: id of the negotiation

        :return: True if it exists, False otherwise
        """
        return negotiation_id in self._negotiations

    def add(self, negotiation_id: uuid.UUID, assignment: Negotiation):
        """Add a concrete negotiation

        :param negotiation_id: the UUID of the negotiation
        :param assignment: the assignment for the negotiation
        """
        self._negotiations[negotiation_id] = assignment


@json_serializable
class NegotiationMessage:
    """Message wrapper for negotiation messages.
    """

    def __init__(self, coalition_id: uuid.UUID, negotiation_id: uuid.UUID, message, message_weight=None) -> None:
        self._negotiation_id = negotiation_id
        self._coalition_id = coalition_id
        self._message = message
        self.message_weight = message_weight

    @property
    def negotiation_id(self) -> uuid.UUID:
        """Id of the negotiation

        :return: the id
        """
        return self._negotiation_id

    @property
    def coalition_id(self) -> uuid.UUID:
        """Id of the coalition this negotiation belongs to

        :return: UUID
        """
        return self._coalition_id

    @property
    def message(self):
        """Return the wrapped message

        :return: wrapped message
        """
        return self._message


@json_serializable
class StopNegotiationMessage:
    """
    Message that informs an agent that a negotiation should be stopped.
    """
    def __init__(self, negotiation_id: uuid.UUID) -> None:
        self._negotiation_id = negotiation_id

    @property
    def negotiation_id(self) -> uuid.UUID:
        """Return the negotiation id

        :return: the negotiation id
        """
        return self._negotiation_id


class NegotiationStarterRole(ProactiveRole):
    """Starting role for a negotiation. Will use a specific negotiation message creator to start
    a negotiation within its coalition.
    """

    def __init__(self, message_creator, coalition_model_matcher=None, coalition_uuid=None, send_weight: bool = True) \
            -> None:
        super().__init__()
        self._message_creator = message_creator
        self._send_weight = send_weight

        if coalition_uuid is not None:
            # if id is provided create matcher matching this id when the coalition is looked up
            self._coalition_model_matcher = lambda coal: coal.coalition_id == coalition_uuid
        elif coalition_model_matcher is not None:
            # when a matcher itself is provided use this if no uuid has been provided
            self._coalition_model_matcher = coalition_model_matcher
        else:
            # when nothing has been provided just match the first coalition
            self._coalition_model_matcher = lambda _: True
        
    def setup(self):
        super().setup()

        self.context.schedule_task(ConditionalTask(self.start(), self.is_startable))

    def is_startable(self):
        coalition_model = self.context.get_or_create_model(CoalitionModel)

        # check there is an assignment
        return len(coalition_model.assignments.values()) > 0

    def _look_up_assignment(self, all_assignments):
        for assignment in all_assignments:
            if self._coalition_model_matcher(assignment):
                return assignment
        # default to the first one
        return list(all_assignments)[0]

    async def start(self):
        """Start a negotiation. Send all neighbors a starting negotiation message.
        """
        coalition_model = self.context.get_or_create_model(CoalitionModel)

        # Assume there is a exactly one coalition
        matched_assignment = self._look_up_assignment(coalition_model.assignments.values())
        negotiation_uuid = uuid.uuid1()

        weight_per_msg = Fraction(1, len(matched_assignment.neighbors))  # relevant for termination detection
        for neighbor in matched_assignment.neighbors:
            neg_msg = NegotiationMessage(matched_assignment.coalition_id, negotiation_uuid,
                                         self._message_creator(matched_assignment))
            if self._send_weight:
                neg_msg.message_weight = weight_per_msg
            await self.context.send_message(
                content=neg_msg,
                receiver_addr=neighbor[1],
                receiver_id=neighbor[2],
                acl_metadata={'sender_addr': self.context.addr,
                              'sender_id': self.context.aid},
                create_acl=True)


class NegotiationParticipantRole(SimpleReactiveRole, ABC):
    """Abstract role for participating a negotiation. Handles the wrapper message, the stop message and the internal
    agent model about the meta data of the negotiation.
    """

    def __init__(self):
        """

        :param on_stop:
        """
        super().__init__()

    def handle_msg(self, content: Union[NegotiationMessage, StopNegotiationMessage],
                   meta: Dict[str, Any]):
        """Handles any NegotiationMessages, updating the internal model of the agent.

        :param content: the message
        :param meta: meta
        """
        if isinstance(content, NegotiationMessage):
            if not self.context.get_or_create_model(CoalitionModel).exists(content.coalition_id):
                return

            assignment = self.context.get_or_create_model(
                CoalitionModel).by_id(content.coalition_id)
            negotiation_model = self.context.get_or_create_model(NegotiationModel)

            if not negotiation_model.exists(content.negotiation_id):
                negotiation_model.add(content.negotiation_id, Negotiation(
                    content.coalition_id, content.negotiation_id))

            self.handle(content.message, assignment,
                        negotiation_model.by_id(content.negotiation_id), meta)

        elif isinstance(content, StopNegotiationMessage):

            # set stopped
            negotiation_model = self.context.get_or_create_model(NegotiationModel)
            if not negotiation_model.exists(content.negotiation_id):
                negotiation_model.add(content.negotiation_id, Negotiation(
                    content.coalition_id, content.negotiation_id))
            negotiation_model.by_id(content.negotiation_id).stopped = True
            self.handle_neg_stop(negotiation=negotiation_model.by_id(content.negotiation_id), meta=meta)

        else:
            logger.warning(f'NegotiationParticipantRole received unexpected Message of type {type(content)}')

    @abstractmethod
    def handle(self, message, assignment: CoalitionAssignment, negotiation: Negotiation, meta: Dict[str, Any]):
        """Handle the message and execute the specific negotiation step.

        :param message: the message
        :param assignment: the assignment the negotiations is in
        :param negotiation: the negotiation model
        :param meta: meta data
        """

    @abstractmethod
    def handle_neg_stop(self, negotiation: Negotiation, meta: Dict[str, Any]):
        """
        Behaviour once a StopNegotiationMessage is Received
        :param negotiation: the corresponding Negotiation object
        :param meta: meta of the message
        """

    async def send_to_neighbors(self, assignment: CoalitionAssignment, negotation: Negotiation, message):
        """Send a message to all neighbors

        :param assignment: the coalition you want to use the neighbors of
        :param negotation: the negotiation message
        :param message: the message you want to send
        """
        for neighbor in assignment.neighbors:
            await self.send(negotation, message, neighbor)

    async def send(self, negotation: Negotiation, message, neighbor) -> None:
        """Send a negotiation message to the specified neighbor

        :param negotation: the negotiation
        :param message: the content you want to send
        :param neighbor: the neighbor
        """
        await self.context.send_message(
            content=NegotiationMessage(negotation.coalition_id, negotation.negotiation_id, message),
            receiver_addr=neighbor[1], receiver_id=neighbor[2],
            acl_metadata={'sender_addr': self.context.addr, 'sender_id': self.context.aid},
            create_acl=True
        )

    def is_applicable(self, content, meta):
        return isinstance(content, NegotiationMessage) or isinstance(content, StopNegotiationMessage)
