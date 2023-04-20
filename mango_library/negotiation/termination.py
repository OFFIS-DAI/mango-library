"""Module, which implements a simple termination detection for negotiations. Here Huangs
detection algorithm is used (10.1109/ICDCS.1989.37933).

It requires the distributed negotiation to have some kind of controller agent. In general
this can often be the initiator.

Roles:
* :class:`NegotiationTerminationRole`: role for the participants, hooks into sending messages,
                                       adding the weight value

Messages:
* :class:`TerminationMessage`: this message will be sent to the controller, when an agent
considers itself as inactive.
"""
import asyncio
from fractions import Fraction
from typing import Dict, Any, Union, Optional, Set, Tuple, Callable, List
from uuid import UUID

from mango.messages.codecs import json_serializable
from mango import Role

from mango_library.negotiation.cohda.cohda_messages import CohdaNegotiationMessage
from .cohda.cohda_messages import StopNegotiationMessage
from .cohda.cohda_negotiation import CohdaNegotiationModel
from ..coalition.core import CoalitionModel, CoalitionAssignment


@json_serializable
class TerminationMessage:
    """Message for sending the remaining weight to the controller"""

    def __init__(
        self, weight: Fraction, coalition_id: UUID, negotiation_id: UUID
    ) -> None:
        self._weight = weight
        self._coalition_id = coalition_id
        self._negotiation_id = negotiation_id

    @property
    def weight(self) -> Fraction:
        """Return the remaining weight

        :return: remaining weight
        """
        return self._weight

    @property
    def coalition_id(self) -> UUID:
        """Return the coalition id the negotiation is referring to

        :return: the coalition id
        """
        return self._coalition_id

    @property
    def negotiation_id(self) -> UUID:
        """Return the negotiation id

        :return: the negotiation id
        """
        return self._negotiation_id


@json_serializable
class InformAboutTerminationMessage:
    """
    Message that informs an agent that a negotiation should be stopped.
    """

    def __init__(self, negotiation_id: UUID, participants: List[Tuple]) -> None:
        self._negotiation_id = negotiation_id
        self._participants = participants

    @property
    def negotiation_id(self) -> UUID:
        """Return the negotiation id

        :return: the negotiation id
        """
        return self._negotiation_id

    @property
    def participants(self) -> List[Tuple]:
        """Return the list of participants as tuple (addr, aid)

        :return: the participants
        """
        return self._participants


class NegotiationTerminationParticipantRole(Role):
    """Role for negotiation participants. Will add the weight attribute to every
    coalition related message send.
    """

    def __init__(
        self,
        negotiation_model_class=CohdaNegotiationModel,
        negotiation_message_class=CohdaNegotiationMessage,
    ):
        super().__init__()
        self._negotiation_message_class = negotiation_message_class
        self._negotiation_model_class = negotiation_model_class
        self._negotiation_model = (
            None  # will be defined in setup(), once the context exists
        )
        self._weight_map: Dict[UUID, Fraction] = {}
        self._termination_check_tasks: Dict[UUID, asyncio.Task] = {}

    def setup(self):
        super().setup()
        self._negotiation_model = self.context.get_or_create_model(
            self._negotiation_model_class
        )
        self.context.subscribe_send(self, self.on_send)
        self.context.subscribe_message(
            self,
            self.handle_neg_msg,
            lambda c, _: isinstance(c, self._negotiation_message_class),
        )

    def on_send(
        self,
        content,
        receiver_addr: Union[str, Tuple[str, int]],
        *,
        receiver_id: Optional[str] = None,
        **kwargs
    ):
        """Add the weight to every coalition related message

        :param content: content of the message
        :param receiver_addr: address
        :param receiver_id: id of the receiver. Defaults to None.
        :param kwargs: additional parameters
        """
        if isinstance(content, self._negotiation_message_class):

            if content.negotiation_id not in self._weight_map:
                self._weight_map[content.negotiation_id] = Fraction(0, 1)
            if not hasattr(content, "message_weight") or content.message_weight is None:
                content.message_weight = self._weight_map[content.negotiation_id] / 2
            self._weight_map[content.negotiation_id] /= 2

    def handle_neg_msg(self, content, _: Dict[str, Any]) -> None:
        """Check whether a coalition related message has been received and manipulate the internal
        weight accordingly. Setup a conditional task that checks for termination.

        :param content: the incoming neogtiation message
        :param _: the meta data
        """
        if content.negotiation_id in self._weight_map:
            self._weight_map[content.negotiation_id] += content.message_weight
        else:
            self._weight_map[content.negotiation_id] = content.message_weight

        coalition_assignment: CoalitionAssignment = self.context.get_or_create_model(
            CoalitionModel
        ).by_id(content.coalition_id)

        term_detector = (
            coalition_assignment.controller_agent_addr,
            coalition_assignment.controller_agent_id,
        )

        def _check_weight_condition() -> bool:
            """
            Function that checks whether the negotiation has 'probably' terminated.
            :return: boolean
            """
            return (
                not self._negotiation_model.by_id(
                    negotiation_id=content.negotiation_id
                ).active
                and self._weight_map[content.negotiation_id] != 0
            )

        if (
            content.negotiation_id not in self._termination_check_tasks
            or self._termination_check_tasks[content.negotiation_id].done()
        ):
            # create a new conditional task that checks for termination
            self._termination_check_tasks[
                content.negotiation_id
            ] = self.context.schedule_conditional_task(
                self._send_weight(term_detector, content),
                condition_func=_check_weight_condition,
            )

    async def _send_weight(self, termination_detector: Tuple, content):
        """
        Sends the current weight to the termination controller
        :param termination_detector: Address (addr, aid) of the termination detector that should receive the
        weight message
        :param content: The NeogotiationMessage

        """
        # store weight
        current_weight = self._weight_map[content.negotiation_id]
        # reset weight
        self._weight_map[content.negotiation_id] = Fraction(0, 1)
        # Send weight
        await (
            self.context.send_acl_message(
                content=TerminationMessage(
                    current_weight, content.coalition_id, content.negotiation_id
                ),
                receiver_addr=termination_detector[0],
                receiver_id=termination_detector[1],
                acl_metadata={
                    "sender_addr": self.context.addr,
                    "sender_id": self.context.aid,
                },
            )
        )


class NegotiationTerminationDetectorRole(Role):
    """ """

    def __init__(
        self,
        on_termination: Callable = None,
        aggregator_addr=None,
        aggregator_id: str = None,
    ):
        super().__init__()
        self._weight_map: Dict[UUID, Fraction] = {}
        self._participant_map: Dict[UUID, Set[Tuple[Tuple, str]]] = {}
        self._on_termination = (
            on_termination if on_termination is not None else self._send_stop_and_inform
        )
        self._aggregator_addr = aggregator_addr
        self._aggregator_id = aggregator_id

    def setup(self):
        super().setup()
        self.context.subscribe_message(
            self, self.handle_term_msg, lambda c, _: isinstance(c, TerminationMessage)
        )
        if self._aggregator_addr is None and self._aggregator_id is None:
            self._aggregator_addr = self.context.addr
            self._aggregator_id = self.context.aid

    async def _send_stop_and_inform(self, negotiation_id):
        # send stopNegotiationMessage first
        for agent_addr, agent_id in self._participant_map[negotiation_id]:
            await self.context.send_acl_message(
                content=StopNegotiationMessage(negotiation_id=negotiation_id),
                receiver_addr=agent_addr,
                receiver_id=agent_id,
                acl_metadata={
                    "sender_addr": self.context.addr,
                    "sender_id": self.context.aid,
                },
            )

        # now send message to aggregator
        if self._aggregator_addr is not None and self._aggregator_id is not None:
            await self.context.send_acl_message(
                content=InformAboutTerminationMessage(
                    negotiation_id=negotiation_id,
                    participants=list(self._participant_map[negotiation_id]),
                ),
                receiver_addr=self._aggregator_addr,
                receiver_id=self._aggregator_id,
                acl_metadata={
                    "sender_addr": self.context.addr,
                    "sender_id": self.context.aid,
                },
            )

    def handle_term_msg(
        self, content: TerminationMessage, meta: Dict[str, Any]
    ) -> None:
        """Handle the termination message.

        :param content: the message
        :param meta: meta data
        """
        neg_id = content.negotiation_id
        if "sender_addr" in meta and "sender_id" in meta:
            sender_addr = meta["sender_addr"]
            if isinstance(sender_addr, list):
                sender_addr = tuple(sender_addr)

            if neg_id not in self._participant_map:
                self._participant_map[neg_id] = set()
            self._participant_map[neg_id].add((sender_addr, meta["sender_id"]))

        if neg_id not in self._weight_map:
            self._weight_map[neg_id] = content.weight
        else:
            self._weight_map[neg_id] += content.weight

        if self._weight_map[neg_id] == 1:
            self.context.schedule_instant_task(self._on_termination(neg_id))
