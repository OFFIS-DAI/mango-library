"""Module, which implements a simple termination detection for negotiations. Here Huangs
detection algorithm is used (10.1109/ICDCS.1989.37933).

It requires the distributed negotiation to have some kind of controller agent. In general
this can often be the initiator.

Roles:
* :class:`NegotiationTerminationRole`: role for the participants, hooks into sending messages,
                                       adding the weight value

Messages:
* :class:`TerminationMessage`: this message will be send to the controller, when an agent
considers itself as inactive.
"""
import asyncio
from fractions import Fraction
from typing import Dict, Any, Union, Tuple, Optional
from uuid import UUID

from ..coalition.core import CoalitionModel
from .core import NegotiationModel
from mango.role.api import Role
from mango.messages.codecs import json_serializable


@json_serializable
class TerminationMessage:
    """Message for sending the remaining weight to the controller
    """
    def __init__(self, weight: Fraction, coalition_id: UUID, negotiation_id: UUID) -> None:
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


class NegotiationTerminationRole(Role):
    """Role for negotiation participants. Will add the weight attribute to every
    coalition related message send.
    """

    def __init__(self, is_controller: bool = 0) -> None:
        super().__init__()
        self._weight_map: Dict[UUID, Fraction] = {}
        self._termination_check_tasks: Dict[UUID, asyncio.Task] = {}
        self._is_controller: bool = is_controller

    def setup(self):
        super().setup()
        self.context.subscribe_send(self, self.on_send)
        self.context.subscribe_message(self, self.handle_term_msg, lambda c, _: isinstance(c, TerminationMessage))
        self.context.subscribe_message(self, self.handle_msg_start,
                                       lambda c, _: hasattr(c, 'negotiation_id') and not isinstance(c, TerminationMessage), priority=float('-inf'))

    def handle_term_msg(self, content: TerminationMessage, _: Dict[str, Any]) -> None:
        """Handle the termination message.

        :param content: the message
        :param _: meta data
        """
        self._weight_map[content.negotiation_id] += content.weight

    def on_send(self, content,
                receiver_addr: Union[str, Tuple[str, int]], *,
                receiver_id: Optional[str] = None,
                create_acl: bool = False,
                acl_metadata: Optional[Dict[str, Any]] = None,
                mqtt_kwargs: Dict[str, Any] = None):
        """Add the weight to every coalition related message

        :param content: content of the message
        :param receiver_addr: address
        :param receiver_id: id of the receiver. Defaults to None.
        :param create_acl: If you want to wrap the message in an ACL. Defaults to False.
        :param acl_metadata: ACL meta data. Defaults to None.
        :param mqtt_kwargs: Args for MQTT. Defaults to None.
        """
        if hasattr(content, 'negotiation_id') and not isinstance(content, TerminationMessage):
            if content.negotiation_id not in self._weight_map:
                self._weight_map[content.negotiation_id] = Fraction(1, 1) if self._is_controller else Fraction(0, 1)
            content.message_weight = self._weight_map[content.negotiation_id] / 2
            self._weight_map[content.negotiation_id] /= 2

    def handle_msg_start(self, content, _: Dict[str, Any]) -> None:
        """Check whether a coalition related message has been received and manipulate the internal
        weight accordingly. Setup a conditional task that checks for termination.

        :param content: the incoming neogtiation message
        :param _: the meta data
        """
        if content.negotiation_id in self._weight_map:
            self._weight_map[content.negotiation_id] += content.message_weight
        else:
            self._weight_map[content.negotiation_id] = content.message_weight

        negotiation_model = self.context.get_or_create_model(NegotiationModel)

        def _check_weight_condition() -> bool:
            """
            Function that checks whether the negotiation has 'probably' terminated.
            :return: boolean
            """
            return (not negotiation_model.by_id(content.negotiation_id).active and
                    self._weight_map[content.negotiation_id] != 0)

        if not self._is_controller and \
                (content.negotiation_id not in self._termination_check_tasks or
                 self._termination_check_tasks[content.negotiation_id].done()):
            # create a new conditional task that checks for termination
            self._termination_check_tasks[content.negotiation_id] = self.context.schedule_conditional_task(
                self._send_weight(negotiation_model, content),
                condition_func=_check_weight_condition)

    async def _send_weight(self, negotiation_model: NegotiationModel, content):
        """
        Sends the current weight to the termination controller
        :param negotiation_model: Model of the current negotiation
        :param content: The NeogotiationMessage

        """
        coalition = self.context.get_or_create_model(CoalitionModel).by_id(content.coalition_id)
        # store weight
        current_weight = self._weight_map[content.negotiation_id]
        # reset weight
        self._weight_map[content.negotiation_id] = Fraction(0, 1)
        # Send weight
        await self.context.send_message(
            content=TerminationMessage(current_weight, content.coalition_id, content.negotiation_id),
            receiver_addr=coalition.controller_agent_addr,
            receiver_id=coalition.controller_agent_id,
            acl_metadata={'sender_addr': self.context.addr, 'sender_id': self.context.aid},
            create_acl=True)
