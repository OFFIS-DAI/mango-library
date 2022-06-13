
import fractions
from typing import Any, Dict
import uuid
from unittest import mock
from mango.role.core import  RoleAgent
from mango.messages.codecs import JSON
from mango.role.api import SimpleReactiveRole
import pytest
from mango_library.coalition.core import CoalitionAssignment, CoalitionModel
from mango_library.negotiation.core import Negotiation, NegotiationModel
from mango_library.negotiation.termination import NegotiationTerminationRole, TerminationMessage
from mango_library.negotiation.util import extra_serializers


def test_serialization():
    codec = JSON()
    for serializer in extra_serializers:
        codec.add_serializer(*serializer())
    my_data = TerminationMessage(coalition_id=uuid.uuid1(), negotiation_id=uuid.uuid1(), weight=fractions.Fraction(1, 2))

    encoded = codec.encode(my_data)
    decoded = codec.decode(encoded)

    assert my_data.coalition_id == decoded.coalition_id
    assert my_data.negotiation_id == decoded.negotiation_id
    assert my_data.weight == decoded.weight


class NegotMsg:
    def __init__(self, id, we):
        self.negotiation_id = id
        self.coalition_id = id
        self.message_weight = we


class NegotRole(SimpleReactiveRole):
    def __init__(self, order_check_list) -> None:
        super().__init__()
        self._order_check_list = order_check_list

    def handle_msg(self, content, _: Dict[str, Any]) -> None:
        self._order_check_list.append(1)


class NegotTermCheckRole(NegotiationTerminationRole):
    def __init__(self, order_check_list) -> None:
        super().__init__(True)
        self._order_check_list = order_check_list

    def handle_msg_start(self, content, meta: Dict[str, Any]) -> None:
        self._order_check_list.append(0)
        super().handle_msg_start(content, meta)
    def handle_msg_end(self, content, meta: Dict[str, Any]) -> None:
        self._order_check_list.append(2)
        super().handle_msg_end(content, meta)


 # test is not necessary anymore, as we don't have to subscribe two times
# @pytest.mark.asyncio
# async def test_order_handle_negotiation_msg():
#     # GIVEN
#     order_check_list = []
#     container_mock = mock.Mock()
#     controller_term = NegotTermCheckRole(order_check_list)
#     negot_role = NegotRole(order_check_list)
#     role_agent = RoleAgent(container_mock)
#     role_agent.add_role(controller_term)
#     role_agent.add_role(negot_role)
#     coalition_model = role_agent._agent_context.get_or_create_model(CoalitionModel)
#     coalition_model.add(1, CoalitionAssignment(1, [], "", 1, "", ()))
#     negot_model = role_agent._agent_context.get_or_create_model(NegotiationModel)
#     negot_model.add(1, Negotiation(None, None, True))
#
#     # WHEN
#     role_agent._agent_context.handle_msg(NegotMsg(1, 0.5), None)
#
#     # THEN
#     assert order_check_list == [0, 1, 2]

