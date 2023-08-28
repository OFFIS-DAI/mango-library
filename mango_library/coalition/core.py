"""Module, which provides some roles and models to support coalition creation with mango.

There are mainly two roles involved in this:
* :class:`CoalitionInitiatorRole`: Initiates a coalition, therfore sends the invites, receives the
                                   responses and informs all participants, which accepted, about the
                                   coalition and its topology
* :class:`CoalitionParticipantRole`: Participates in a coalition, the main responsibility is
                                     answering the coalition invite and storing the
                                     coalition-assignment when a coalition creation was successful

The messages defined in this module:
* :class:`CoalitionAssignment`: sent by the initiator when the accepted coalition really got created
* :class:`CoalitionInvite`: sent by the initiator to start the coalition creation
* :class:`CoaltitionResponse`: sent by any participant as answer to an CoalitionInvite

The role models defined in this module:
* :class:`CoalitionModel`: contains all information for all coalitions an agent participates in
"""
import asyncio
import logging
import random
import uuid
from typing import Dict, Any, List, Tuple, Union
from uuid import UUID

from mango.messages.codecs import json_serializable
from mango import Role, RoleContext
from mango.util.scheduling import InstantScheduledTask

logger = logging.getLogger(__name__)

ContainerAddress = Union[str, Tuple[str, int]]
ParticipantKey = Tuple[
    str, ContainerAddress, str
]  # part_id as str, ContainerAddress, AgentID


@json_serializable
class CoalitionAssignment:
    """Message/Model for assigning a participant to an already accepted coalition. In this
    assignment all relevant information about the coalition are contained,
    f.e. participant id, neighbors, ... .
    """

    def __init__(
        self,
        coalition_id: UUID,
        neighbors: List[ParticipantKey],
        topic: str,
        part_id: str,
        controller_agent_id: str,
        controller_agent_addr,
    ):
        self._coalition_id = coalition_id
        self._neighbors = neighbors
        self._topic = topic
        self._part_id = part_id
        self._controller_agent_id = controller_agent_id
        self._controller_agent_addr = controller_agent_addr

    @property
    def coalition_id(self) -> UUID:
        """Id of the colaition (unique)

        :return: id of the coalition as UUID
        """
        return self._coalition_id

    @property
    def neighbors(self) -> List[ParticipantKey]:
        """Neighbors of the participant.

        :return: List of the participant, a participant is modelled as
                                        tuple (part_id, address, aid)
        """
        return self._neighbors

    @property
    def topic(self):
        """The topic of the coalition, f.e. COHDA

        :return: the topic
        """
        return self._topic

    @property
    def part_id(self) -> str:
        """The id of the participant

        :return: id
        """
        return self._part_id

    @property
    def controller_agent_id(self):
        """Id of the controller agent

        :return: agent_id
        """
        return self._controller_agent_id

    @property
    def controller_agent_addr(self) -> ContainerAddress:
        """Adress of the controller agent

        :return: adress as tuple
        """
        return self._controller_agent_addr


class CoalitionModel:
    """Role-model for coalitions"""

    def __init__(self) -> None:
        self._assignments = {}

    @property
    def assignments(self) -> Dict[UUID, CoalitionAssignment]:
        """Dict of assignments coalition_id -> assignment

        :return: the dict of assignments
        """
        return self._assignments

    def add(self, coalition_id: UUID, assignment: CoalitionAssignment):
        """Add a new assignment

        :param coalition_id: uuid of the coalition you want to add
            assignment (CoalitionAssignment): new assignment
        """
        self._assignments[coalition_id] = assignment

    def by_id(self, coalition_id: UUID) -> CoalitionAssignment:
        """Get an assignment by coalition id

        :param coalition_id: the coalition id

        :return: the assignment
        """
        return self._assignments[coalition_id]

    def exists(self, coalition_id: UUID):
        """Checks whether there exists an assignment for the given coalition id

        :param coalition_id: the coalition id

        :return: the assignment
        """
        return coalition_id in self._assignments


@json_serializable
class CoalitionInvite:
    """Message for inviting an agent to a coalition."""

    def __init__(self, coalition_id: UUID, topic: str, details=None):
        self._coalition_id = coalition_id
        self._topic = topic
        self._details = details

    @property
    def coalition_id(self) -> UUID:
        """Return id of the coalition

        :return: id of the coalition
        """
        return self._coalition_id

    @property
    def topic(self) -> str:
        """Return the topic of the coalition

        :return: the topic
        """
        return self._topic

    @property
    def details(self):
        """Return details additional to the topic

        :return: additional details
        """
        return self._details


@json_serializable
class CoaltitionResponse:
    """Message for responding to a coalition invite."""

    def __init__(self, accept: bool):
        self._accept = accept

    @property
    def accept(self) -> bool:
        """
        Flag whether the coalition is accpeted
        :return: true if accepted, false otherwise
        """
        return self._accept


@json_serializable
class CoalitionAssignmentConfirm:
    def __init__(self, coalition_id: UUID):
        self._coalition_id = coalition_id

    @property
    def coalition_id(self):
        return self._coalition_id


@json_serializable
class CoalitionBuildConfirm:
    def __init__(self, coalition_id: UUID):
        self._coalition_id = coalition_id

    @property
    def coalition_id(self):
        return self._coalition_id


def clique_creator(
    participants: List[ParticipantKey],
) -> Dict[ParticipantKey, List[ParticipantKey]]:
    """
    Create a clique topology

    :param participants: the list of all participants

    :return: a map, mapping every participant to a list of their neighbors
    """
    part_to_neighbors = {}
    for part in participants:
        part_to_neighbors[part] = list(
            filter(lambda p, c_p=part: p != c_p, participants)
        )
    return part_to_neighbors


def small_world_creator(
    participants: List[ParticipantKey], k=2, w=0.0
) -> Dict[ParticipantKey, List[ParticipantKey]]:
    """
    Builds a small world ring topology with neighbors in a distance of k and with random neighbors with the
    probability w
    :param participants:
    :param k: maximum distance of connections in the ring
    :param w: probability of random connections
    """
    neighborhood: Dict[ParticipantKey, List[ParticipantKey]] = {}
    n_particpants = len(participants)

    for agent in participants:
        neighborhood[agent] = []

    # create the ring
    for index, participant in enumerate(participants):
        for distance in range(1, k + 1):
            left_neighbor = participants[
                (index - distance) % n_particpants
            ]  # left neighbor
            right_neighbor = participants[
                (index + distance) % n_particpants
            ]  # right neighbor

            if (
                participant != left_neighbor
                and left_neighbor not in neighborhood[participant]
            ):
                neighborhood[participant].append(left_neighbor)
            if (
                participant != right_neighbor
                and right_neighbor not in neighborhood[participant]
            ):
                neighborhood[participant].append(right_neighbor)

    # create random connections with probability w
    for index, participant in enumerate(participants):
        if random.random() < w:
            random_agent = random.choice(participants)
            if (
                random_agent != participant
                and random_agent not in neighborhood[participant]
            ):
                neighborhood[participant].append(random_agent)
                neighborhood[random_agent].append(participant)
    return neighborhood


class CoalitionInitiatorRole(Role):
    """Role responsible for initiating a coalition. Considered as proactive role.

    The role will invite all given participants and add them to coalition if they accept the invite.
    """

    def __init__(
        self,
        participants: List[Tuple[ContainerAddress, str]],
        topic: str,
        details: str,
        topology_creator=small_world_creator,
        topology_creator_kwargs: dict = None,
    ):
        super().__init__()
        self._participants = participants
        self._topic = topic
        self._details = details
        self._topology_creator = topology_creator
        self._topology_creator_kwargs = (
            topology_creator_kwargs if topology_creator_kwargs else {}
        )
        self._part_to_state = {}
        self._assignments_sent = False
        self._coal_id = None
        self._assignments_confirmed = {}

    def setup(self):

        # subscriptions
        self.context.subscribe_message(
            self,
            self.handle_coalition_response_msg,
            lambda c, m: isinstance(c, CoaltitionResponse),
        )

        # coalition assignment confirms
        self.context.subscribe_message(
            self,
            self.handle_assignment_confirms,
            lambda c, m: isinstance(c, CoalitionAssignmentConfirm),
        )

        # tasks
        self.context.schedule_task(
            InstantScheduledTask(self.send_invitiations(self.context))
        )

    async def send_invitiations(self, agent_context: RoleContext):
        """Send invitiations to all wanted participant for the coalition

        :param agent_context: the context
        """
        self._coal_id = uuid.uuid1()

        for participant in self._participants:
            await agent_context.send_acl_message(
                content=CoalitionInvite(self._coal_id, self._topic),
                receiver_addr=participant[0],
                receiver_id=participant[1],
                acl_metadata={
                    "sender_addr": agent_context.addr,
                    "sender_id": agent_context.aid,
                },
            )

    def handle_coalition_response_msg(
        self, content: CoaltitionResponse, meta: Dict[str, Any]
    ) -> None:
        """Handle the responses to the invites.
        :param content: the invite response
        :param meta: meta data
        """

        sender_addr = meta["sender_addr"]
        sender_id = meta["sender_id"]
        if isinstance(sender_addr, list):
            sender_addr = tuple(sender_addr)

        sender_addr: ContainerAddress

        self._part_to_state[(sender_addr, sender_id)] = content.accept

        if (
            len(self._part_to_state) == len(self._participants)
            and not self._assignments_sent
        ):
            self.context.schedule_instant_task(self._send_assignments(self.context))
            self._assignments_sent = True

    async def _send_assignments(self, agent_context: RoleContext):
        part_id = 0
        accepted_participants = []
        for agent_addr, agent_id in self._participants:
            if (agent_addr, agent_id) in self._part_to_state and self._part_to_state[
                (agent_addr, agent_id)
            ]:
                part_id += 1
                accepted_participants.append((str(part_id), agent_addr, agent_id))

        part_to_neighbors = self._topology_creator(
            accepted_participants, **self._topology_creator_kwargs
        )
        for part in accepted_participants:
            agent_context.schedule_instant_acl_message(
                content=CoalitionAssignment(
                    self._coal_id,
                    part_to_neighbors[part],
                    self._topic,
                    part[0],
                    agent_context.aid,
                    agent_context.addr,
                ),
                receiver_addr=part[1],
                receiver_id=part[2],
                acl_metadata={
                    "sender_addr": agent_context.addr,
                    "sender_id": agent_context.aid,
                },
            )
            self._assignments_confirmed[(part[1], part[2])] = asyncio.Future()

        self.context.schedule_conditional_task(
            self._send_coalition_build_confirms(agent_context, accepted_participants),
            self._all_assignment_confirms_received,
        )

    def _send_coalition_build_confirms(self, agent_context, accepted_participants):
        for part in accepted_participants:
            agent_context.schedule_instant_acl_message(
                content=CoalitionBuildConfirm(coalition_id=self._coal_id),
                receiver_addr=part[1],
                receiver_id=part[2],
                acl_metadata={
                    "sender_addr": agent_context.addr,
                    "sender_id": agent_context.aid,
                },
            )

    def _all_assignment_confirms_received(self):
        return all([fut.result() for fut in self._assignments_confirmed.values()])

    def handle_assignment_confirms(
        self, content: CoalitionAssignmentConfirm, meta: Dict[str, Any]
    ) -> None:
        """Handle the responses to the invites.
        :param content: the invite response
        :param meta: meta data
        """
        sender_addr = meta["sender_addr"]
        sender_id = meta["sender_id"]

        if isinstance(sender_addr, list):
            sender_addr = tuple(sender_addr)

        assert self._coal_id == content.coalition_id
        sender_identifier = (sender_addr, sender_id)

        if sender_identifier in self._assignments_confirmed:
            self._assignments_confirmed[sender_identifier].set_result(True)
        else:
            raise ValueError(
                f"Received confirmation about assignment from an agent which is not part of coalition. "
                f"AgentId: {sender_id}"
            )


class CoalitionParticipantRole(Role):
    """Role responsible for participating in a coalition. Handles the messages
    :class:`CoalitionInvite` and :class:`CoalitionAssignment`.

    When a valid assignment was received the model :class:`CoalitionModel` will be created
    as central role model.
    """

    def __init__(self, join_decider=lambda _: True):
        super().__init__()
        self._join_decider = join_decider

    def setup(self) -> None:
        # subscriptions
        self.context.subscribe_message(
            self, self.handle_invite, lambda c, m: isinstance(c, CoalitionInvite)
        )
        self.context.subscribe_message(
            self,
            self.handle_assignment,
            lambda c, m: isinstance(c, CoalitionAssignment),
        )

    def handle_invite(self, content: CoalitionInvite, meta: Dict[str, Any]) -> None:
        """Handle invite messages, responding with a CoalitionResponse.

        :param content: the invite
        :param meta: meta data
        """
        self.context.schedule_instant_acl_message(
            content=CoaltitionResponse(self._join_decider(content)),
            receiver_addr=meta["sender_addr"],
            receiver_id=meta["sender_id"],
            acl_metadata={
                "sender_addr": self.context.addr,
                "sender_id": self.context.aid,
            },
        )

    def handle_assignment(
        self, content: CoalitionAssignment, meta: Dict[str, Any]
    ) -> None:
        """Handle an incoming assignment to a coalition. Store the information in a CoalitionModel.

        :param content: the assignment
        :param meta: the meta data
        """
        assignment = self.context.get_or_create_model(CoalitionModel)
        assignment.add(content.coalition_id, content)
        self.context.update(assignment)

        self.context.schedule_instant_acl_message(
            content=CoalitionAssignmentConfirm(coalition_id=content.coalition_id),
            receiver_addr=meta["sender_addr"],
            receiver_id=meta["sender_id"],
            acl_metadata={
                "sender_addr": self.context.addr,
                "sender_id": self.context.aid,
            },
        )
