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
import random
from uuid import UUID
import asyncio
import uuid
from typing import Dict, Any, List, Tuple, Union

from mango.role.api import ProactiveRole, Role, RoleContext
from mango.util.scheduling import InstantScheduledTask
from mango.messages.codecs import json_serializable

NeighborhoodAddress = Tuple[str, Union[str, Tuple[str, int]], str]


@json_serializable
class AgentAddress:
    """
    The address of an agent
    """
    def __init__(self, host, port, aid, agent_type=None):
        self.host = host
        self.port = port
        self.aid = aid
        if agent_type is None:
            agent_type = ""
        self.agent_type = agent_type

@json_serializable
class CoalitionAssignment:
    """Message/Model for assigning a participant to an already accepted coalition. In this
    assignment all relevant information about the coalition are contained,
    f.e. participant id, neighbors, ... .
    """

    def __init__(self, coalition_id: UUID, neighbors: List[Tuple[str, Union[str, Tuple[str, int]], str]],
                 topic: str,
                 part_id: str,
                 controller_agent_id: str,
                 controller_agent_addr):
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
    def neighbors(self) -> List[Tuple[str, Union[str, Tuple[str, int]], str]]:
        """Neighbors of the participant.

        :return: List of the participant, a participant is modelled as
                                        tuple (part_id, adress, aid)
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
    def controller_agent_addr(self) -> Union[str, Tuple[str, int]]:
        """Adress of the controller agent

        :return: adress as tuple
        """
        return self._controller_agent_addr


class CoalitionModel:
    """Role-model for coalitions
    """

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
    """Message for inviting an agent to a coalition.
    """

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
    """Message for responding to a coalition invite.
    """

    def __init__(self, accept: bool):
        self._accept = accept

    @property
    def accept(self) -> bool:
        """""Flag whether the coalition is accepted

        :return: true if accepted, false otherwise
        """""
        return self._accept


def small_world_neighborhood_creator(participants: List[NeighborhoodAddress], k=1, w=0.0) -> \
        Dict[NeighborhoodAddress, List[NeighborhoodAddress]]:
    """
    Creates a small world topology
    :param participants:
    :param k: connection depth in the ring
    :param w: probability of random connections
    :return:
    """
    # sort neighbors for testing
    unit_agents = sorted(participants, key=lambda x: (x.host, x.port, x.aid))

    neighborhood: Dict[NeighborhoodAddress, List[NeighborhoodAddress]] = {}

    # empty neighborhood in the beginning
    for agent in unit_agents:
        neighborhood[agent] = []

    # create the ring
    for index, agent in enumerate(unit_agents):
        for distance in range(1, k + 1):
            neighborhood[agent].append(unit_agents[(index - distance)])
            neighborhood[unit_agents[(index - distance)]].append(agent)

    # create random connections
    for index, agent in enumerate(unit_agents):
        if random.random() < w:
            random_agent = random.choice(unit_agents)
            if (
                    random_agent not in neighborhood[agent]
                    and random_agent != agent
            ):
                neighborhood[agent].append(random_agent)
                neighborhood[random_agent].append(agent)
    return neighborhood


def clique_creator(participants: List[NeighborhoodAddress], **kwargs) -> \
        Dict[NeighborhoodAddress, List[NeighborhoodAddress]]:
    """Create a clique topology

    :param o:
    :param participants: the list of all participants

    :return: a map, mapping every participant to a list of their neighbors
    """
    part_to_neighbors = {}
    for part in participants:
        part_to_neighbors[part] = list(
            filter(lambda p, c_p=part: p != c_p, participants))
    return part_to_neighbors


class CoalitionInitiatorRole(ProactiveRole):
    """Role responsible for initiating a coalition. Considered as proactive role.

    The role will invite all given participants and add them to coalition if they accept the invite.
    """

    def __init__(self, participants: List, topic: str, details: str,
                 topology_creator=clique_creator, topology_creator_kwargs=None,
                 neighborhood_size: int = None):
        super().__init__()

        if topology_creator_kwargs is None:
            topology_creator_kwargs = {}
        self._participants_ready = asyncio.Future()
        if participants is None:
            participants = []
        if neighborhood_size is None:
            neighborhood_size = len(participants)

        # backwards compatibility:
        #   only participant and no subscription -> participants ready
        #   subscription -> wait for subscriptions until neighborhood size is achieved
        if neighborhood_size == len(participants):
            self._participants_ready.set_result(True)

        self._participants = participants
        self._topic = topic
        self._details = details
        self._topology_creator = topology_creator
        self._topology_creator_kwargs = topology_creator_kwargs
        self._neighborhood_size = neighborhood_size
        self._part_to_state = {}
        self._assignments_sent = False
        self._coal_id = None

    def setup(self):

        # subscriptions
        self.context.subscribe_message(self, self.handle_coalition_response,
                                       lambda c, m: isinstance(c, CoaltitionResponse))
        self.context.subscribe_message(self, self.handle_unit_agent_subscription,
                                       lambda content, meta: isinstance(content, AgentAddress))

        # tasks
        self.context.schedule_task(InstantScheduledTask(
            self.send_invitations(self.context)))

    def handle_unit_agent_subscription(self, content: AgentAddress, meta):
        """
        Adds a new agent to the list of neighbors
        :param meta:
        :param content:
        """
        agent = ((content.host, content.port), content.aid)
        self._participants.append(agent)

        if len(self._participants) == self._neighborhood_size:
            self._participants_ready.set_result(True)
            self.context.data.neighborhood_confirmed.set_result(True)

    async def send_invitations(self, agent_context: RoleContext):
        """Send invitiations to all wanted participant for the coalition

        :param agent_context: the context
        """
        self._coal_id = uuid.uuid1()

        await self._participants_ready

        for participant in self._participants:
            await agent_context.send_message(
                content=CoalitionInvite(self._coal_id, self._topic),
                receiver_addr=participant[0],
                receiver_id=participant[1],
                acl_metadata={'sender_addr': agent_context.addr,
                              'sender_id': agent_context.aid},
                create_acl=True)

    def handle_coalition_response(self, content: CoaltitionResponse, meta: Dict[str, Any]) -> None:
        """Handle the responses to the invites.
        :param content: the invite response
        :param meta: meta data
        """
        self._part_to_state[(meta['sender_addr'][0], meta['sender_addr'][1],
                             meta['sender_id'])] = content.accept

        if len(self._part_to_state) == len(self._participants) and not self._assignments_sent:
            self._send_assignments(self.context)
            self._assignments_sent = True

    def _send_assignments(self, agent_context: RoleContext):
        part_id = 0
        accepted_participants = []
        for part in self._participants:
            part_key = part[0][0], part[0][1], part[1]
            if part_key in self._part_to_state and self._part_to_state[part_key]:
                part_id += 1
                accepted_participants.append((str(part_id), (part_key[0], part_key[1]), part_key[2]))

        part_to_neighbors = self._topology_creator(accepted_participants, **self._topology_creator_kwargs)
        for part in accepted_participants:
            asyncio.create_task(agent_context.send_message(
                content=CoalitionAssignment(self._coal_id, part_to_neighbors[part],
                                            self._topic, part[0],
                                            agent_context.aid, agent_context.addr),
                receiver_addr=part[1], receiver_id=part[2],
                acl_metadata={'sender_addr': agent_context.addr,
                              'sender_id': agent_context.aid},
                create_acl=True))


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
        self.context.subscribe_message(self, self.handle_invite,
                                       lambda c, m: isinstance(c, CoalitionInvite))
        self.context.subscribe_message(self, self.handle_assignment,
                                       lambda c, m: isinstance(c, CoalitionAssignment))

    def handle_invite(self, content: CoalitionInvite, meta: Dict[str, Any]) -> None:
        """Handle invite messages, responding with a CoalitionResponse.

        :param content: the invite
        :param meta: meta data
        """
        print("Got invite")
        asyncio.create_task(self.context.send_message(
            content=CoaltitionResponse(self._join_decider(content)),
            receiver_addr=meta['sender_addr'], receiver_id=meta['sender_id'],
            acl_metadata={'sender_addr': self.context.addr,
                          'sender_id': self.context.aid},
            create_acl=True))

    def handle_assignment(self, content: CoalitionAssignment, _: Dict[str, Any]) -> None:
        """Handle an incoming assignment to a coalition. Store the information in a CoalitionModel.

            :param content: the assignment
            :param _: the meta data
        """
        assignment = self.context.get_or_create_model(CoalitionModel)
        assignment.add(content.coalition_id, content)
        self.context.update(assignment)
