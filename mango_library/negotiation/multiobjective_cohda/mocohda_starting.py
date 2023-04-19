import logging
import uuid
from fractions import Fraction

from mango import Role

from mango_library.coalition.core import CoalitionModel
from mango_library.negotiation.multiobjective_cohda.cohda_messages import (
    MoCohdaNegotiationMessage,
    MoCohdaNegotiationStartMessage,
)
from mango_library.negotiation.multiobjective_cohda.data_classes import (
    WorkingMemory,
    SystemConfig,
    SolutionCandidate,
)

logger = logging.getLogger(__name__)


class MoCohdaNegotiationInteractiveStarterRole(Role):
    """
    Role for starting a MOCOHDA negotiation with MoCohdaNegotiationStartMessage
    """

    def __init__(
        self,
        target_params,
        num_solution_points,
        coalition_model_matcher=None,
        coalition_uuid=None,
        send_weight=True,
    ) -> None:
        """

        :param target_params: Parameter that are necessary for the agents to calculate the performance.
        Could be e.g. the target schedule.
        :param coalition_model_matcher:
        :param coalition_uuid:
        """
        super().__init__()
        self._target_params = target_params
        self._send_weight = send_weight
        self._num_solution_points = num_solution_points

        if coalition_uuid is not None:
            # if id is provided create matcher matching this id when the coalition is looked up
            self._coalition_model_matcher = (
                lambda coal: coal.coalition_id == coalition_uuid
            )
        elif coalition_model_matcher is not None:
            # when a matcher itself is provided use this if no uuid has been provided
            self._coalition_model_matcher = coalition_model_matcher
        else:
            # when nothing has been provided just match the first coalition
            self._coalition_model_matcher = lambda _: True

    def setup(self):
        super().setup()

        self.context.subscribe_message(
            self,
            self.handle_mocohda_start,
            lambda c, m: isinstance(c, MoCohdaNegotiationStartMessage),
        )

    def handle_mocohda_start(self, msg, m):
        self.context.schedule_conditional_task(self.start(msg), self.is_startable)

    def is_startable(self):
        coalition_model = self.context.get_or_create_model(CoalitionModel)

        # check if there is a coalition that can be used
        for assignment in coalition_model.assignments.values():
            if self._coalition_model_matcher(assignment):
                return True
        return False

    def _look_up_assignment(self, all_assignments, coalition_id=None, start_msg=None):
        if coalition_id is not None:
            for assignment in all_assignments:
                if start_msg is not None:
                    if assignment.coalition_id == start_msg.coalition_id:
                        break
                if assignment.coalition_id == coalition_id:
                    return assignment
        # default to the first one
        return list(all_assignments)[0]

    async def start(self, start_msg: MoCohdaNegotiationStartMessage = None):
        """Start a negotiation. Send all neighbors a starting negotiation message."""

        coalition_model = self.context.get_or_create_model(CoalitionModel)

        # Find any matching coalition assignment
        matched_assignment = self._look_up_assignment(
            coalition_model.assignments.values(), start_msg
        )

        # create a new negotiation id
        negotiation_uuid = uuid.uuid1()

        # update target_params
        if start_msg is not None:
            if start_msg.target_params is not None:
                self._target_params = start_msg.target_params
            self._send_weight = start_msg.send_weight

        empty_wm = WorkingMemory(
            target_params=self._target_params,
            system_config=SystemConfig(
                {}, num_solution_points=self._num_solution_points
            ),
            solution_candidate=SolutionCandidate(
                agent_id=matched_assignment.part_id,
                schedules={},
                num_solution_points=self._num_solution_points,
                perf=None,
                hypervolume=None,
            ),
        )

        # send message to all neighbors
        for neighbor in matched_assignment.neighbors:
            neg_msg = MoCohdaNegotiationMessage(
                working_memory=empty_wm,
                negotiation_id=negotiation_uuid,
                coalition_id=matched_assignment.coalition_id,
            )
            if self._send_weight:
                # relevant for termination detection
                neg_msg.message_weight = Fraction(1, len(matched_assignment.neighbors))
            self.context.schedule_instant_acl_message(
                content=neg_msg,
                receiver_addr=neighbor[1],
                receiver_id=neighbor[2],
                acl_metadata={
                    "sender_addr": self.context.addr,
                    "sender_id": self.context.aid,
                },
            )


class MoCohdaNegotiationDirectStarterRole(Role):
    """
    Convenience role for starting a COHDA negotiation directly from setup
    """

    def __init__(
        self,
        target_params,
        num_solution_points,
        coalition_model_matcher=None,
        coalition_uuid=None,
        send_weight=True,
    ) -> None:
        """

        :param target_params: Parameter that are necessary for the agents to calculate the performance.
        Could be e.g. the target schedule.
        :param coalition_model_matcher:
        :param coalition_uuid:
        """
        super().__init__()
        self._target_params = target_params
        self._send_weight = send_weight
        self._num_solution_points = num_solution_points

        if coalition_uuid is not None:
            # if id is provided create matcher matching this id when the coalition is looked up
            self._coalition_model_matcher = (
                lambda coal: coal.coalition_id == coalition_uuid
            )
        elif coalition_model_matcher is not None:
            # when a matcher itself is provided use this if no uuid has been provided
            self._coalition_model_matcher = coalition_model_matcher
        else:
            # when nothing has been provided just match the first coalition
            self._coalition_model_matcher = lambda _: True

    def setup(self):
        super().setup()

        self.context.schedule_conditional_task(self.start(), self.is_startable)

    def is_startable(self):
        coalition_model = self.context.get_or_create_model(CoalitionModel)

        # check if there is a coalition that can be used
        for assignment in coalition_model.assignments.values():
            if self._coalition_model_matcher(assignment):
                return True
        return False

    def _look_up_assignment(self, all_assignments, coalition_id=None):
        if coalition_id is not None:
            for assignment in all_assignments:
                if assignment.coalition_id == coalition_id:
                    return assignment
        # default to the first one
        return list(all_assignments)[0]

    async def start(self):
        """Start a negotiation. Send all neighbors a starting negotiation message."""

        coalition_model = self.context.get_or_create_model(CoalitionModel)

        # Find any matching coalition assignment
        matched_assignment = self._look_up_assignment(
            coalition_model.assignments.values()
        )

        # create a new negotiation id
        negotiation_uuid = uuid.uuid1()

        empty_wm = WorkingMemory(
            target_params=self._target_params,
            system_config=SystemConfig(
                {}, num_solution_points=self._num_solution_points
            ),
            solution_candidate=SolutionCandidate(
                agent_id=matched_assignment.part_id,
                schedules={},
                num_solution_points=self._num_solution_points,
                perf=None,
                hypervolume=None,
            ),
        )

        # send message to all neighbors
        for neighbor in matched_assignment.neighbors:
            neg_msg = MoCohdaNegotiationMessage(
                working_memory=empty_wm,
                negotiation_id=negotiation_uuid,
                coalition_id=matched_assignment.coalition_id,
            )
            if self._send_weight:
                # relevant for termination detection
                neg_msg.message_weight = Fraction(1, len(matched_assignment.neighbors))
            self.context.schedule_instant_acl_message(
                content=neg_msg,
                receiver_addr=neighbor[1],
                receiver_id=neighbor[2],
                acl_metadata={
                    "sender_addr": self.context.addr,
                    "sender_id": self.context.aid,
                },
            )
