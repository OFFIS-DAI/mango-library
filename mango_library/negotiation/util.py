import numpy as np
import fractions
import uuid
import base64
from mango_library.coalition.core import CoalitionInvite, CoaltitionResponse, CoalitionAssignment, \
    CoalitionAssignmentConfirm, CoalitionBuildConfirm
from mango_library.negotiation.cohda.data_classes import ScheduleSelection, \
    SystemConfig, SolutionCandidate, WorkingMemory
from mango_library.negotiation.cohda.cohda_messages import CohdaNegotiationMessage, CohdaProposedSolutionMessage, \
    CohdaSolutionRequestMessage, CohdaFinalSolutionMessage, ConfirmCohdaSolutionMessage
from mango_library.negotiation.multiobjective_cohda.cohda_messages import MoCohdaNegotiationMessage
from mango_library.negotiation.multiobjective_cohda.data_classes import ScheduleSelections, \
    SystemConfig as SystemConfig_m, SolutionCandidate as SolutionCandidate_m, WorkingMemory as WorkingMemory_m
from mango_library.negotiation.termination import TerminationMessage, StopNegotiationMessage, \
    InformAboutTerminationMessage


def get_np_serializer():
    """Return a tuple *(type, serialize(), deserialize())* for NumPy arrays
    """
    return np.ndarray, _serialize_ndarray, _deserialize_ndarray


def _serialize_ndarray(obj: np.ndarray):
    return {
        "type": obj.dtype.str,
        "shape": obj.shape,
        "data": base64.b64encode(obj.tobytes()).decode("ascii"),
    }


def _deserialize_ndarray(obj):
    return np.frombuffer(
        base64.b64decode(obj["data"]), dtype=np.dtype(obj["type"])
    ).reshape(obj["shape"])


def get_uuid_serializer():
    return uuid.UUID, _serialize_uuid, _deserialize_uuid


def _serialize_uuid(obj: uuid.UUID):
    """"""
    return str(obj)


def _deserialize_uuid(obj):
    return uuid.UUID(obj)


def get_fraction_serializer():
    return fractions.Fraction, serialize_fraction, deserialize_fraction


def serialize_fraction(obj: fractions.Fraction):
    return str(obj)


def deserialize_fraction(obj):
    return fractions.Fraction(obj)


cohda_serializers = [
    get_np_serializer,
    get_uuid_serializer,
    CoalitionInvite.__serializer__,
    CoaltitionResponse.__serializer__,
    CoalitionAssignment.__serializer__,
    CoalitionAssignmentConfirm.__serializer__,
    CoalitionBuildConfirm.__serializer__,
    ScheduleSelection.__serializer__,
    SystemConfig.__serializer__,
    SolutionCandidate.__serializer__,
    WorkingMemory.__serializer__,
    CohdaNegotiationMessage.__serializer__,
    CohdaSolutionRequestMessage.__serializer__,
    CohdaProposedSolutionMessage.__serializer__,
    TerminationMessage.__serializer__,
    get_fraction_serializer,
    StopNegotiationMessage.__serializer__,
    InformAboutTerminationMessage.__serializer__,
    CohdaFinalSolutionMessage.__serializer__,
    ConfirmCohdaSolutionMessage.__serializer__,
]

multi_objective_serializers = [
    get_np_serializer,
    get_uuid_serializer,
    CoalitionInvite.__serializer__,
    CoaltitionResponse.__serializer__,
    CoalitionAssignment.__serializer__,
    ScheduleSelections.__serializer__,
    SystemConfig_m.__serializer__,
    SolutionCandidate_m.__serializer__,
    WorkingMemory_m.__serializer__,
    MoCohdaNegotiationMessage.__serializer__,
    TerminationMessage.__serializer__,
    get_fraction_serializer,
    StopNegotiationMessage.__serializer__,
]
