import numpy as np
import uuid
import fractions
from mango.messages.codecs import JSON
from mango_library.negotiation.util import multi_objective_serializers
from mango_library.negotiation.multiobjective_cohda.data_classes import SolutionCandidate, ScheduleSelections, \
    SystemConfig, WorkingMemory
from mango_library.negotiation.multiobjective_cohda.multiobjective_cohda import CohdaMessage
from mango_library.negotiation.core import NegotiationMessage


def test_serialize():
    codec = JSON()
    for serializer in multi_objective_serializers:
        codec.add_serializer(*serializer())

    schedules_1 = np.array([[1, 2, 3], [2, 3, 4]], np.int32)
    schedules_2 = np.array([[4, 2, 1], [4, 5, 4]], np.int32)
    candidate = SolutionCandidate(agent_id='1', schedules={'1': schedules_1, '2': schedules_2}, num_solution_points=2,
                                  perf=None, hypervolume=None)
    encoded_candidate = codec.encode(candidate)
    assert codec.decode(encoded_candidate) == candidate

    selections_1 = ScheduleSelections(schedules=np.array([[1, 2, 3], [4, 5, 6]]), counter=42)
    selections_2 = ScheduleSelections(schedules=np.array([[4, 2, 1], [4, 5, 6]]), counter=4)
    encoded_selection = (codec.encode(selections_1), codec.encode(selections_2))
    assert codec.decode(encoded_selection[0]) == selections_1
    assert codec.decode(encoded_selection[1]) == selections_2

    sysconf = SystemConfig(schedule_choices={'1': selections_1, '2': selections_2}, num_solution_points=2)
    encoded = codec.encode(sysconf)
    decoded = codec.decode(encoded)
    assert sysconf == decoded

    working_memory = WorkingMemory(target_params=[[1, 2, 3], [1, 1, 1]],
                                   system_config=sysconf,
                                   solution_candidate=candidate)
    encoded = codec.encode(working_memory)
    decoded = codec.decode(encoded)
    assert working_memory == decoded

    msg = CohdaMessage(working_memory=working_memory)
    encoded = codec.encode(msg)
    decoded = codec.decode(encoded)
    assert msg.working_memory == decoded.working_memory

    negotiation_msg = NegotiationMessage(coalition_id=uuid.uuid1(), negotiation_id=uuid.uuid4(), message=msg)
    negotiation_msg.message_weight = fractions.Fraction(2, 5)
    encoded = codec.encode(negotiation_msg)
    decoded = codec.decode(encoded)
    assert negotiation_msg.coalition_id == decoded.coalition_id
    assert negotiation_msg.negotiation_id == decoded.negotiation_id
    assert negotiation_msg.message.working_memory == decoded.message.working_memory
    assert negotiation_msg.message_weight == decoded.message_weight