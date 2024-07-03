import uuid

from mango_library.negotiation.winzent import xboole


class GridMessage:
    def __init__(self, rhs=None):
        if rhs is None:
            self.msg_type = None
            self.id = str(uuid.uuid4())
            self.sender = None
            self.is_answer = None
            self.answer_to = None
            self.receiver = -1

        elif isinstance(rhs, int):  # message_type
            self.msg_type = rhs
            self.id = str(uuid.uuid4())
            self.answer_to = None
            self.is_answer = False
            self.receiver = - 1
        else:
            self.msg_type = rhs.msg_type()
            self.id = rhs.id()
            self.sender = rhs.sender()
            self.receiver = rhs.receiver()
            self.is_answer = rhs.is_answer()
            self.answer_to = rhs.answerTo()
        self.value = None

    def __eq__(self, other):
        return (self.msg_type == other.msg_type
                and self.id == other.id
                and self.sender == other.sender
                and self.receiver == other.receiver)

    def clear(self):
        self.id = str(uuid.uuid4())
        self.msg_type = xboole.MessageType.Null
        self.is_answer = False
        self.answer_to = None
        self.sender = None
        self.receiver = - 1


class ForwardableGridMessage(GridMessage):
    def __init__(self, rhs):
        self.start_ttl = 1056
        GridMessage.__init__(self, rhs)
        if isinstance(rhs, int):
            self.ttl = self.start_ttl
        elif rhs is None:
            self.ttl = None
        else:
            self.ttl = rhs.ttl

    def dict(self):
        return self.__dict__


class PowerMessage(ForwardableGridMessage):
    def __init__(self, rhs=None):
        ForwardableGridMessage.__init__(self, rhs)
        if rhs is None:
            self.time_span = None
            self.value = None
            self.answerUntil = None
        elif not isinstance(rhs, int):
            self.time_span = rhs.time_span
            self._value = rhs.value
            self.answerUntil = rhs.answerUntil
        else:  # rhs = type
            self.time_span = None
            self.value = None

    def dict(self):
        if self.time_span is not None:
            self.time_span = [self.time_span.upper.strftime(
                '%Y-%m-%dT%H:%M:%S+00:00'), self.time_span.lower.strftime(
                '%Y-%m-%dT%H:%M:%S+00:00')]
        return self.__dict__
