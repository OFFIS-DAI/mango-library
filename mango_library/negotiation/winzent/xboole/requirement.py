import uuid

from mango_library.negotiation.winzent import xboole
from mango_library.negotiation.winzent.winzent_message_pb2 import WinzentMessage


class Requirement:
    def __init__(self, other, owner='', ttl=2):
        if isinstance(other, xboole.Forecast):
            self._message = self.build_message(other, owner, ttl)
            self._forecast = other
        elif isinstance(other, Requirement):
            self._message = other.message
            self._forecast = other.forecast
            self._from_target = other._from_target
        else:  # PowerMessage
            self._message = other
            self._forecast = self.build_forecast(other)
        self._from_target = False
        self._from_self = True

    @property
    def message(self):
        return self._message

    @message.setter
    def message(self, message):
        self._from_self = False
        self._message = message
        self._forecast = self.build_forecast(message)

    @property
    def forecast(self):
        return self._forecast

    @property
    def time_span(self):
        return self._forecast.first

    @property
    def power(self):
        return self._forecast.second

    @property
    def from_target(self):
        return self._from_target

    @from_target.setter
    def from_target(self, from_target):
        self._from_target = from_target

    @property
    def from_self(self):
        return self._from_self

    @from_self.setter
    def from_self(self, from_self):
        self._from_self = from_self

    def __eq__(self, other):
        equals = 1
        equals &= int(self._message == other.message)
        return equals

    @staticmethod
    def build_message(forecast, owner, ttl):
        if forecast.second[0] < 0:
            msg_type = xboole.MessageType.DemandNotification
        else:
            msg_type = xboole.MessageType.OfferNotification
        return WinzentMessage(msg_type=msg_type,
                              time_span=forecast.first,
                              value=forecast.second,
                              sender=owner, id=str(uuid.uuid4()), ttl=ttl)

    @staticmethod
    def build_forecast(message):
        if message.msg_type == xboole.MessageType.DemandNotification:
            val = [-val for val in message.value]
            return xboole.Forecast(
                (message.time_span, val))
        else:
            return xboole.Forecast((message.time_span, message.value))
