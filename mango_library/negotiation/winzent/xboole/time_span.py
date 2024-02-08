import datetime

import intervals


class TimeSpan:
    def __init__(self, upper, lower):
        if not isinstance(upper, datetime.date):
            if isinstance(upper, str):
                upper = datetime.datetime.strptime(upper,
                                                   "%Y-%m-%dT%H:%M:%S")
                lower = datetime.datetime.strptime(lower,
                                                   "%Y-%m-%dT%H:%M:%S")
            else:
                upper = upper.datetime
                lower = lower.datetime
        self._time_span = intervals.DateTimeInterval(lower, upper)
        self._time_span.lower = lower
        self._time_span.upper = upper

    @property
    def upper(self):
        return self._time_span.upper

    @upper.setter
    def upper(self, upper):
        self._time_span.upper = upper

    @property
    def lower(self):
        return self._time_span.lower

    @lower.setter
    def lower(self, lower):
        self._time_span.lower = lower

    def length(self):
        return self._time_span.length

    def equals(self, other):
        return self._time_span.upper == other.time_span.upper \
               and self._time_span.lower == other.time_span.lower

    @property
    def time_span(self):
        return self._time_span

    def to_str(self):
        return [
            datetime.datetime.strftime(self.time_span.lower,
                                       "%Y-%m-%dT%H:%M:%S"),
            datetime.datetime.strftime(self.time_span.upper,
                                       "%Y-%m-%dT%H:%M:%S")]
