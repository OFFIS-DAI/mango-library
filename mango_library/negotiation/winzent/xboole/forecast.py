class Forecast:

    def __init__(self, forecast_tuple=None):
        if forecast_tuple is None:
            self._forecast = []
        else:
            self._forecast = [forecast_tuple[0],
                              forecast_tuple[1]]

    @property
    def first(self):  # time span
        return self._forecast[0]

    @first.setter
    def first(self, time_span):  # time span
        self._forecast[0] = time_span

    @property
    def second(self):  # kw
        return self._forecast[1]

    @second.setter
    def second(self, kw):  # kw
        self._forecast = (self._forecast[0], kw)
