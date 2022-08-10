import uuid


class Individual:

    def __init__(self, genome, objective_values):
        self._objective_values = objective_values
        self._genome = genome
        self._id = str(uuid.uuid4())

    @property
    def objective_values(self):
        if self._objective_values is None:
            raise ValueError('Objective Values not set!')
        return self._objective_values

    @objective_values.setter
    def objective_values(self, objective_values):
        self._objective_values = objective_values

    @property
    def genome(self):
        return self._genome

    @property
    def id(self):
        return self._id
