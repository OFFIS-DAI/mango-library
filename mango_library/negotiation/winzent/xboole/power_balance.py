# class Module saves Governor
import collections


class PowerBalance:
    # Mapping between time und power values
    # stores Ledger
    def __init__(self):
        # key = time_span, value = requirement
        self._ledger = collections.OrderedDict()
        self._max = 0
        # dictionary with timestamps of disequilibrium, if request for
        # demand or supply has already been sent, value is True at key
        self._imbalance_timestamps = {}

    def __iter__(self):
        self.n = -1
        self.len_req = len(list(self._ledger.items())[0][
                               1])  # len from current requirement list
        self.pos_in_ledger = 0  # current position in ledger key
        self.pos_in_req = -1  # current position in requirements list
        return self

    def __next__(self):
        if self.n <= self._max:
            self.n += 1
            self.pos_in_req += 1
            # all requirements for one key done
            if self.pos_in_req == self.len_req:
                self.pos_in_ledger += 1
                self.pos_in_req = 0
                if len(self._ledger) == self.pos_in_ledger:
                    raise StopIteration
                else:
                    self.len_req = len(
                        list(self._ledger.items())[self.pos_in_ledger][1])
            first = list(self._ledger.items())[self.pos_in_ledger][0]
            sec = list(self._ledger.items())[self.pos_in_ledger][1][
                self.pos_in_req]
            return first, sec
        else:
            raise StopIteration

    def __getitem__(self, key):
        return list(self._ledger.items())[0][1]

    def __len__(self):
        return len(self._ledger)

    def add(self, requirement):
        for k, v in self._ledger.items():
            if k == requirement.time_span[0]:
                self._ledger[k].append(requirement)
                self._max += 1
                return
        self._ledger[requirement.time_span[0]] = []
        self._ledger[requirement.time_span[0]].append(requirement)
        self._max += 1
        return self

    def clear(self):
        self._ledger.clear()
        self._max = 0

    def begin(self):
        """
        returns first position of ledger
        """
        if len(list(self._ledger.items())[0][1]) > 0:
            return list(self._ledger.items())[0][0], \
                   list(self._ledger.items())[0][1][0]
        return list(self._ledger.items())[0]

    def end(self):
        """
        :return: index of last position of ledger
        """
        return list(self._ledger.items())[len(self._ledger) - 1]

    def delete_first_req(self):
        if len(list(self._ledger.items())[0][1]) > 0:
            del (list(self._ledger.items())[0][0],
                 list(self._ledger.items())[0][1][0])
        del list(self._ledger.items())[0]

    def is_unbalanced(self, time):
        return time[0] in self._imbalance_timestamps.keys()

    def help_requested(self, time, requirement):
        self._imbalance_timestamps[time[0]] = requirement

    def withdrawn(self, time):
        del self._imbalance_timestamps[time]

    def disequilibrium(self, time):
        return self._imbalance_timestamps[time]

    def empty(self):
        return len(self._ledger) == 0

    @property
    def ledger(self):
        return self._ledger
