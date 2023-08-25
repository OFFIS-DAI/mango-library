from copy import deepcopy

from mango_library.negotiation.winzent import xboole


class MessageJournal:

    def __init__(self):
        self._entries = {}

    def get_message_for_id(self, id):
        return self._entries[id]

    def add(self, message):
        self._entries[message.id] = message

    def entries(self):
        return self._entries

    def is_empty(self):
        if len(self._entries) == 0:
            return True
        return False

    def contains_message(self, id):
        if id in self._entries.keys():
            return True
        return False

    def remove_message(self, id):
        if id in self._entries.keys():
            del self._entries[id]

    def clear(self):
        self._entries.clear()


class Governor:
    # controlling business logic that ties all modules together

    def __init__(self):
        self._power_balance = None
        self._power_balance_strategy = None
        self._id = None
        self._forecaster = None
        self._hardware_backend_module = None
        self._requirement = None
        self.diff_to_real_value = []
        self.curr_time = None
        self.message_journal = MessageJournal()
        self.solution_journal = MessageJournal()
        self.solver_triggered = False
        self.triggered_due_to_timeout = True

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, id):
        self._id = id

    @property
    def power_balance(self):
        return self._power_balance

    @power_balance.setter
    def power_balance(self, power_balance):
        self._power_balance = power_balance

    @property
    def power_balance_strategy(self):
        return self._power_balance_strategy

    @power_balance_strategy.setter
    def power_balance_strategy(self, strategy):
        self._power_balance_strategy = strategy

    async def try_balance(self):
        assert self._power_balance is not None
        assert self._power_balance_strategy is not None
        time_span = self._power_balance.ledger[0][0].time_span

        if len(time_span) > 1:
            initial = self._power_balance_strategy.find_initial_requirement(self._power_balance,
                                                                            xboole.InitiatingParty.Local)
            power_balance_original = [deepcopy(self._power_balance) for _ in range(len(time_span))]
            solutions = []

            for idx, interval in enumerate(time_span):
                power_balance = power_balance_original[idx]
                for i in power_balance:
                    val = deepcopy(i[1].power)
                    del i[1].power[:]
                    i[1].power.append(val[idx])

                    val = deepcopy(i[1].time_span)
                    del i[1].time_span[:]
                    i[1].time_span.append(val[idx])

                    val = deepcopy(i[1].message.time_span)
                    del i[1].message.time_span[:]
                    i[1].message.time_span.append(val[idx])

                    val = deepcopy(i[1].message.value)
                    del i[1].message.value[:]
                    i[1].message.value.append(val[idx])

                res = self._power_balance_strategy.solve(
                    power_balance, xboole.InitiatingParty.Local)

                if res is not None and len(res[0].tv[0]) != 0:
                    solutions.append((res[0], res[1], initial))
                else:
                    return None
            return solutions
        # only one interval
        return self._power_balance_strategy.solve(
            self._power_balance, xboole.InitiatingParty.Local)
