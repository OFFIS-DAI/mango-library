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

    def replace(self, dict):
        self._entries = dict


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
        self.curr_requirement_values = 0
        self.message_journal = MessageJournal()
        self.solution_journal = MessageJournal()
        self.solver_triggered = False
        self.triggered_due_to_timeout = False

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, id):
        self._id = id

    @property
    def power_balance(self):
        return self._power_balance

    def remove_from_power_balance(self, agent_id):
        if self._power_balance is not None:
            for offer in self._power_balance._ledger[self.curr_time]:
                if offer.message.sender == agent_id:
                    self._power_balance._ledger[self.curr_time].remove(offer)
                    break

    def get_from_power_balance(self, agent_id, start_time):
        if self._power_balance is not None:
            if start_time in self._power_balance._ledger:
                for offer in self._power_balance._ledger[start_time]:
                    if offer.message.sender == agent_id:
                        return offer

    def get_msg_from_message_journal(self, sender, type):
        return_msg = None
        for msg in self.message_journal._entries.values():
            if msg.sender == sender and msg.msg_type == type:
                return_msg = msg
        return return_msg

    @power_balance.setter
    def power_balance(self, power_balance):
        self._power_balance = power_balance

    @property
    def power_balance_strategy(self):
        return self._power_balance_strategy

    @power_balance_strategy.setter
    def power_balance_strategy(self, strategy):
        self._power_balance_strategy = strategy

    def try_balance(self):
        assert self._power_balance is not None
        assert self._power_balance_strategy is not None
        return self._power_balance_strategy.solve(
            self._power_balance, xboole.InitiatingParty.Local)
