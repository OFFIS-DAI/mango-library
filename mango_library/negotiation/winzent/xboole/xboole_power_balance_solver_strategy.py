from collections import namedtuple
from copy import deepcopy

from mango_library.negotiation.winzent import xboole
from mango_library.negotiation.winzent.xboole import PowerBalanceSolverStrategy

GCD = namedtuple("GCD", "gcd_t gcd_p")


class InsufficientOffersError(Exception):
    """Exception raised for errors in the input.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


class XboolePowerBalanceSolverStrategy(PowerBalanceSolverStrategy):

    @staticmethod
    def variables_for(requirement, gcds):
        vv = []
        id = str(requirement.message.id)
        kw = abs(requirement.forecast.second)

        for i in range(len(gcds)):
            if abs(gcds[i]) <= kw:
                vv.append(id + ':' + str(i))
            else:
                break
        return vv

    def create_request_tvls(self, power_balance, initial_requirement, gcds):
        request_tvls = []
        request_tvls.append(xboole.Tvl())

        for idx in power_balance:
            if idx[1] == initial_requirement:
                continue
            id = str(idx[1]._message.id)
            kw = abs(idx[1].forecast.second)
            for p in range(len(gcds)):
                if abs(gcds[p]) <= kw:
                    request_tvls[0].vv[id + ':' + str(p)] = []
        for tvl in request_tvls:
            tvl = self.fill_tvs(len(tvl), tvl, idx)
        return request_tvls

    def fill_tvs(self, num_p_ones, tvl, i):
        tv = xboole.Tv()
        tv._val = []
        # tv.fill(0, (len(tvl.vv()) - int(num_p_ones)), xboole.Tval('0'))
        # tv.fill(len(tvl.vv()) - int(num_p_ones),
        # int(num_p_ones), xboole.Tval('1'))
        tv.fill(0, len(tvl.vv), xboole.Tval('1'))
        tvl.append(tv)
        return tvl

    def offer_tvl_for(self, offer, gcds):
        vv = self.variables_for(offer, gcds)
        offer_tvl = xboole.Tvl(vv)
        ones = xboole.Tv()
        for i in range(0, len(vv)):
            ones.append(True)
        offer_tvl.append(ones)

        zeros = xboole.Tv()
        zeros._val = []
        for i in range(0, len(vv)):
            zeros.append(False)
        offer_tvl.append(zeros)
        offer_tvl.oobc()
        return offer_tvl

    def calculate_partitions(self, power_balance):
        values = []
        for i in power_balance:
            if not isinstance(i[1].forecast.second, int):
                i[1].forecast.second = i[1].forecast.second[0]
            i[1].forecast.second = abs(i[1].forecast.second)
        for i in power_balance:
            if len(values) == 0:
                values.append(i[1].forecast.second)
            else:
                if abs(values[-1]) > abs(i[1].forecast.second):
                    if abs(values[0]) > abs(i[1].forecast.second):
                        values.insert(0, i[1].forecast.second)
                    else:
                        for j in range(len(values)):
                            if abs(values[j]) <= \
                                    abs(i[1].forecast.second):
                                continue
                            if abs(values[j]) > abs(i[1].forecast.second):
                                values.insert(j, i[1].forecast.second)
                                break
                elif abs(values[-1]) < abs(i[1].forecast.second):
                    values.append(i[1].forecast.second)
        return values

    @staticmethod
    def find_initial_requirement(power_balance, initiator):
        r = None
        if initiator == xboole.InitiatingParty.Local:
            for i in power_balance:
                if i[1].from_target:
                    r = i[1]
                    break
        else:
            pb = []
            for i in power_balance:
                pb.append(i)
            r = pb[0]
            for i in range(1, len(pb) - 1):
                if (abs(r[1].forecast.second) <= abs(pb[i][1].forecast.second)) \
                        and ((abs(r[0]) * 1000)
                             <= (abs(pb[i][0] * 1000))):
                    r = pb[i]

        assert r is not None
        if isinstance(r, tuple):
            return r[1]
        return r

    def create_offer_tvls(self, power_balance, initial_requirement, gcds):
        tvls = []
        for i in power_balance:
            if i[1] == initial_requirement:
                continue
            tvls.append(self.offer_tvl_for(i[1], gcds))
        return tvls

    def solve(self, power_balance, initiator):
        time_span = list(power_balance.ledger.values())[0][0].time_span
        if len(time_span) > 1:
            initial = self.find_initial_requirement(power_balance, xboole.InitiatingParty.Local)

            power_balance_original = [deepcopy(power_balance) for _ in range(len(time_span))]
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

                res = self.solve_single_time_slot(
                    power_balance, xboole.InitiatingParty.Local)

                if res is not None and len(res[0].tv[0]) != 0:
                    solutions.append((res[0], res[1], initial))
                else:
                    return None
            return self.solution_postprocessing(solutions)
        # only one interval
        return self.solution_postprocessing([self.solve_single_time_slot(
            power_balance, xboole.InitiatingParty.Local)])

    def solve_single_time_slot(self, power_balance, initiator):
        if len(power_balance) == 0:
            return None

        result = xboole.Result()
        result.solved(False)

        # 1. find out, which requirement initiated this calculation
        initial_requirement = self.find_initial_requirement(power_balance,
                                                            initiator)
        # 2. Calculate gcds for t and p values
        values = self.calculate_partitions(power_balance)

        # 3. create TVLs for all offers
        offer_tvls = self.create_offer_tvls(power_balance,
                                            initial_requirement, values)
        # 4. formulate requirement clauses
        solution = xboole.Tvl()
        request_tvls = self.create_request_tvls(power_balance,
                                                initial_requirement, values)

        # 5. Solve, request by request
        request_tvl = xboole.Tvl()
        can_permute = True
        while can_permute and len(solution) == 0:
            for tvl in request_tvls:
                assert (len(tvl) == 1)
                request_tvl.append(tvl)
            solution = request_tvl

            for tvl in offer_tvls:
                offer_tvl = tvl
                solution = solution.isc(offer_tvl)
                if len(solution) == 0:
                    break

            if len(solution) == 0:
                break

            j = 0
            for i in reversed(request_tvls):
                j += 1
                did_permute = i.lexicographically_next_permutation(0)

                if did_permute:
                    break

                if not did_permute and j != len(request_tvls) - 2:
                    i.sort_tvl(0)
                else:
                    can_permute = did_permute

        return solution, values, initial_requirement

    @staticmethod
    def find_afforded_value(power_balance, initial_requirement):
        value = 0
        for i in power_balance:
            if i[1] == initial_requirement:
                continue
            value += i[1].forecast.second
        return value

    @staticmethod
    def adapt_requirements_value(power_balance, initial_requirement,
                                 value):
        for i in power_balance:
            if i[1] == initial_requirement:
                i[1].forecast.second(value)

    @staticmethod
    def solution_postprocessing(result):
        initial_req = result[0][2]
        if is_iterable(result):
            answers = {}
            gcd_p = {}
            for idx, entry in enumerate(result):
                solution = entry[0]
                if len(solution) > 0:
                    # There was actually a solution. Split solution values according
                    # to agents taking part in it
                    for k, v in solution.vv.items():
                        if v[0] == xboole.Tval(1):
                            if idx not in answers.keys():
                                answers[idx] = []
                            answers[idx].append(k)
                    gcd_p[idx] = entry[1]
            afforded_values = {}
            final = {}
            for key, value in gcd_p.items():
                initial_value = initial_req.message.value[key]
                afforded_value = 0
                answer_objects = []
                # create dict with agents and used power value per agent
                for j in range(len(answers[key])):
                    answer_objects.append(answers[key][j].split(':', 1)[0])
                sol = DictList()
                # from the solution, store the actual values for each agent in a dict (sol)
                for j in range(len(answer_objects)):
                    sol.add((answer_objects[j], gcd_p[key][int(answers[key][j][-1])]))

                for k, v in sol.values.copy().items():
                    if abs(afforded_value) + abs(v) >= abs(initial_value):
                        diff = abs(initial_value) - abs(afforded_value)
                        sender = k
                        if sender not in final.keys():
                            final[sender] = {}
                        if key not in final[sender].keys():
                            final[sender][key] = []
                        final[sender][key].append(diff)
                        afforded_value += diff
                        break
                    sender = k
                    if sender not in final.keys():
                        final[sender] = {}
                    if key not in final[sender].keys():
                        final[sender][key] = []
                    final[sender][key].append(v)
                    afforded_value += v
                afforded_values[key] = afforded_value
            return final, afforded_values, initial_req
        return None


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


class DictList:
    def __init__(self):
        """
        DictList is a helper object.
        """
        self._values = {}

    def add(self, answer_tuple):
        if len(self._values.items()) == 0:
            self._values[answer_tuple[0]] = answer_tuple[1]
            return

        for k, v in self._values.copy().items():
            if k == answer_tuple[0]:
                if answer_tuple[1] > v:
                    self._values[k] = answer_tuple[1]
                return
            self._values[answer_tuple[0]] = answer_tuple[1]
        return self

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, values):
        self._values = values
