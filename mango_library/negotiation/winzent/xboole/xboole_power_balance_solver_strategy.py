from collections import namedtuple

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
                if (abs(r[1].forecast.second) <= abs(pb[i][1].forecast.second))\
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
        result = xboole.Result()
        result.solved(False)
        if len(power_balance) == 0:
            return None
        # 2. find out, which requirement initiated this calculation
        initial_requirement = self.find_initial_requirement(power_balance,
                                                            initiator)

        # 1. Calculate gcds for t and p values
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
        value = value=0
        for i in power_balance:
            if i[1] == initial_requirement:
                continue
            value += i[1].forecast.second
        return value

    @staticmethod
    def adopt_requirements_value(power_balance, initial_requirement,
                                 value):
        for i in power_balance:
            if i[1] == initial_requirement:
                i[1].forecast.second(value)
            # i[1].message.value(value)
