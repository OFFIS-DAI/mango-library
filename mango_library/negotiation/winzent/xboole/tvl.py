import collections
import copy

from mango_library.negotiation.winzent import xboole


class Tvl:
    def __init__(self, vv=None):
        self._vv = collections.OrderedDict()
        self._max = 0
        if type(vv) == list:
            if len(vv) is not 0:
                for i in range(0, len(vv)):
                    self._vv[vv[i]] = []
        elif vv is not None:
            raise ValueError("TVL variables vector needs to be a list.")
        self._tv = []  # saves tvals for tvs

    def __getitem__(self, key):
        """proxy is created, tv = values at position key in vv

        :param key:
        :return: tv proxy
        """
        tv = []
        if isinstance(key, int):
            for k, v in self._vv.items():
                tv.append(v[key])
            return xboole.TVProxy(tv, self._vv, key)
        else:  # for sensibility
            for k, v in self._vv.items():
                if k == key:
                    tv = list(v)
        return xboole.TVProxy(tv, self._vv)

    def __iter__(self):
        return xboole.Tv()

    def __eq__(self, other):
        if isinstance(other, xboole.Tvl):
            if len(self) != len(other):
                return False
            # bring keys in same order
            other_tvl = xboole.Tvl(list(self._vv.keys()))
            for k, v in self._vv.items():
                if k not in other.vv:
                    return False
                other_tvl._vv[k] = other.vv[k]

            # initialize tv in same order
            for i in range(0, len(other)):
                a = []
                for k, v in other_tvl.vv.items():
                    a.append(v[i])
                other_tvl._append_tv(a)
                del a[:]

            # check if tvs are same in both tvls
            for i in range(0, len(self._tv)):
                if self._tv[i] not in other_tvl._tv:
                    return False

            return True

    @property
    def vv(self):
        return self._vv

    @vv.setter
    def vv(self, vv):
        self._vv = vv

    @property
    def tv(self):
        return self._tv

    @tv.setter
    def tv(self, tv):
        self._tv = tv

    def append(self, item):
        """ distinguishes whether tvl, list or string was appended,
        appends item to vv and tv by calling function for type op item

        :param item: object which is appended
        """
        if isinstance(item, xboole.Tvl):
            self._append_tvl(item)
        elif (isinstance(item, xboole.Tv)) or (isinstance(item, str)):
            self._append_tv(item)
        elif isinstance(item, list):
            tv1 = xboole.Tv(item)
            self._append_tv(tv1)
        self._max = len(self.tv) - 1

    def _append_tv(self, tv):
        i = 0
        diff = 0
        if len(self._vv.keys()) > len(tv):
            diff = len(self.vv.keys()) - len(tv)
        for k, v in self.vv.items():  # initialize vv
            if i <= len(tv):
                if tv[i] == '-':
                    self.vv[k].append('-')
                else:
                    self.vv[k].append(int(tv[i]))
                i += 1
            else:
                self._vv[k].append('-')
        zw = []
        for i in range(0, len(tv)):  # initialize tv
            if tv[i] == '-':
                zw.append(xboole.Tval('-'))
            else:
                zw.append(xboole.Tval(int(tv[i])))
        for i in range(0, diff):
            zw.append(xboole.Tval('-'))
        self._tv.append(zw)

    def _append_tvl(self, other):
        num_of_new_keys = 0
        for k, v in other.vv.items():  # initialize vv
            if k in self._vv:
                self._vv[k].append(list(v))
            else:
                self._vv[k] = []
                # other vectors in vv and tv at new key = '-'
                for i in range(0, len(self)):
                    self._vv[k].append('-')
                    self._tv[i].append(xboole.Tval('-'))
                self._vv[k].append(list(v))
                num_of_new_keys += 1

        a = []
        for k, v in other.vv.items():  # initialize tv
            a.append(list(v))
        for i in range(0, len(a)):
            a[i] = xboole.Tval(a[i][0])
        self._tv.append(a)
        self.orth()

    def __len__(self):
        """

        :return: number of inserted tvs
        """
        return len(self._tv)

    def __iter__(self):
        self._n = -1
        return self

    def __next__(self):
        if self._n < self._max:
            self._n += 1
            return xboole.TVProxy(self._tv[self._n], self._vv)
        else:
            raise StopIteration

    def insert(self, index, item):
        """
        append item at position index in vv and tv
        :param index: position
        :param item: object (here tv) which is appended
        """
        i = 0
        for k, v in self._vv.items():  # initialize vv
            if i <= len(item):
                if isinstance(item[i], xboole.Tval):
                    v.insert(index, item[i])
                else:
                    v.insert(index, xboole.Tval(item[i]))
                i += 1
        zw = []
        for i in range(0, len(item)):  # initialize tv
            if isinstance(item[i], xboole.Tval):
                zw.append(item[i])
            else:
                zw.append(xboole.Tval(item[i]))
        self._tv.insert(index, zw)

    def pop(self, index=None):
        """ remove tv

        if no index is given, last tv is deleted
        :param index: position from tv which is supposed to be deleted
        :return:
        """
        if index is None:
            self._tv.pop()
            for k, v in self._vv.items():
                v.pop()
            return
        self._tv.pop(index)
        for k, v in self._vv.items():
            v.pop(index)

    def orth(self):
        """called to make tvl orthogonal

        Creates new tvl with only tvs that are orthogonal to each
        other. Always considering last tv in tvl and comparing it
        with each one left in tvl, if it is not orthogonal to the
        other one, its made orthogonal by calling method make_orth
        and then is added to new tvl, if it is already orthogonal
        to the other one, it is added to new tvl directly.

        :return:
        """
        a = list(self._vv.keys())
        tvl2 = xboole.Tvl(a)
        for i in range(0, len(list(self._vv.keys()))):
            while len(self._tv) >= 1:
                curr_vec = xboole.Tv(self._get_vector())
                orth = True
                if self._is_last_vector():
                    tvl2.insert(0, curr_vec)
                    self._tv = tvl2.tv
                    self._vv = tvl2.vv
                    return

                for i in range(0, len(self._tv)):
                    if (xboole.Tv(self._tv[i]).orth_test(curr_vec)
                            == False):
                        tvl2.insert(0, xboole.Tv(self._tv[i])
                                    .make_orth(curr_vec))
                        orth = False
                        break
                if orth:
                    tvl2.insert(0, curr_vec)

    def _is_last_vector(self):
        """ checks if there is only one tv left in tvl

        used for function orth

        :return: True, if theres only one tv left in tvl, else False
        """
        if len(self._tv) == 0:
            return True
        return False

    def _get_vector(self):
        """deletes last tv in tv and vv of tvl

        used for function orth

        :return: last tv in tv of tvl
        """
        for k, v in self._vv.items():
            if len(v) != 0:
                # if it was null, there can still be values in tv left,
                # proxy '-' would be returned
                v.pop()
        return self._tv.pop()

    def isc(self, other):
        """ creates intersection between two tvls

        :param other: tvl with which self builds intersection
        :return: intersection as tvl
        """
        isc_vv = copy.copy(self._vv)
        for k, v in isc_vv.items():
            v = []
        if len(self) > 0 and len(other) > 0:
            for i in range(0, len(self)):
                for j in range(0, len(other)):
                    if not self[i].orth_test(other[j]):
                        for k, v in self._vv.items():
                            isc_vv[k].append(self[k][i].isc(other[k][j]))

                        for k, v in other.vv.items():
                            if k not in self._vv:
                                if not self[k][i].orth_test(other[k][j]):
                                    if k not in isc_vv:
                                        isc_vv[k] = []
                                    isc_vv[k].append(
                                        self[k][i].isc(other[k][j]))
            isc_tvl = xboole.Tvl(list(isc_vv.keys()))
            isc_tvl._vv = isc_vv

            a = []
            for k, v in isc_vv.items():  # initialize tv
                a.append(list(v))
            for i in range(0, len(a)):
                a[i] = xboole.Tval(a[i][0])
            isc_tvl._tv.append(a)
            isc_tvl.orth()
            return isc_tvl
        else:
            emtpy = xboole.Tvl()
            return emtpy

    def oobc(self):
        """ minimizes tvl to minimum number of rows

        calls blockbuilding method (obb) and then blockswitch (obc)
        after each blockswitch call blockbuilding method again
        while blockbuilding reduces number of rows: begin
        with calling blockswitch again
        z = obb(x)
        z1 = obc(z)
        z = obb(z1)
        """
        rows_changed = True
        self.obb()
        while rows_changed is True:
            self.obc()
            rows_changed = self.obb()

    def obb(self):
        """ for each pair of tvs in self, check if blockbuilding is
        possible and build blocks

        :return: True if at least one block was builded
        """
        blocks_built = False
        for i in range(0, len(self._tv) - 1):
            if self._bb(i, i + 1):
                blocks_built = True
        return blocks_built

    def _bb(self, i, j):
        """ if blockbuilding is possible, build blocks by replacing
        old pair of rows by new row
        """
        if xboole.Tv(self._tv[i]).bb_test(xboole.Tv(self._tv[j])):
            tv = self._bb1(xboole.Tv(self._tv[i]), xboole.Tv(self[j]))
            self.pop(j)
            self.pop(i)
            self.insert(i, tv)
            return True
        return False

    def _bb1(self, tv1, tv2):
        """ summarize two tv which differ in one column
        """
        a = []
        for i in range(0, len(tv1)):
            if tv1[i] == tv2[i]:
                if tv1[i] == xboole.Tval('0'):
                    a.append(0)
                elif tv1[i] == xboole.Tval('1'):
                    a.append(1)
                else:
                    a.append('-')
            else:
                a.append('-')
        return xboole.Tv(a)

    def obc(self):
        """ switch blocks if possible

        """
        for i in range(0, len(self._tv)):
            for j in range(0, len(self._tv)):
                if i == j:
                    continue
                if self._blocks_to_switch(i, j):
                    return True
        return False

    def _blocks_to_switch(self, position_one, position_two):
        """ to switch blocks, tvs only differ in two positions, one position
        needs to be '-'

        possibilities : tv1: 01   or 11  or 00  or 10
                        tv2: 1-      0-     1-     0-

        :param position_one: position in first tv
        :param position_two: position on second tv
        :return:
        """
        counter = 0
        pos = []
        is_stroke = False
        for i in range(0, len(self._tv[position_one])):
            if i >= len(self._tv[position_two]):
                return False
            if self._tv[position_one][i] != self._tv[position_two][i]:
                if self._tv[position_one][i] == xboole.Tval('-'):
                    return False
                if (self._tv[position_two][i]) == xboole.Tval('-'):
                    if i == len(self._tv[position_two]) - 1:
                        is_stroke = True
                    else:
                        return False
                counter += 1
                pos.append(i)
            if counter == 2:  # differ in two columns
                if is_stroke is False:  # no stroke in different positions
                    return False
                self._switch_blocks(position_one, position_two, pos)
                return True

        return False

    def _switch_blocks(self, position_one, position_two, pos_in_tvs):
        """ switches blocks of given tvs at given postions

        :param position_one: position of tv1 in tvl
        :param position_two: position of tv2 in tvl
        :param pos_in_tvs: positions in tvs as list, where
        blockswitch takes place
        """
        first_pos_in_tv = pos_in_tvs[0]
        sec_pos_in_tv = pos_in_tvs[1]
        if position_one > position_two:
            position_one, position_two = position_two, position_one

        key_one = None
        key_two = None

        for k, v in self._vv.items():
            if list(self._vv.keys()).index(k) == first_pos_in_tv:
                key_one = k
            if list(self._vv.keys()).index(k) == sec_pos_in_tv:
                key_two = k

        if self._tv[position_two][sec_pos_in_tv] == xboole.Tval('-'):
            if self._tv[position_one][first_pos_in_tv] == \
                    self._tv[position_one][sec_pos_in_tv]:
                self._vv[key_two][position_two] = self._tv[position_one][
                    first_pos_in_tv].ops().get_val()
                self._tv[position_two][sec_pos_in_tv] = \
                    self._tv[position_one][first_pos_in_tv].ops()
                self._vv[key_one][position_one] = xboole.Tval('-')
                self._tv[position_one][first_pos_in_tv] = xboole.Tval('-')
            else:
                self._tv[position_one][first_pos_in_tv], \
                self._tv[position_two][sec_pos_in_tv] = \
                    self._tv[position_two][sec_pos_in_tv], \
                    self._tv[position_one][first_pos_in_tv]
                self._vv[key_one][position_one], self._vv[key_two][
                    position_two] \
                    = self._vv[key_two][position_two], self._vv[key_one][
                    position_one]

    def lexicographically_next_permutation(self, position):
        """
        Generates the lexicographically next permutation.

        Input: a permutation, called "a". This method modifies
        "a" in place. Returns True if we could generate a next
        permutation. Returns False if it was the last permutation
        lexicographically.
        """
        tv = self._tv[position]
        i = len(tv) - 2
        while not (i < 0 or tv[i] < tv[i + 1]):
            i -= 1
        if i < 0:
            return False
        # else
        j = len(tv) - 1
        while tv[j] < tv[i] or tv[j] == tv[i]:
            j -= 1
        tv[i], tv[j] = tv[j], tv[i]  # swap
        # reverse elements from position i+1 till the end of the sequence
        tv[i + 1:] = reversed(tv[i + 1:])
        i = 0
        while i != len(tv):
            for k, v in self._vv.items():
                v[position] = tv[i]
                i += 1
        return True

    def sort_tvl(self, position):
        self._tv[position].sort()
        i = 0
        while i != len(self._tv[position]):
            for k, v in self._vv.items():
                v[position] = self._tv[position][i]
                i += 1
