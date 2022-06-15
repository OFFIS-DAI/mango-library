from mango_library.negotiation.winzent.xboole import Tv, Tval


class TVProxy:
    def __init__(self, tv, parent_tvl_vv, position=None):
        self._tv = Tv(tv)
        self._parent_tvl_vv = parent_tvl_vv
        self._max = len(self._tv) - 1
        self._position = position

    def __getitem__(self, key):
        # value, with which proxy was created is not in parent.tvl.vv
        if len(self._tv) == 0:
            return Tval('-')
        if isinstance(key, int):
            if isinstance(self._tv[key], Tval):
                return self._tv[key]
            return Tval(self._tv[key])
        elif key in self._parent_tvl_vv:
            for k, v in self._parent_tvl_vv.items():
                if k == key:
                    return list(v)[self._position]

    def __iter__(self):
        self._n = -1
        return self

    def __next__(self):
        if self._n < self._max:
            self._n += 1
            return self._tv[self._n]
        else:
            raise StopIteration

    def __len__(self):
        return len(self._tv)

    @property
    def tv(self):
        return self._tv

    @tv.setter
    def tv(self, tv):
        self._tv = tv

    def orth_test(self, other):
        """ check if two vectors are orthogonal
        by calling orth_test of xboole.Tval for each tval

        :param tv2: vector with which self is compared regarding orthogonality
        :return:
        """
        for i in range(0, len(self)):
            orth = self._tv[i].orth_test(other.tv[i])
            if orth:
                return True
        return False

    def sort(self):
        self._tv.sort()
