from mango_library.negotiation.winzent import xboole


class Tv:
    def __init__(self, values=[]):
        if type(values) == list:
            for i in range(0, len(values)):
                values[i] = xboole.Tval(values[i])
            self._val = values
        else:
            self._val = []
            if isinstance(values, int):
                self._val.append(xboole.Tval(values))
            else:
                for i in range(0, len(values)):
                    self._val.append(values[i])

    def __getitem__(self, key):
        if key >= len(self):
            return xboole.Tval('-')
        return self._val[key]

    def __len__(self):
        return len(self._val)

    def append(self, value):
        if isinstance(value, int):
            self._val.append(xboole.Tval(value))
        else:
            self._val.append(value)

    def orth_test(self, tv2):
        """ check if two vectors are orthogonal
        by calling orth_test of xboole.Tval for each tval

        :param tv2: vector with which self is compared regarding orthogonality
        :return:

        """
        for i in range(0, len(self)):
            # check if orthogonal
            if xboole.Tval(self[i]).orth_test(tv2[i]):
                return True
        return False

    def make_orth(self, tv2):
        """

        :param tv2: self is made orthogonal according to tv2
        :return:

        """
        marked_positions = []
        for i in range(0, len(self)):
            if self[i].get_val() == 1 or self[i].get_val() == 0:
                if tv2[i].get_val() == '-':
                    marked_positions.append(i)
        orth_vec = []
        for i in range(0, len(self)):
            if i in marked_positions:
                orth_vec.append(int(not self[i].get_val()))
            else:
                orth_vec.append(self[i].get_val())
        return xboole.Tv(orth_vec)

    def bb_test(self, other):
        """checks if blockbuilding between two tvs(self and other)
        is possible

        :param other: other tv
        :return: True if blockbuilding is possible
        """
        bb_test = False
        if len(self) != len(other):
            return False
        for i in range(0, len(self)):
            if self[i].b ^ other[i].b:
                # would be combination of '-' with 0 or 1:
                # signs are either equal nor 01 combination
                return False

            if self[i].a ^ other[i].a:
                # 01 combination found
                if bb_test:  # second time combination 01
                    return False
                bb_test = True
        return bb_test

    def resize(self, new_length):
        for i in range(len(self), new_length):
            self._val.append('-')

    def sort(self):
        return sorted(self._val)

    def fill(self, start, count, value):
        for i in range(start, (start + count)):
            self._val.append(value)
