from mango_library.negotiation.winzent import xboole


class Tval:
    def __init__(self, value):
        if isinstance(value, xboole.Tval):
            self._a = value.a
            self._b = value.b
            return
        if value == 1 or value == '1':
            self._a = True
            self._b = True
        elif value == 0 or value == '0':
            self._a = False
            self._b = True
        elif value == '-':
            self._a = False
            self._b = False
        elif isinstance(value, list):
            self._a = value[0]
            self._b = value[1]
        else:
            raise ValueError("Illegal value '{}' for Ternary Value"
                             .format(value))

    def __eq__(self, other):
        if type(other) == Tval:
            return self._a == other.a and self._b == other.b
        elif type(other) == bool:
            return self._b and other == self._a or not self._b
        elif other == 1 or other == '1':
            if self._a and self._b:
                return True
        elif other == 0 or other == '0':
            if not self._a and self._b:
                return True
        elif other == '-':
            if not self._a and not self._b:
                return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        if self._a and self._b:
            return '1'
        elif not self._a and self._b:
            return '0'
        return '-'

    def __bool__(self):
        if self._b:
            return self._a
        return False

    def get_val(self):
        """ returns value of tval as int or string

        :return:
        """
        if self._b:
            if self._a:
                return 1
            return 0
        return '-'

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, a):
        self._a = a

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, b):
        self._b = b

    def orth_test(self, other):
        """checks if two values are orthogonal

        :param other:
        :return: bool whether values are orthogonal or not
        """
        return (self.a ^ other.a) and self._b and other.b

    def isc(self, other):
        """creates intersection of two tvals

        :rtype: tval
        """
        a = self.a or other.a
        b = self.b or other.b
        return xboole.Tval([a, b])

    def ops(self):
        """return opposite of self, for 1 return 0
        and for 0 return 1 (never used with strokes)

        """
        if not self.b:  # called for '-'
            return None
        return xboole.Tval(not self.a)

    def __int__(self):
        if self.get_val() == '-':
            return '-'
        return int(self.get_val())

    def __lt__(self, other):
        if self.b and other.b:
            return self.a < other.a
        elif self.b and not other.b:
            return False
        else:
            return True
