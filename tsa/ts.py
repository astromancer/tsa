import numbers
import operator

import numpy as np


# import uncertainties.unumpy as unp  # linear uncertainty propagation
# since this library is tracking correlations between rvs operations like mean
# on an array is n^2 or n^3 which means its actually too slow to be useful.

# need class that tracks uncertainties but with correlation tracking
# optionally disabled
# Options: Linearized uncertainty propagation via `uncertainties`
#        : Full distribution calculation with rv arithmetic via `pacal`
# if _handle_uncertainties == 0:

# for now calculations on uncertainty has to be done manually which is a pain..


class TimeSeries(object):
    """
    A basic univariate time series with uncertainties
    """

    @staticmethod
    def _parse_init_args(*args, **kws):
        # all parameters passed as keyword args
        if len(args) == 0:
            t, x, u = map(kws.get, 'txu')
        # signals only
        elif len(args) == 1:
            # Assume here each row gives individual signal for a TS
            x, = args
            t = kws.get('t')
            u = kws.get('u')
        # times & signals given
        elif len(args) == 2:  # No errors given
            t, x = args
            u = kws.get('u')
        # times, signals, errors given
        elif len(args) == 3:
            t, x, u = args
        # bad params
        else:
            raise ValueError('Incorrect number of arguments: %i' % len(args))

        # check data
        if x is None:
            raise ValueError('Data should be a numerical sequence')
            # otherwise a object array containing `None` results below

        return t, x, u

    def __init__(self, *args, **kws):

        t, x, u = self._parse_init_args(*args, **kws)

        # make sure we have 1d array
        x = np.ma.asarray(x).squeeze()
        if x.ndim != 1:
            raise ValueError('Time Series data should be 1D')

        # times
        self._t = self._x = self._u = None
        self.t = t
        self.x = x
        self.u = u

        # if u is None:
        #     # data is array
        #     self._x = x
        # else:
        #     # data represented internally as unumpy.uarray
        #     self._x = unp.uarray(x, u)

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, t):
        # if t is None:
        #     t = np.arange(len(x))
        #     # todo maybe better to keep None and handle special case at
        #     # function level
        # else:
        if t is not None:
            t = np.asanyarray(t).squeeze()
            n = len(self.x)
            if len(t) != n:
                raise ValueError(
                        f'Unequal number of points between data (n={n}) and '
                        f'time (n={len(t)}) arrays.')

        self._t = t

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        self._x = np.ma.array(x)

    @property
    def u(self):
        return self._u

    @u.setter
    def u(self, u):
        self._u = np.ma.array(u)

    @property
    def n(self):
        """Number of data points"""
        return len(self)

    # def fold(self, eph):


    def plot(self, ax=None, *args, **kws):
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        return ax.errorbar(self.t, self.x, self.u)

    # @property
    # def data(self):
    #     return unp.nominal_values(self._x)
    #     # always returns masked array with full mask. probably not that
    #     # efficient
    #
    # @property
    # def std(self):
    #     return unp.std_devs(self._x)
    #     # always returns masked array with full mask. probably not that
    #     # efficient

    # def __repr__(self):
    #     ''

    def __len__(self):
        return len(self._x)

    def __iter__(self):
        """allow unpacking: `t, y, u = ts`"""
        return self.t, self.x, self.u

    def __pos__(self):
        return self

    def __neg__(self):
        return self.__class__(self.t, -self._x)

    def __abs__(self):
        return self.__class__(self.t, abs(self._x))

    # arithmetic
    # --------------------------------------------------------------------------
    def _arithmetic(self, other, op):
        #
        if isinstance(other, TimeSeries):
            # Can only really do time series if they are simultaneous
            if self.n != other.n:
                raise ValueError('Arithmetic on %s objects with different '
                                 'sizes not permitted' % self.__class__)

            # TODO: propagate uncertainties!
            return self.__class__(self.t, op(self._x, other._x))

        # arithmetic with complex numbers not supported
        if isinstance(other, numbers.Complex) and \
                not isinstance(other, numbers.Real):
            raise TypeError('Arithmetic with complex numbers not supported')
            # all other number types should be OK

        # array-like (any object that can create an array / any duck-type array)
        other = np.asanyarray(other)
        return self.__class__(self.t, op(self._x, other))

    def __add__(self, other):
        return self._arithmetic(other, operator.add)

    def __sub__(self, other):
        return self._arithmetic(other, operator.sub)

    def __mul__(self, other):
        return self._arithmetic(other, operator.mul)

    def __truediv__(self, other):
        return self._arithmetic(other, operator.truediv)

    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__
    __rtruediv__ = __truediv__




class TimeSeriesDev:
    """
    A basic time series object.

    This class ties together the functionality of various libraries for
    analysing time series data.

    Features
    --------
     * automatic (linearized) uncertainty propagation (assumes normally
       distributed data)
     *
     * TODO:
      interface with
       -statmodels
      - pandas
      - pacal
      - uncertainties
        ~ Approximate (linear) uncertainty propagation. Will only yield
          accurate results when used in computation if uncertainties are small
          compared to variation of the (non-linear) function through which it is
          mapped.

          note:
          "Error estimates for non-linear functions are biased on account of
          using a truncated series expansion. The extent of this bias depends
          on the nature of the function. For example, the bias on the error
          calculated for log(1+x) increases as x increases, since the expansion
          to x is a good approximation only when x is near zero."
          from: https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Caveats_and_warnings


     * support for multivariate (simultaneous) data
     * spectral estimation techniques
     * sampled time series + visualizations.

     Not supported
     - units.  Make sure you use compatible units when doing arithmetic

    """

    # _allow_empty = False
    _handle_uncertainties = 0

    def __init__(self, *args, **kws):

        # handle uncertainties
        # Options: Linearized uncertainty propagation via `uncertainties`
        #        : Full distribution calculation with rv arithmetic via `pacal`
        # if _handle_uncertainties == 0:

        # null time series
        # if len(args) == 0:
        #     '' # is this useful??

        # signals only
        if len(args) == 1:
            # Assume here each row gives individual signal for a TS
            self.x = np.ma.asarray(args[0])
            self.t = kws.get('t')
            self.u = kws.get('u')

        # times & signals given
        elif len(args) == 2:  # No errors given
            t, data = args
            assert len(t) == len(data)
            self.t = np.asanyarray(t)
            self.x = np.ma.asarray(data)
            self.u = kws.get('u')

        # times, signals, errors given
        elif len(args) == 3:
            t, data, u = args
            self.t = np.asanyarray(t)
            self.x = np.ma.asarray(data)
            self.u = u

    # todo: check if `t` equi-spaced

    # def __repr__(self):
    # nice repr with pm etc

    def __len__(self):
        return len(self.x)

    def __pos__(self):
        return self

    def __neg__(self):
        return self.__class__(self.t, -self.x, self.u)

    def __abs__(self):
        return self.__class__(self.t, abs(self.x), self.u)

    # arithmetic
    # --------------------------------------------------------------------------
    def __add__(self, other):
        #
        if isinstance(other, numbers.Real):
            # real numeric type
            return self.__class__(self.t, self.x + other, self.u)

        if isinstance(other, TimeSeries):
            # Can only add time series if they are simultaneous
            if not all(self.t == other.t):
                raise ValueError('Cannot add %s with different time stamps' %
                                 self.__class__)

            return self.__class__(
                    self.t,
                    # add data
                    self.x + other.x,
                    # add stddev uncertainty in quadrature
                    np.sqrt(np.square(self.u) + np.square(other.u))
            )

        # array-like (any object that can create an array / any duck-type array)
        other = np.asanyarray(other)
        return self.__class__(self.t, self.x + other, self.u)

    def __sub__(self, o):
        return self + -o

    # element-wise comparison
    # object.__lt__(self, other)
    # object.__le__(self, other)
    # object.__eq__(self, other)
    # object.__ne__(self, other)
    # object.__gt__(self, other)
    # object.__ge__(self, other)

    # arithmetic
    # object.__add__(self, other)
    # object.__sub__(self, other)
    # object.__mul__(self, other)
    # object.__matmul__(self, other)
    # object.__truediv__(self, other)
    # object.__floordiv__(self, other)
    # object.__mod__(self, other)
    # object.__divmod__(self, other)
    # object.__pow__(self, other[, modulo])
    # object.__lshift__(self, other)
    # object.__rshift__(self, other)
    # object.__and__(self, other)
    # object.__xor__(self, other)
    # object.__or__(self, other)
    #
    #
    # object.__radd__(self, other)
    # object.__rsub__(self, other)
    # object.__rmul__(self, other)
    # object.__rmatmul__(self, other)
    # object.__rtruediv__(self, other)
    # object.__rfloordiv__(self, other)
    # object.__rmod__(self, other)
    # object.__rdivmod__(self, other)
    # object.__rpow__(self, other)
    # object.__rlshift__(self, other)
    # object.__rrshift__(self, other)
    # object.__rand__(self, other)
    # object.__rxor__(self, other)
    # object.__ror__(self, other)
    #
    #
    # object.__iadd__(self, other)
    # object.__isub__(self, other)
    # object.__imul__(self, other)
    # object.__imatmul__(self, other)
    # object.__itruediv__(self, other)
    # object.__ifloordiv__(self, other)
    # object.__imod__(self, other)
    # object.__ipow__(self, other[, modulo])
    # object.__ilshift__(self, other)
    # object.__irshift__(self, other)
    # object.__iand__(self, other)
    # object.__ixor__(self, other)
    # object.__ior__(self, other)

    # object.__round__(self[, ndigits])¶
    # object.__trunc__(self)
    # object.__floor__(self)
    # object.__ceil__(self)

    # def plot(self):
    #     ''


class MultiVariateTimeSeries:
    pass


if __name__ == '__main__':
    # tests
    z = np.random.randn(10)
    ts = TimeSeries(z)

    ts + ts
    42 * ts
