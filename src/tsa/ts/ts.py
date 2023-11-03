"""
Time series objects
"""

# std
import numbers
import operator

# third-party
import numpy as np

# local
from recipes import api
from recipes.functionals import echo

# relative
from .plotting import CONFIG, TimeSeriesPlot


# import uncertainties.unumpy as unp  # linear uncertainty propagation
# since this library is tracking correlations between rvs operations like mean
# on an array is n^2 or n^3 which means its actually too slow to be useful.

# need class that tracks uncertainties but with correlation tracking
# optionally disabled
# Options: Linearized uncertainty propagation via `uncertainties`
#        : Full distribution calculation with rv arithmetic via `pacal`
# if _handle_uncertainties == 0:

# for now calculations on uncertainty has to be done manually which is a pain..


class TimeSeries:
    """
    A basic univariate time series with optional uncertainties.
    """

    # This class ties together the functionality of various libraries for
    # analysing time series data.

    # TODO
    # --------
    #  * automatic (linearized) uncertainty propagation (assumes normally
    #    distributed data)
    #  *
    #   interface with
    #    -statmodels
    #   - pandas
    #   - pacal
    #   - uncertainties
    #     ~ Approximate (linear) uncertainty propagation. Will only yield
    #       accurate results when used in computation if uncertainties are small
    #       compared to variation of the (non-linear) function through which it is
    #       mapped.

    #       note:
    #       "Error estimates for non-linear functions are biased on account of
    #       using a truncated series expansion. The extent of this bias depends
    #       on the nature of the function. For example, the bias on the error
    #       calculated for log(1+x) increases as x increases, since the expansion
    #       to x is a good approximation only when x is near zero."
    #       from: https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Caveats_and_warnings

    #  * spectral estimation techniques
    #  * sampled time series + visualizations.

    #  Not supported
    #  - units.  Make sure you use compatible units when doing arithmetic

    # @classmethod
    # def fromfile(cls, filename):

    @staticmethod
    def _parse_init_args(t_or_x, x=None, u=None):
        # signals only
        if x is None:
            x = t_or_x
            return (x.t, x.x, x.u) if isinstance(x, TimeSeries) else (None, x, u)

        # times & signals given
        return t_or_x, x, u

    def __new__(cls, *args, **kws):
        t, x, u = cls._parse_init_args(*args)

        if np.squeeze(x).ndim > 1:
            obj = super().__new__(MultiVariateTimeSeries)
            # init will not run automatically since this returns an object of a
            # different class
            obj.__init__(t, x, u)
            return obj

        return super().__new__(cls)

    def __init__(self, *args, **kws):
        """
        Create a TimeSeries object

        Examples
        --------
        >>> TimeSeries(np.random.randn(100))

        """
        t, x, u = self._parse_init_args(*args)

        # times
        self._t = self._x = self._u = None
        self.x = x
        self.t = t  # must be after set_x
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
        if t is None:
            self._t = None  # t = np.arange(len(x))
            return

        t = np.asanyarray(t).squeeze()
        self._check_against_x(t, 'time')
        self._t = t

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        # make sure we have masked array
        self._x = np.ma.array(x)
        # .squeeze()
        # if x.ndim != 1:
        #     raise ValueError(f'Time Series data should be 1D, not {x.ndim}')

        # self._x = np.ma.array(x)

    @property
    def u(self):
        return self._u

    @u.setter
    def u(self, u):
        if u is None:
            self._u = None
            return

        u = np.ma.array(u)
        self._check_against_x(u, 'uncertainty')
        if np.any(u < 0):
            raise ValueError('Cannot have negative uncertainties.')
        self._u = u

    def _check_against_x(self, vector, name):
        n, m = len(self), len(vector)
        if m != n:
            raise ValueError(
                f'Unequal number of points between data `x` ({n=}) and {name} `'
                f'{name[0]}` ({m=}) vectors.'
            )

    @property
    def n(self):
        """Number of data points."""
        return len(self)

    @property
    def m(self):
        """Number of variates (time series)."""
        return 1 if self.x.ndim == 1 else self.x.shape[1]

    # def fold(self, eph):

    # ------------------------------------------------------------------------ #

    def __repr__(self):
        return f'{type(self).__name__}(n={self.n:d})'  # .replace(',', ' ')

    def __getitem__(self, key):
        data = self.x[key]
        kls = TimeSeries if len(data) else echo
        return kls(None if self.t is None else self.t[key],
                   data,
                   None if self.u is None else self.u[key])

    # ------------------------------------------------------------------------ #
    def __len__(self):
        return len(self._x)

    def __iter__(self):
        """allow unpacking: `t, y, u = ts`"""
        yield from (self.t, self.x, self.u)

    def __pos__(self):
        return self

    def __neg__(self):
        # pylint: disable=invalid-unary-operand-type
        return self.__class__(self.t, -self.x)

    def __abs__(self):
        return self.__class__(self.t, abs(self.x))

    # arithmetic
    # --------------------------------------------------------------------------
    def _arithmetic(self, other, op):
        #
        if isinstance(other, TimeSeries):
            # Can only really do time series if they are simultaneous
            if self.n != other.n:
                raise ValueError(f'Arithmetic on {self.__class__} objects with'
                                 f' different sizes not permitted')

            # TODO: propagate uncertainties!
            return self.__class__(self.t, op(self.x, other.x), self.u)

        # arithmetic with complex numbers not supported
        if isinstance(other, numbers.Complex) and not isinstance(other, numbers.Real):
            raise TypeError('Arithmetic with complex numbers not currently '
                            'supported.')
            # all other number types should be OK

        # array-like (any object that can create an array / any duck-type array)
        other = np.asanyarray(other)
        return self.__class__(self.t, op(self._x, other), self.u)

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

    # ------------------------------------------------------------------------ #
    def append(self, ts):

        if isinstance(ts, tuple):
            ts = type(self)(*ts)

        self.x = np.hstack([self.x, ts.x])

        if self.t is not None:
            self.t = np.hstack([self.t, ts.t])

        if self.u is not None:
            self.u = np.hstack([self.u, ts.u])

    # ------------------------------------------------------------------------ #
    @api.synonyms({'(histogram)|(marginal)': 'hist'})
    def plot(self, ax=None, title='', hist=(), plims=CONFIG.plims, **kws):
        #
        tsp = TimeSeriesPlot(ax, title, hist, plims)
        tsp.plot(*self, **kws)
        tsp.ax.set(xlabel='Time (s)',
                   ylabel='Signal')
        return tsp

    def periodogram(self, window=None, detrend=None, pad=None, normalize=None):
        from tsa.spectral import Periodogram

        return Periodogram(self.t, self.x, window, detrend, pad, normalize)

    def spectrogram(self, nwindow, noverlap=0, window='hanning', detrend=None,
                    pad=None, split=None, normalize=False):
        from tsa.spectral import Spectrogram

        return Spectrogram(self.t, self.x,
                           nwindow, noverlap,
                           window, detrend,
                           pad, split, normalize)


class MultiVariateTimeSeries(TimeSeries):
    # support for simultaneous multivariate data

    # def decorrelate()

    def __repr__(self):
        return f'{type(self).__name__}(n={self.n:d}, m={self.m:d})'  # .replace(',', ' ')

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            return super().__getitem__(key)

        # select variate
        key, m = key
        data = self.x[key, m]
        kls = TimeSeries if len(data) else echo
        return kls(None if self.t is None else self.t[key],
                   data,
                   None if self.u is None else self.u[key, m])
