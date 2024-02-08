"""
Time series objects
"""

# std
import numbers
import operator
import warnings
import itertools as itt

# third-party
import numpy as np
from scipy.signal import correlate

# local
from recipes.functionals import echo
from recipes.concurrent import Executor
from recipes.logging import LoggingMixin
from recipes.oo.property import CachedProperty

# relative
from .plotting import TimeSeriesPlot


# ---------------------------------------------------------------------------- #
# import uncertainties.unumpy as unp  # linear uncertainty propagation
# since this library is tracking correlations between rvs operations like mean
# on an array is n^2 or n^3 which means its actually too slow to be useful.

# need class that tracks uncertainties but with correlation tracking
# optionally disabled
# Options: Linearized uncertainty propagation via `uncertainties`
#        : Full distribution calculation with rv arithmetic via `pacal`
# if _handle_uncertainties == 0:

# for now calculations on uncertainty has to be done manually which is a pain..

# ---------------------------------------------------------------------------- #


class ACFDirect(Executor):
    def compute(self, data, index, **kws):
        i, j = index
        r = _lag_acor_norm(data[i], j + 1)
        if not np.ma.is_masked(r):
            self.results[index] = r
        # print(i, j, r)


def _lag_acor_norm(x, lag):
    """Lagged autocorrelation for standard normal variable"""
    n = len(x)
    return (x[:n - lag] * x[lag:]).sum(0) / n


def _acf_direct(x, max_lag, njobs=-1, backend='multiprocessing'):

    x = x[(..., *[np.newaxis] * (x.ndim == 1))].T
    indices = itt.product(range(len(x)), range(max_lag))

    task = ACFDirect(backend=backend)
    task.init_memory((len(x), max_lag))
    task.run(x, indices, njobs)

    return np.ma.MaskedArray(task.results, np.isnan(task.results))


# ---------------------------------------------------------------------------- #
class Smoothing(Interface):

    def __call__(self, x, wsize=11, window='hanning'):
        return KernelSmoother(window, wsize)(x)

    def tv(self, amount=None, nwindow=None, noverlap=0, njobs=-1):

        t, x, u = self.get_data(())
        # x = x[(..., *[np.newaxis] * (x.ndim == 1))].T
        y = np.empty_like(x)

        if nwindow:
            njobs = (njobs, )
            smoother = tv.WindowSmoother(nwindow, noverlap)
            name = 'tv.WindowSmoother'
        else:
            # no windowing. might bork for long ts
            njobs = ()
            smoother = tv.smooth
            name = 'tv.smooth'

        if (m := x.shape[1]) > 1:
            self.logger.debug('Looping over {} variates.', m)

        optima = []
        for i, xx in enumerate(x.T):
            smoother.jobname = f'{name} ({i + 1}/{m})'
            self.logger.debug('Running {} with njobs={} on {} array, λ = {}.',
                              smoother.jobname, njobs, xx.shape, amount)
            result = smoother(t, xx, amount, *njobs)

            if amount:
                y[:len(result), i] = result
            else:
                y[:len(result), i], optimum = result
                optima.append(optimum)

        if optima:
            'TODO: set as meta data'
        #     return y, np.reshape(optima, (-1, i + 1))

        return TimeSeries(t, y)


# ---------------------------------------------------------------------------- #

class TimeSeries(LoggingMixin):
    """
    A basic univariate time series with optional uncertainties.
    """

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

    # ------------------------------------------------------------------------ #
    plot = TimeSeriesPlot(xlabel='Time (s)',
                          ylabel='Signal')

    smooth = Smoothing()

    # ------------------------------------------------------------------------ #
    # Constructors

    # @classmethod
    # def fromfile(cls, filename):

    def __new__(cls, *args, **kws):
        t, x, u = cls._parse_init_args(*args)

        if np.squeeze(x).ndim > 1:
            obj = super().__new__(MultiVariateTimeSeries)
            # init will not run automatically since this returns an object of a
            # different class
            cls.__init__(obj, t, x, u)
            return obj

        return super().__new__(cls)

    # ------------------------------------------------------------------------ #
    #
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

        if self.m == 3:
            raise RuntimeError

    @staticmethod
    def _parse_init_args(t_or_x, x=None, u=None):
        # signals only
        if x is None:
            x = t_or_x
            return (x.t, x.x, x.u) if isinstance(x, TimeSeries) else (None, x, u)

        # times & signals given
        return t_or_x, x, u

    # Properties
    # ------------------------------------------------------------------------ #

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

        del self.mean
        del self.var

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

    # ------------------------------------------------------------------------ #

    def __repr__(self):
        return f'{type(self).__name__}(n={self.n:d})'  # .replace(',', ' ')

    def __getitem__(self, key):
        data = self.x[key]
        kls = TimeSeries if len(data) else echo
        return kls(None if self.t is None else self.t[key],
                   data,
                   None if self.u is None else self.u[key])

    #
    # ------------------------------------------------------------------------ #
    def __len__(self):
        return len(self._x)

    def __iter__(self):
        """allow unpacking: `t, y, u = ts`"""
        yield from (self.t, self.x, self.u)

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

    def __pos__(self):
        return self

    def __neg__(self):
        # pylint: disable=invalid-unary-operand-type
        return self.__class__(self.t, -self.x)

    def __abs__(self):
        return self.__class__(self.t, abs(self.x))

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
    @CachedProperty
    def mean(self):
        # In standard statistical practice, ``ddof=1`` provides an unbiased
        # estimator of the variance of a hypothetical infinite population.
        # ``ddof=0`` provides a maximum likelihood estimate of the variance for
        # normally distributed variables.
        return self.x.mean(0)

    @CachedProperty(depends_on=mean)
    def var(self):
        # unbiased estimate of population variance
        return self.x.var(0, ddof=1)

    @CachedProperty(depends_on=var)
    def std(self):
        return np.sqrt(self.var)

    # ------------------------------------------------------------------------ #

    def copy(self):
        return type(self)(*self)

    def append(self, ts):

        if isinstance(ts, tuple):
            ts = type(self)(*ts)

        self.x = np.hstack([self.x, ts.x])

        if self.t is not None:
            self.t = np.hstack([self.t, ts.t])

        if self.u is not None:
            self.u = np.hstack([self.u, ts.u])

    # ------------------------------------------------------------------------ #
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

    def correlogram(self, max_lag=None, method=None, njobs=-1):

        if method is None:
            method = 'direct' if np.ma.is_masked(self.x) else 'fft'
        else:
            method = str(method).lower()

        assert method in {'fft', 'direct'}

        max_lag = int(max_lag or self.n)
        top = max_lag
        t = self.t[:top] - self.t[0]

        self.logger.info('Computing Auto-correlation spectrum via {} method.', method)
        x = self.normalize().x
        
        if method == 'direct':
            return type(self)(t, _acf_direct(x, max_lag, njobs).T)
    
        # FFT method
        v = np.ma.empty((self.m, max_lag))
        for i, x in enumerate(x[(..., *[np.newaxis] * (self.m == 1))].T):
            if np.ma.is_masked(x):
                warnings.warn('Imputing masked data with sample mean.')
                y = x.filled(x.mean())

            c = correlate(y, y, 'full')
            v[i] = c[self.n - 1:]

        # normalize
        norm = np.sum(x ** 2, 0, keepdims=True)
        return type(self)(t, (v / norm).T)

    acf = correlogram

    def normalize(self, loc=True, scale=True):

        y = self.x
        v = self.u

        if loc:
            y = y - self.mean

        if scale:
            y = y / self.std

            if v is not None:
                v = v / self.std

        return type(self)(self.t, y, v)

    # def fold(self, eph):


class MultiVariateTimeSeries(TimeSeries):
    # support for simultaneous multivariate data

    # def decorrelate()

    def __repr__(self):
        # .replace(',', ' ')
        return f'{type(self).__name__}(n={self.n:d}, m={self.m:d})'

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


# alias
MultivariateTimeSeries = MultiVariateTimeSeries
