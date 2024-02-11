"""
Time series objects
"""

# std
import warnings
import numbers as nr
import operator as op
import itertools as itt

# third-party
import numpy as np
from scipy.signal import correlate

# local
from recipes.flow import Emit
from recipes.concurrent import Executor
from recipes.logging import LoggingMixin
from recipes.oo.property import CachedProperty

# relative
from ..smoothing import KernelSmoother, tv
from .interface import Interface
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

# def mean(x, w):
#     """Weighted Mean"""
#     return np.ma.average(x, w)

# def cov(x, y, w):
#     """Weighted Covariance"""
#     return np.ma.average((x - mean(x, w)) * (y - mean(y, w)), w)

# def corr(x, y, w):
#     """Weighted Correlation"""
#     return cov(x, y, w) / np.sqrt(cov(x, x, w) * cov(y, y, w))

# ---------------------------------------------------------------------------- #

def _resolve_stat(stat, obj, lookup, *args):
    if stat is True:
        stat = lookup

    if isinstance(stat, str):
        stat = getattr(obj, stat)

    if isinstance(stat, (nr.Number, np.ndarray)):
        return stat

    if callable(stat):
        return stat(*args)

    # 
    raise TypeError(
        f'Numeric input required for {lookup!r}, not {type(stat).__name__}.'
    )


# ---------------------------------------------------------------------------- #

class ACFDirectCompute(Executor):
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

    task = ACFDirectCompute(backend=backend)
    task.init_memory((len(x), max_lag))
    task.run(x, indices, njobs)

    return np.ma.MaskedArray(task.results, np.isnan(task.results))


def quadnorm(a):
    return np.ma.sqrt((a ** 2).sum())


def quadmean(a):
    return quadnorm(a) / (~a.mask).sum()


# ---------------------------------------------------------------------------- #
class Smoothing(Interface):

    def __call__(self, x, wsize=11, window='hanning'):
        return KernelSmoother(window, wsize)(x)

    def tv(self, smoothing=None, nwindow=None, noverlap=0, λ0=1, njobs=-1, **kws):

        t, x, u = self.get_data(())
        # x = x[(..., *[np.newaxis] * (x.ndim == 1))].T
        y = np.empty_like(x)

        if nwindow:
            njobs = (njobs, )
            smoother = tv.WindowSmoother(nwindow, noverlap, **kws)
            name = 'tv.WindowSmoother'
        else:
            # no windowing. might bork for long ts
            njobs = ()
            smoother = tv.smooth
            name = 'tv.smooth'

        if (m := x.shape[1]) > 1:
            self.logger.debug('Looping over {} variates.', m)

        self.optima = []
        for i, xx in enumerate(x.T):
            smoother.jobname = f'{name} ({i + 1}/{m})'
            self.logger.debug('Running {} with njobs={} on {} array, λ = {}.',
                              smoother.jobname, njobs, xx.shape, smoothing)
            result = smoother(t, xx, smoothing, λ0, *njobs)

            if smoothing:
                y[:len(result), i] = result
            else:
                result, optimum = result
                y[:len(result), i] = result
                self.optima.append(optimum)

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

    # Interfaces
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

    @staticmethod
    def _parse_init_args(t_or_x, x=None, u=None):
        # signals only
        if x is None:
            x = t_or_x
            return (x.t, x.x, x.u) if isinstance(x, TimeSeries) else (None, x, u)

        # times & signals given
        return t_or_x, x, u

    # Time
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

    # Data
    # ------------------------------------------------------------------------ #
    @property
    def x(self):
        return self._x  # .squeeze()

    @x.setter
    def x(self, x):
        # make sure we have masked array
        x = np.ma.array(x, ndmin=2)
        if (x.ndim < 1) | (x.ndim > 2):
            raise ValueError(f'Time Series data should be 1D or 2D '
                             f'(multivariate case) not {x.ndim}.')

        # make sure variate index in last position
        if 1 in x.shape and len(x) == 1:
            x = x.T

        self._x = x

        # delete cached stats
        del self.mean
        del self.var

    def _check_against_x(self, vector, name):
        n, m = len(self), len(vector)
        if m != n:
            raise ValueError(
                f'Unequal number of points between data `x` ({n=}) and {name} `'
                f'{name[0]}` ({m=}) vectors.'
            )

    # Uncertainty
    # ------------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------------ #
    @property
    def n(self):
        """Number of data points."""
        return len(self)

    @property
    def m(self):
        """Number of variates (time series)."""
        return self._x.shape[1]

    # ------------------------------------------------------------------------ #
    def __repr__(self):
        return f'{type(self).__name__}(n={self.n:d})'  # .replace(',', ' ')

    def __getitem__(self, key):
        data = self._x[key]
        kls = TimeSeries if len(data) else tuple
        return kls(None if self.t is None else self.t[key],
                   data,
                   None if self.u is None else self.u[key])

    # ------------------------------------------------------------------------ #
    def __len__(self):
        return len(self._x)

    def __iter__(self):
        """allow unpacking: `t, y, u = ts`"""
        yield from (self.t, self.x, self.u)

    # arithmetic
    # --------------------------------------------------------------------------
    def __array__(self, *args, **kws):
        return self._x

    def __pos__(self):
        return self

    def __neg__(self):
        # pylint: disable=invalid-unary-operand-type
        return self.__class__(self.t, -self.x, self.u)

    def __abs__(self):
        return self.__class__(self.t, abs(self.x), self.u)

    def __add__(self, other):
        return self._arithmetic(other, op.add)

    def __sub__(self, other):
        return self._arithmetic(other, op.sub)

    def __mul__(self, other):
        return self._arithmetic(other, op.mul)

    def __truediv__(self, other):
        return self._arithmetic(other, op.truediv)

    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__
    __rtruediv__ = __truediv__

    def _arithmetic(self, other, operator):
        #
        if isinstance(other, TimeSeries):
            # Can only really do time series if they are simultaneous
            if self.n != other.n:
                raise ValueError(f'Arithmetic on {self.__class__} objects with'
                                 f' different sizes not permitted')

            # TODO: propagate uncertainties!
            return self.__class__(self.t, operator(self.x, other.x), self.u)

        # arithmetic with complex numbers not supported
        if isinstance(other, nr.Complex) and not isinstance(other, nr.Real):
            raise TypeError('Arithmetic with complex numbers not currently '
                            'supported.')
            # all other number types should be OK

        # array-like (any object that can create an array / any duck-type array)
        other = np.asanyarray(other)
        warnings.warn('Uncertainties not propagated!')
        return self.__class__(self.t, operator(self.x, other), self.u)

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

    # Statistics
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

    def extend(self, ts):

        if isinstance(ts, tuple):
            ts = type(self)(*ts)

        self.x = np.hstack([self.x, ts.x])

        if self.t is not None:
            self.t = np.hstack([self.t, ts.t])

        if self.u is not None:
            self.u = np.hstack([self.u, ts.u])

    # Transformations
    # ------------------------------------------------------------------------ #
    def normalize(self, loc='mean', scale='std', t0=0, tscale='ptp'):

        y = self.x
        v = self.u

        if loc not in {None, False}:
            loc = _resolve_stat(loc, self, 'mean')
            y = y - loc

        if scale:
            scale = _resolve_stat(scale, self, 'std')
            y = y / scale

            if v is not None:
                v = v / scale

        t = self.t
        if not (t0 is None or t0 is False):  # or (tscale != 1)
            t0 = t[t0] if isinstance(t0, int) else float(t0)
            t = t - t0

        if tscale:
            tscale = _resolve_stat(tscale, np.ma, 'ptp', self.t)
            t = t / float(tscale)

        return type(self)(t, y, v)


    def compressed(self):
        if np.ma.is_masked(self.x):
            return self

        return self[self.x.mask.any(axis=-1)]

    def impute(self, n=10, method=np.ma.median, emit='silent'):
        s = n // 2
        x = self.x
        r, c = x.mask.nonzero()
        y = np.ma.empty(x.shape)
        u = None if self.u is None else np.ma.empty(self.u.shape)
        for i in range(self.m):
            bad = r[c == i]
            segments = list(map(slice, *(bad + [[-s], [s + 1]])))
            y[ok, i] = x[(ok := ~x[:, i].mask), i]
            y[bad, i] = [method(x[seg, i]) for seg in segments]

            if self.u is None:
                continue

            u[ok, i] = self.u[ok, i]
            u[bad, i] = [quadmean(self.u[seg, i]) for seg in segments]

        #
        Emit(emit)('Imputing masked data with sample mean from neighbourhood '
                   'n = {}.', n)

        return type(self)(self.t, y, u)

    # Spectral estimators
    # ------------------------------------------------------------------------ #

    def periodogram(self, window=None, detrend=None, pad=None, normalize=None, **kws):
        from tsa.spectral import Periodogram

        return Periodogram(self.t, self.x, window, detrend, pad, normalize, **kws)

    def spectrogram(self, nwindow, noverlap=0, window='hanning', detrend=None,
                    pad=None, split=None, normalize=False, **kws):
        from tsa.spectral import Spectrogram

        return Spectrogram(self.t, self.x,
                           nwindow, noverlap,
                           window, detrend,
                           pad, split, normalize, **kws)

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
        sv = self.normalize()

        if method == 'direct':
            return type(self)(t, _acf_direct(sv.x, max_lag, njobs).T)

        # FFT method
        x = sv.impute(emit='warning').x
        v = np.ma.empty((self.m, max_lag))
        for i, x in enumerate(x[(..., *[np.newaxis] * (self.m == 1))].T):
            c = correlate(x, x, 'full')
            v[i] = c[self.n - 1:]

        # normalize
        norm = np.sum(x ** 2, 0, keepdims=True)
        return type(self)(t, (v / norm).T)

    acf = correlogram

    # ------------------------------------------------------------------------ #
    # def fold(self, eph):


class MultiVariateTimeSeries(TimeSeries):
    # support for simultaneous multivariate data

    # def decorrelate()

    def __repr__(self):
        # .replace(',', ' ')
        return f'{type(self).__name__}(n={self.n:d}, m={self.m:d})'

    def __getitem__(self, key):
        if isinstance(key, int):
            key = (..., key)

        if not isinstance(key, tuple):
            return super().__getitem__(key)

        # select variate
        key, m = key
        data = self.x[key, m]
        kls = TimeSeries if len(data) else tuple
        return kls(None if self.t is None else self.t[key],
                   data,
                   None if self.u is None else self.u[key, m])

    # def __iter__(self):


# alias
MultivariateTimeSeries = MultiVariateTimeSeries
