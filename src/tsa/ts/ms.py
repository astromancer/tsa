"""
Support for multivariate measurement sequences with optional time stamps and
uncertainties. Base for `TimeSeries` and `SpectralEstimate` classes.
"""


# std
import numbers as nr
import operator as op

# third-party
import numpy as np
from loguru import logger

# local
from recipes import api
from recipes.oo import slots
from recipes.flow import Emit
from recipes.logging import LoggingMixin
from recipes.oo.property import Alias, cached_property

# relative
from .. import io


# ---------------------------------------------------------------------------- #

def quadnorm(a):
    return np.ma.sqrt((a ** 2).sum())


def quadmean(a):
    return quadnorm(a) / (~a.mask).sum()


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

class UnivariateDescriptor:

    def __init__(self, kls):
        self.kls = kls

    def __get__(self, instance, kls):
        return self.kls

    def __set__(self, instance, value):

        if issubclass(value, MeasurementSequence):
            self.kls = value
            return

        # if value is None:
        #     from IPython import embed
        #     embed(header="Embedded interpreter at 'src/tsa/ts/ts.py':675")

        raise TypeError(
            f'Unvariate class for objects of type {type(self).__name__} should'
            ' inherit from `MeasuremnetSequence`.'
        )

    def __set_name__(self, owner, name):
        # set the class that uses this descriptor as `multivariate` attribute on
        # instance of univariate class
        if self.kls:
            logger.debug('Assigned {} as multivariate class of {!r}.',
                         owner, self.kls)
            self.multivariate = owner


class MultiVariate:
    # Mixin for simultaneous multivariate data

    univariate = UnivariateDescriptor(None)

    __repr__ = slots.Represent(['npoints', 'nvariates'], enclose='')

    def __init_subclass__(cls):
        bases = set(cls.__bases__) - {MultiVariate}

        if not bases:
            # direct inheritance of MultiVariate class. OK
            return

        for parent in set(cls.__bases__) - {MultiVariate}:
            if issubclass(parent, MeasurementSequence):
                # multiple inheritance, require univariate base class
                cls.univariate = parent
                parent.multivariate = cls
                return

        raise TypeError(f'No univariate counterpart to {cls}.')

    def __getitem__(self, key):
        if isinstance(key, int):
            key = (..., key)

        if not isinstance(key, tuple):
            return super().__getitem__(key)

        # select variate
        key, m = key
        data = self.value[key, m]
        dv, dd = self.values.ndim, data.ndim
        if dd == dv:
            kls = type(self)
        elif dd == dv - 1:
            kls = self.univariate
        else:
            kls = tuple

        if kls and isinstance(kls, type):
            return kls(None if self.index is None else self.index[key],
                       data,
                       None if self.sigma is None else self.sigma[key, m])

        raise TypeError(
            f'Invalid univariate class {kls.__name__} for multivariate '
            f'{type(self).__name__}.'
        )


# ---------------------------------------------------------------------------- #

class MeasurementSequence(LoggingMixin):
    """
    A basic univariate measurement sequence with optional uncertainties.
    Base class for `TimeSeries` and `SpectralEstimate` classes.
    """

    __repr__ = slots.Represent(['n_points'], enclose='')

    # expected dimensionality of value array for univariate data
    _base_object_dimensions = 1

    # ------------------------------------------------------------------------ #
    # Constructors

    # @classmethod
    # def fromfile(cls, filename):

    def __new__(cls, *args, **kws):
        *params, values, sigma = cls._parse_init_args(*args)

        if (nd := np.squeeze(values).ndim) > cls._base_object_dimensions:
            cls.logger.debug(
                'Creating multivariate object {!r}, since data dimensions ({}) '
                'larger than base object dimensions ({}).',
                cls.multivariate.__name__, nd, cls._base_object_dimensions
            )
            obj = super().__new__(cls.multivariate)
            # NOTE: init will not run automatically since this returns an object
            # of a different class
            cls.__init__(obj, *params, values, sigma)
            return obj

        return super().__new__(cls)

    # ------------------------------------------------------------------------ #
    def __init__(self, index, value=None, sigma=None, **metadata):
        """
        Create a MeasurementSequence object.

        Examples
        --------
        >>> MeasurementSequence(np.random.randn(100))

        """

        # times
        self._index = self._value = self._sigma = None
        self.value = value
        self.index = index  # must be after set_value
        self.sigma = sigma
        self.metadata = metadata

    @classmethod
    def _parse_init_args(cls, index, value=None, sigma=None):
        if value is None:
            # signals only
            value = index
            if isinstance(value, MeasurementSequence):
                return (value.index, value.x, value.sigma)
            return (None, value, sigma)

        # times & signals given
        return index, value, sigma

    # ------------------------------------------------------------------------ #
    # IO
    @classmethod
    def read(cls, filename, *_, **__):
        data, meta = io.read(filename)
        return cls(*data, **meta)

    def write(self, filename, **kws):

        data = self
        if io.SupportedFormats(filename).value == 'txt' and self.index is None:
            data = (np.arange(self.n), self.value, self.sigma)

        return io.write(filename, *data, **self.metadata, **kws)

    # aliases
    load = Alias('read')
    save = Alias('write')
    values = Alias('value')

    normalise = Alias('norlamize')

    n = n_points = Alias('npoints')
    m = n_variates = Alias('nvariates')

    # Time
    # ------------------------------------------------------------------------ #
    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index):
        if index is None:
            self._index = None  # t = np.arange(len(x))
            return

        index = np.asanyarray(index).squeeze()
        self._check_against_value(index, 'index')
        self._index = index

    # Data
    # ------------------------------------------------------------------------ #
    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        # set
        self._value = self._check_array(value)

        # delete cached stats
        del self.mean
        del self.var

    def _check_array(self, array):
        # make sure we have masked array
        array = np.ma.array(array, ndmin=2)
        ndmax = self._base_object_dimensions + 1
        if array.ndim > ndmax:
            raise ValueError(
                f'{type(self).__name__} data should be at most '
                f'{ndmax}-dimensional not {array.ndim}D.'
            )

        # make sure variate index in last position
        if 1 in array.shape and len(array) == 1:
            array = np.moveaxis(array, 0, -1)

        return array

    def _check_against_value(self, array, name):

        m = len(array)
        if m not in (shape := self.values.shape):
            raise ValueError(
                f'Unequal number of points between data ({shape=}) and {name} '
                f' arrays {array.shape}.'
            )

    @property
    def ndim(self):
        return self.values.ndim
    
    # Uncertainty
    # ------------------------------------------------------------------------ #
    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, sigma):
        if sigma is None:
            self._sigma = None
            return

        # check dimensionality
        sigma = self._check_array(sigma)

        # check valid values
        if np.any(bad := (sigma < 0)):
            raise ValueError(f'Cannot have negative uncertainties: {np.where(bad)}')

        # check shape same as values
        self._check_against_value(sigma, 'uncertainty')

        self._sigma = sigma

    # ------------------------------------------------------------------------ #
    @property
    def npoints(self):
        """Number of data points."""
        return len(self)

    @property
    def nvariates(self):
        """Number of variates (time series)."""
        return self._value.shape[-1]

    # ------------------------------------------------------------------------ #
    def __getitem__(self, key):
        data = self._value[key]
        kls = type(self) if len(data) else tuple
        return kls(None if self.index is None else self.index[key],
                   data,
                   None if self.u is None else self.sigma[key])

    def __len__(self):
        return len(self._value)

    def __iter__(self):
        """allow unpacking: `t, y, u = ts`"""
        yield from (self._index, self._value, self.sigma)

    # arithmetic
    # --------------------------------------------------------------------------
    def __array__(self, *args, **kws):
        return self._value

    def __pos__(self):
        return self

    def __neg__(self):
        # pylint: disable=invalid-unary-operand-type
        return self.__class__(self.index, -self.value, self.sigma)

    def __abs__(self):
        return self.__class__(self.index, abs(self.value), self.sigma)

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
        if isinstance(other, MeasurementSequence):
            # Can only really do time series if they are simultaneous
            if self.n != other.n:
                raise ValueError(f'Arithmetic on {self.__class__} objects with'
                                 f' different sizes not permitted')

            #  propagate uncertainties!

            if operator in {op.add, op.sub}:
                sigma = np.sqrt(self.sigma ** 2 + other.sigma ** 2)
            elif operator in {op.mul, op.truediv}:
                # assuming uncorrellated
                sigma = (self.value * other.value) * np.sqrt(
                    (self.sigma / self.value) ** 2 + (other.sigma / other.value) ** 2)

            return self.__class__(self.index, operator(self.value, other.value), sigma)

        # arithmetic with complex numbers not supported
        if isinstance(other, nr.Complex) and not isinstance(other, nr.Real):
            raise TypeError('Arithmetic with complex numbers not currently '
                            'supported.')
            # all other number types should be OK

        # array-like (any object that can create an array / any duck-type array)
        other = np.asanyarray(other)
        #  propagate uncertainties!
        sigma = self.sigma
        if operator in {op.mul, op.truediv}:
            # assuming uncorrellated
            sigma = operator(self.sigma, other)

        return self.__class__(self.index, operator(self.value, other), sigma)

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
    @cached_property
    def mean(self):
        # In standard statistical practice, ``ddof=1`` provides an unbiased
        # estimator of the variance of a hypothetical infinite population.
        # ``ddof=0`` provides a maximum likelihood estimate of the variance for
        # normally distributed variables.
        return self.value.mean(0)

    @cached_property(depends_on=mean)
    def var(self):
        # unbiased estimate of population variance
        return self.value.var(0, ddof=1)

    @cached_property(depends_on=var)
    def std(self):
        return np.sqrt(self.var)

    # ------------------------------------------------------------------------ #
    def copy(self):
        return type(self)(*self)

    def extend(self, other):

        if isinstance(other, tuple):
            other = type(self)(*other)

        self.value = np.vstack([self.value, other.value])

        if self.index is not None:
            self.index = np.hstack([self.index, other.index])

        if self.sigma is not None:
            self.sigma = np.vstack([self.sigma, other.sigma])

    def stack(self, others):
        value = np.vstack([self.value, *(o.value for o in others)])

        if (index := self.index) is not None:
            index = np.hstack([self.index, *(o.index for o in others)])

        if (sigma := self.sigma) is not None:
            sigma = np.vstack([self.sigma, *(o.sigma for o in others)])

        return type(self)(index, value, sigma)

    # Transformations
    # ------------------------------------------------------------------------ #
    @api.synonyms({'t0': 'start',
                   't(ime_)?scale': 'scale_index'})
    def normalize(self, loc='mean', scale='std', start=None, scale_index='ptp'):

        y = self.value
        v = self.sigma

        if loc not in {None, False}:
            loc = _resolve_stat(loc, self, 'mean')
            y = y - loc

        if scale:
            scale = _resolve_stat(scale, self, 'std')
            y = y / scale

            if v is not None:
                v = v / scale

        t = self.index
        if not (start is None or start is False):  # or (scale_index != 1)
            start = t[start] if isinstance(start, int) else float(start)
            t = t - start

        if scale_index:
            scale_index = _resolve_stat(scale_index, np.ma, 'ptp', self.index)
            t = t / float(scale_index)

        return type(self)(t, y, v)

    def compressed(self):
        if np.ma.is_masked(self.value):
            return self

        return self[self.value.mask.any(axis=-1)]

    def impute(self, n=10, method=np.ma.median, emit='silent'):
        s = n // 2
        x = self.value
        r, c = x.mask.nonzero()
        y = np.ma.empty(x.shape)
        u = None if self.sigma is None else np.ma.empty(self.sigma.shape)
        for i in range(self.m):
            bad = r[c == i]
            segments = list(map(slice, *(bad + [[-s], [s + 1]])))
            y[ok, i] = x[(ok := ~x[:, i].mask), i]
            y[bad, i] = [method(x[seg, i]) for seg in segments]

            if self.sigma is None:
                continue

            u[ok, i] = self.sigma[ok, i]
            u[bad, i] = [quadmean(self.sigma[seg, i]) for seg in segments]

        #
        Emit(emit)('Imputing masked data with sample mean from neighbourhood '
                   'n = {}.', n)

        return type(self)(self.index, y, u)
