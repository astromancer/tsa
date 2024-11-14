"""
Tools for Frequency Spectral Estimation (a.k.a. Fourier Analysis)
"""


# std
import textwrap as txw
import itertools as itt
from warnings import warn

# third-party
import scipy
import numpy as np
import matplotlib.pyplot as plt

# local
from recipes import api
from recipes.array import fold
from recipes.config import ConfigNode
from recipes.functionals import raises
from recipes.oo.slots import SlotHelper
from recipes.concurrence import Executor
from recipes.logging import LoggingMixin
from recipes.oo.represent import Represent
from recipes.oo.property import Alias, cached_property

# relative
from .. import timing, ts, detrend as dtr, window as wdw
from ..ts.ms import MeasurementSequence
from .tfr import TimeFrequencyRepresentation


# ---------------------------------------------------------------------------- #
#
CONFIG = ConfigNode.load_module(__file__)

NORMS = (None, True, False, 'rms', 'pds', 'leahy', 'leahy density')

PADDING = ('constant', 'mean', 'median', 'minimum', 'maximum', 'reflect',
           'symmetric', 'wrap', 'linear_ramp', 'edge')

# ---------------------------------------------------------------------------- #
# TODO: subclass for LS TFR
#   methods for non-uniform window length??
#   functions for plotting segments etc...
#   more unit tests!!!


def periodogram(signal, window=None, detrend=None, pad=None, norm=None, dt=1, **kws):
    """
    Compute FFT power (aka periodogram). optionally normalize and or detrend
    """
    return Periodogram.fit(signal, dt=dt, window=window, detrend=detrend,
                           pad=pad, norm=norm, **kws)


def pds(signal, dt=None):
    """
    Power density spectrum is a theoretical construct. This computes a
    periodogram (a psd estimate), normalized to have the same units as the true
    psd.

    Parameters
    ----------
    signal : [type]
        [description]
    dt : [type], optional
        [description], by default None

    Examples
    --------
    >>> 

    Returns
    -------
    [type]
        [description]
    """
    return periodogram(signal, dt, 'pds')


def fft_power(y, axis=0):
    """
    Compute FFT power (aka periodogram).
    """

    # Power
    return np.square(np.abs(scipy.fft.rfft(y, axis=axis, workers=-1)))


# def cross_spectrum(signalA, signalB):


def resolve_padding(nwindow, dt, args):
    if args is None:
        return nwindow, None, {}

    if isinstance(args, tuple):
        return _resolve_padding(nwindow, dt, args)

    raise ValueError(txw.dedent(
        '''Padding needs to be a tuple containing:
            1) desired size of output signal (int)
            2) padding method (str)
            3) optional keyword arguments for method (dict)
        '''))


def _resolve_padding(nwindow, dt, args):

    size, method, *kws = args
    assert method in PADDING

    size = wdw.resolve.size(size, nwindow, dt)

    if size < nwindow:
        raise ValueError(
            f'Total padded segment length {size} cannot be smaller than '
            f'nwindow {nwindow}'
        )

    kws, = kws or [{}]
    return size, method, kws


# ---------------------------------------------------------------------------- #

class SpectralEstimator(SlotHelper, LoggingMixin):
    """Base class for spectral density estimators"""

    def __call__(self, *args, **kws):
        return self.fit(*args, **kws)

    def frequencies(self, *args, **kws):
        """Compute frequency points"""
        raise NotImplementedError()

    def prepare(self, times, signal, *args, **kws):
        """Prepare times and signals for compute."""
        if times is None:
            times = range(len(signal))
        return times, signal

    def fit(self, *args, **kws):
        """Fit the data. Subclass to implement."""
        raise NotImplementedError()


class Spectrum(MeasurementSequence):
    """Base class representing an estimated spectrum and its uncertainty."""

    estimator = SpectralEstimator

    @classmethod
    def fit(cls, *args, **kws):
        """Fit the data, construct an instance of this class and return it."""
        estimator = cls.estimator(**kws)
        sde = cls(*estimator(*args))
        sde.estimator = estimator
        return sde


# ---------------------------------------------------------------------------- #

class UniformFFT(SpectralEstimator):
    """
    Base class for Fast Fourier Transform based spectral density estimators.
    """

    __slots__ = ('strict', )

    def __init__(self, strict=True):
        self.strict = bool(strict)

    def _fit(self, *args, dt=1, **kws):

        # use TimeSeries class to check and sanitize times / signals
        times, signal, sigma = ts.TimeSeries(*args)

        # check
        signal, dt = self._check_input(times, signal, dt)

        # prepare
        times, segments = self.prepare(times, signal, dt, **kws)

        # Compute frequencies
        frq = self.frequencies(times, dt)

        # calculate
        sde = self.compute(segments)

        return times, segments, frq, sde  # todo sigma

    def fit(self, *args, **kws):
        time, seg, frq, sde = self._fit(*args, **kws)
        return frq, sde

    def frequencies(self, times, dt):
        return np.fft.rfftfreq(len(times), dt)

    def compute(self, signal):
        return scipy.fft.rfft(signal, axis=0, workers=-1)

    def _check_input(self, times, signal, dt=None):
        emit = warn
        if np.ma.is_masked(signal):
            msg = (
                'Your signal contains masked data points. FFT-based spectral '
                'estimation methods are not appropriate for time series with '
                'non-constant time steps. You may wish to first interpolate the'
                ' missing points, although it is probably best to use an '
                'estimator which remains valid for non-constant time steps, '
                'such as such as the Lomb-Scargle periodogram.'
            )
            if self.strict:
                emit = raises(ValueError)
                msg += (
                    ' If you wish to proceed with the assumption of constant '
                    'timesteps, pass `strict = False`.\nThis'
                    ' message will then be emitted as a warning instead of '
                    'raising an exception.'
                )
            #
            emit(msg)

        # check timing
        if times is not None:
            # timestamp array
            times = np.squeeze(times)
            if len(times) != len(signal):
                raise ValueError('Timestamps and signal are unequally sized.')

            dt, _, msg = timing.summary(times)
            if msg:
                emit(f'Your timestamp array contains {msg}. The FFT-based '
                     f'methods is not applicable for time series with non-'
                     f'constant time steps.')
        elif not dt:
            # no timestamps
            raise ValueError(txw.dedent(
                '''Please provide one of the following:
                        t - sequence of time stamps
                        dt - constant sample time interval'''
            ))

        return np.array(signal), dt


class Normalizer:
    """
    Normalize power spectral density estimates.

    see:
    Leahy 1983: http://adsabs.harvard.edu/full/1983ApJ...272..256L
    """
    # FIXME: rms is a density unit!!!

    POWER_UNITS = {
        'rms':              '(rms/mean)$^2$ / Hz',  # '$Hz^{-1}$'
        'leahy':            '{}',
        'pds':              '{} / Hz',
        'leahy density':    '{} / Hz'
    }
    SYNONYMS = {'power density': 'pds', 'psd': 'pds'}

    def __init__(self, how=None):
        self._name = None
        self.name = how
        self.sde = None  # see: `__get__`

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        how = name or None
        if how is True:
            how = 'rms'

        if isinstance(how, str):
            how = how.lower()

        how = self.SYNONYMS.get(how, how)
        if how not in NORMS:
            raise ValueError(f'Unknown normalization: {how!r} ')

        if self._name != how:
            del self.scale
            self._name = how

        # return how

    def __get__(self, sde, kls=None):
        if sde:
            self.sde = sde

        return self

    def __set__(self, instance, how):
        # del self.name
        self.name = how
        if self.sde:
            # reset cached property
            del instance.power

    def __repr__(self):
        extra = ''
        if self.sde and self.name:
            extra = (
                f', scale={self.scale}'
                f', unit={self.POWER_UNITS.get(self.name, "")}')

        return f'{type(self).__name__}({self.name}{extra})'

    def __call__(self, power):
        if not self.name:
            return power

        # NOTE: each segment will be normalized individually
        # in Leahy 83
        #   N_{\gamma} = DC component of FFT
        #   N_{ph} = total

        return np.squeeze(self.scale * power)

    @cached_property  # (depends_on=name)
    def scale(self):
        if not self.name:
            return 1.

        # total time (in window): T = nwindow * dt  # frequency step is 1/T
        T = self.sde.T
        total = np.sqrt(self.sde.value[0])  # parceval

        if self.name == 'pds':
            return T

        if self.name == 'leahy':
            return 2 / total

        if self.name == 'leahy density':
            return 2 * T / total

        if self.name == 'rms':
            return 2 * T / total / total

        raise ValueError(f'Invalid norm: {self.name!r}')

    def get_power_unit(self, signal_unit=''):
        return self.POWER_UNITS.get(self.name, '{}').format(signal_unit or '')


class PowerSpectrum(Spectrum):
    """Estimate Fourier spectrum power components and their uncertainties."""

    frq = Alias('index')
    # power = Alias('value')

    norm = Normalizer()
    estimator = UniformFFT

    __repr__ = Represent(['n'], maybe=['norm.name'], rename={'norm.name': 'norm'})

    @api.synonyms({'norm(ali[sz]e)?':  'norm'})
    def __init__(self, frq, power, sigma=None, /, norm=False):
        super().__init__(frq, power, sigma)
        self.norm = norm


    @classmethod
    def fit(cls, *args, norm=None, **kws):
        """Fit the data and return and construct an instance of this class."""
        ps = super().fit(*args, **kws)
        ps.norm = norm
        return ps

    @cached_property()
    def power(self):
        return self.value * self.norm.scale

    # ------------------------------------------------------------------------ #
    # # IO
    # @classmethod
    # def read(cls, filename, *_, **__):
    #     frq, power, sigma = io.read(filename)

    #     from IPython import embed
    #     embed(header="Embedded interpreter at 'src/tsa/spectral/core.py':311")

    #     obj = object.__new__(cls)
    #     obj.__dict__.update()
    #     return obj

    # def write(self, filename, **kws):
    #     # if io.SupportedFileType.check(str(filename)) == 'npz':
    #     return io.write(filename, *self,
    #                     **{**CONFIG.io.txt.rename('columns', 'col_info'),
    #                        **kws})

    # ------------------------------------------------------------------------ #

    @property
    def nwindow(self):
        """
        Size of analysis window on signal. This is equal to the size of the 
        original signal for estimators that don't employ windowing.
        """
        return round((1. / self.f_nyquist) * (len(self.frq) - 1))

    @property
    def dt(self):
        """Sample time spacing"""
        # calculate sampling time from frequency array
        return self.T / self.nwindow

    @property
    def T(self):
        """Total signal duration"""
        return 1. / self.df

    @property
    def df(self):
        """Frequency step"""
        return np.diff(self.frq[:2])

    @property
    def f_nyquist(self):
        """Nyquist frequency"""
        return self.frq[-1]

    @property
    def omega(self):
        """Angular frequencies"""
        return 2. * np.pi * self.frq

    # alias
    ω = angular_frequency = omega

    # ------------------------------------------------------------------------ #

    def plot(self, ax=None, signal_unit=None, dc=False, **kws):
        if ax is None:
            fig, ax = plt.subplots()

        # dict(ls='-')
        # ignore DC component for plotting
        i = int(not dc)
        frq = self.frq[i:]
        lines = []
        for power in self.power[i:].T:
            lines.extend(ax.plot(frq, power, **kws))

        ax.set(xlabel=self.get_xlabel(),
               ylabel=self.get_ylabel(signal_unit),
               yscale='log')
        ax.grid()
        ax.figure.tight_layout()
        return ax.figure, ax

    def get_xlabel(self):
        return 'Frequency (Hz)'

    def get_ylabel(self, signal_unit=''):
        power_unit = self.norm.get_power_unit(signal_unit)
        if power_unit:
            power_unit = power_unit.join('()')

        name = self.norm.name
        density = name and (('density' in name) or (name == 'pds'))
        density = 'density ' * bool(density)
        return f'Power {density}{power_unit}'


# ---------------------------------------------------------------------------- #

class PowerSpectrumEstimator(UniformFFT):
    """Include optional detrending, padding, windowing and normalization"""

    __slots__ = ('window', 'detrend', 'pad')

    # @api.synonymns({
    #     'apodi[sz]e|taper': 'window',
    #     'norm(ali[sz]e)?':  'normalize',
    #     'overlap':          'noverlap',
    #     'kct':              'dt
    # })
    def __init__(self, window=None, detrend=None, pad=None, *, strict=True):

        # UniformFFT
        super().__init__(strict)

        self.window = window
        self.detrend = detrend
        self.pad = pad

    def prepare(self, times, signal, dt, **kws):

        # detrend
        method, params, kws = dtr.resolve(self.detrend)
        signal = dtr.detrend(signal, method, params, **kws)

        # padding
        if self.pad:
            n = len(self.signal)
            npad, method, kws = resolve_padding(n, dt, self.pad)
            extra = npad - len(signal)

            # this does pre- AND post padding
            #  WARNING: does this mess with the phase??
            div, mod = divmod(extra, 2)
            pad_width = ((0, 0), (div, div + mod))
            # pad_width = ((0, 0),(0, apodise - self.nwindow)
            signal = np.pad(signal, pad_width, mode=method, **kws)

        # apply windowing
        return super().prepare(times, wdw.product(signal, self.window))

    def compute(self, signal):
        # compute spectral power

        power = np.square(np.abs(super().compute(signal)))

        # NOTE: We normalise the fft such that Parceval's theorem holds true.
        # The factor 2 below comes from the fact that the signal is real
        # (one-sided) - we can ignore half the points since they are conjugate.
        # However, we do not need to double the DC component, and in the case of
        # even number of frequencies, the last point (which is the unpaired
        # Nyquist frequency)
        nwindow, *_ = signal.shape
        power[1:(-1, None)[nwindow % 2]] *= 2
        # can check Parceval's theorem here
        return power


class Periodogram(PowerSpectrum):

    estimator = PowerSpectrumEstimator


# ---------------------------------------------------------------------------- #

class STFT(PowerSpectrumEstimator):
    # TODO: use MovingWindowAnalysis?
    """
    Short-Time Fourier Transform as spectral density estimator. This computes a
    sequence of periodograms, aka the spectrogram.  Optional de-trending,
    tapering, window overlap, padding.
    """

    """
    Compute the spectrogram of a time series. Optional arguments allow for
    signal de-trending, padding, windowing (tapering).

    Parameters
    ----------
    args :
        (signal,) - in which case the sampling interval `dt` must be given.
        (t, signal) - in which case the sampling interval `dt` will be 
                        computed from the timestamps `t`.
    t : array-like
        The timestamps in seconds associated with the signal values.
    signal : array-like
        Data values for which to compute the STFT
    nwindow : int
        Size of the DFT window.
    noverlap : int or str, optional
        Number of overlapping points between subsequent windows. The size 
        of the overlap can also be specified as a percentage string
        eg: '50%'. Default is 0, implying no overlap between windows.
    split : int, optional
        Number of windows to split the signal into, by default None
    detrend : [type], optional
        Segment detrending algorithm, by default None
    pad : tuple, optional
        The (size, mode, kws) for the padding algorithm. `size` gives the
        final size of the padded segment. Similarly to `noverlap`, it can be
        specified as a percentage of `nwindow` or as a quantity string
        (number) with unit. By default `pad=None`, no padding of the signal
        is done.
    window : str, optional
        Name of the spectral window to use, by default 'hanning'
    dt : float, optional
        Sampling interval, by default None
    normalize : str, optional
        Normalization scheme for periodograms, by default 'rms'

    Examples
    --------
    >>>
    """

    def __init__(self, nwindow=None, noverlap=0, *args, split=None, **kws):
        kws.setdefault('window', 'hanning')
        super().__init__(*args, **kws)
        self.nwindow = nwindow
        self.noverlap = noverlap
        self.split = split

    def fit(self, *args, **kws):
        time, seg, frq, sde = self._fit(*args, **kws)
        return time, frq, sde

    def prepare(self, times, signal, dt, **kws):
        n = len(signal)

        nwindow = wdw.resolve.nwindow(self.nwindow, self.split, n, dt)
        noverlap = noverlap = wdw.resolve.overlap(nwindow, self.noverlap, dt)
        self.padding = self.npadded, *_ = resolve_padding(nwindow, dt, self.pad)

        # fold
        segments = fold.fold(signal, nwindow, noverlap)
        times = fold.fold(times, nwindow, noverlap)

        return times, segments


class Spectrogram(Periodogram):

    estimator = STFT
    _value_ndim_max = 3

    __repr__ = Represent(['n', 'nwindow'],
                         maybe=['norm.name'],
                         remap={'n':         'n_seg',
                                'norm.name': 'norm'})

    def __init__(self, times, frq, power, sigma=None, norm='rms'):
        self.times = times
        super().__init__(frq, power, sigma, norm=norm)

    @property
    def f_rayleigh(self):
        return 1. / (self.nwindow * self.dt)

    # alias
    fRayleigh = f_rayleigh

    @MeasurementSequence.index.setter
    def index(self, index):
        MeasurementSequence.index.fset(self, index)
        del self.tmid

    @cached_property
    def tmid(self):
        """Median time for each segment."""

        d, r = divmod(self.nwindow, 2)
        if r:
            # odd size window
            return np.mean(self.times[:, [d, d + 1]], 0)

        return self.times[:, d]

    def plot(self):
        return TimeFrequencyRepresentation(self)


# ---------------------------------------------------------------------------- #

class Correlogram(MeasurementSequence):

    def __init__(self, max_lag=None, method=None, njobs=-1):

        if method is None:
            method = 'direct' if np.ma.is_masked(self.x) else 'fft'
        else:
            method = str(method).lower()

        assert method in {'fft', 'direct'}

        max_lag = int(max_lag or self.n)
        lag = self.t[:max_lag] - self.t[0]

        self.logger.info('Computing Auto-correlation spectrum via {} method.',
                         method)
        sv = self.normalize()

        if method == 'direct':
            return type(self)(lag, _acf_direct(sv.x, max_lag, njobs).T)

        # FFT method
        x = sv.impute(emit='warning').x
        v = np.ma.empty((self.m, max_lag))
        for i, x in enumerate(x[(..., *[np.newaxis] * (self.m == 1))].T):
            c = scipy.signal.correlate(x, x, 'full')
            v[i] = c[self.n - 1:]

        # normalize
        norm = np.sum(x ** 2, 0, keepdims=True)
        return type(self)(lag, (v / norm).T)


class ACFDirectCompute(Executor):

    def compute(self, data, index, **kws):
        i, j = index
        r = _lag_acor_norm(data[i], j + 1)
        if not np.ma.is_masked(r):
            self.results[index] = r


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
