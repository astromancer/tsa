"""
Tools for Frequency Spectral Estimation (aka Fourier Analysis)
"""


# std libs
import numbers
import textwrap as txw
import warnings as wrn
import functools as ftl

# third-party libs
import scipy
import numpy as np
import matplotlib.pyplot as plt

# local libs
from recipes.array import fold
from recipes.string import Percentage
from recipes.functionals import raises
from recipes.logging import logging, get_module_logger

# relative libs
from .. import windowing, detrending, timing
from ..ts import TimeSeries


# module level logger
logger = get_module_logger()
logging.basicConfig()
logger.setLevel(logging.INFO)


NORMS = (None, True, False, 'rms', 'pds', 'leahy', 'leahy density')
PADDING = ('constant', 'mean', 'median', 'minimum', 'maximum', 'reflect',
           'symmetric', 'wrap', 'linear_ramp', 'edge')


# TODO: subclass for LS TFR
#   methods for non-uniform window length??
#   functions for plotting segments etc...
#   more unit tests!!!


def periodogram(signal, dt=None, norm=None):
    """
    Compute FFT power (aka periodogram). optionally normalize and or detrend
    """
    # since we are dealing with real signals, spectrum is symmetric
    normalizer = Normalizer(norm, dt)
    return normalizer(FFTpower(signal), signal)


def pds(signal, dt=None):
    """
    Power density spectrum

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


def FFTpower(y):
    """
    Compute FFT power (aka periodogram).
    """

    # Power
    return np.square(np.abs(scipy.fft.rfft(y, workers=-1)))


# def cross_spectrum(signalA, signalB):


def resolve_nwindow(nwindow, split, n, dt):
    """
    Convert semantic `nwindow` value to integer

    Parameters
    ----------
    nwindow : int or str
        [description]
    split : [type]
        [description]
    t : [type]
        [description]
    dt : [type]
        [description]

    Examples
    --------
    >>> 

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    NotImplementedError
        [description]
    NotImplementedError
        [description]
    """
    if nwindow is None:
        if split is None:
            # No segmentation
            return n

        # *split* number of segments
        return n // int(split)

    if isinstance(nwindow, str):
        return _from_unit_string(nwindow, dt)

    return int(nwindow)


def convert_size(nwindow, size, dt, name):
    if not bool(size):
        return 0

    # overlap specified by percentage string eg: 99% or timescale eg: 60s
    if isinstance(size, str):
        # percentage
        if size.endswith('%'):
            size = round(Percentage(size).of(nwindow))
        # units
        else:
            size = _from_unit_string(size, dt)

    if isinstance(size, numbers.Real):
        return round(size)

    raise ValueError(f'Invalid value for {name}={size}')


def _from_unit_string(size, dt):
    if size.endswith('s'):
        return round(float(size.strip('s')) / dt)

    raise NotImplementedError


def resolve_overlap(nwindow, noverlap, dt=None):
    """
    Convert semantic `noverlap` to integer value.

    Parameters
    ----------
    nwindow : [type]
        [description]
    noverlap : [type]
        [description]

    Examples
    --------
    >>> 

    Returns
    -------
    [type]
        [description]
    """
    noverlap = convert_size(nwindow, noverlap, dt, 'noverlap')

    if noverlap > nwindow:
        raise ValueError(f'Size cannot be larger than {noverlap} > {nwindow}')

    if noverlap == nwindow:
        noverlap -= 1  # Maximal overlap!
        wrn.warn('Specified overlap equals window size. Adjusting to '
                 f'maximal {noverlap=}')

    # negative overlap works like negative indexing! :)
    if noverlap < 0:
        noverlap += nwindow

    return noverlap


def resolve_padding(args, nwindow, dt):
    if args is None:
        return nwindow, None, {}

    if isinstance(args, tuple):
        size, method, *kws = args
        assert method in PADDING

        size = convert_size(nwindow, size, dt, 'pad')

        if size < nwindow:
            raise ValueError(
                f'Total padded segment length {size} cannot be smaller than '
                f'nwindow {nwindow}'
            )

        kws, = kws or [{}]
        return size, method, kws

    raise ValueError(txw.dedent(
        '''Padding needs to be a tuple containing
            1) desired signal size (int)
            2) padding method (str)
            3) optional arguments for method (dict)
        '''))


# def prepare_signal(signal, t, dt, gaps):

#     is_masked = np.ma.is_masked(signal)
#     logger.info('Input time series contains masked data.')

#     # Interpolate missing data
#     # NOTE: have to do this before allocating nwindow since len(t) may change
#     if gaps:
#         fillmethod, option = gaps
#         t, signal = fill_gaps(t, signal, dt, fillmethod, option)

#     return t, signal




class Normalizer:
    """
    Normalise periodogram(s)

    see:
    Leahy 1983: http://adsabs.harvard.edu/full/1983ApJ...272..256L
    """
    # FIXME: rms is a density unit!!!
    POWER_UNITS = {'rms': '(rms/mean)$^2$ / Hz',  # '$Hz^{-1}$'
                   'leahy': '{}',
                   'pds': '{} / Hz',
                   'leahy density': '{} / Hz'}
    SYNONYMS = {'power density': 'pds'}

    def __init__(self, how=None, dt=None, signal_unit=''):

        if how is True:
            how = 'rms'

        if isinstance(how, str):
            how = how.lower()

        if how not in NORMS:
            raise ValueError('Unknown normalization %r requested' % how)

        if how and how.endswith(('density', 'pds', 'rms')) and (dt is None):
            raise ValueError(
                'Require sampling time interval to normalise spectrum as '
                'density / rms'
            )

        self.name = self.SYNONYMS.get(how, how)
        self.dt = dt

        # self.get_power_unit(signal_unit)

    def __call__(self, power, segments):
        if not self.name:
            return power

        # NOTE: First We normalise the fft such that Parceval's theorem holds
        # true. The factor 2 below comes from the fact that the signal is real
        # (one-sided) - we ignore half the points. However, we do not need to
        # double the DC component, and in the case of even number of
        # frequencies, the last point (which is unpaired Nyquist freq)
        nwindow = segments.shape[-1]
        end = None if (nwindow % 2) else -1
        power[1:end] *= 2
        # can check Parceval's theorem here

        # NOTE: each segment will be normalized individually
        # in Leahy 83
        #   N_{\gamma} = DC component of FFT
        #   N_{ph} = total_counts
        total_counts = segments.sum()

        # FIXME: are you including the power of the window function?????

        if self.name == 'leahy':
            return np.squeeze((2 / total_counts) * power)

        # total time per segment
        T = nwindow * self.dt  # frequency step is 1/T

        if self.name == 'pds':
            return np.squeeze(T * power)

        if self.name == 'leahy density':
            return np.squeeze((2 * T / total_counts) * power)

        if self.name == 'rms':
            return np.squeeze((2 * T / total_counts ** 2) * power)

        raise ValueError

    def get_power_unit(self, signal_unit=''):
        return self.POWER_UNITS.get(self.name, '{}').format(signal_unit or '')


# def check(self, t, signal, **kws):
#     """Checks"""
#     allowed_kws = self.defaults.keys()
#     for key, val in kws.items():
#         assert key in allowed_kws, 'Keyword %r not recognised' % key
#         # Check acceptable keyword values
#         val = self.valdict.get(val, val)
#         if key in self.allowed_vals:
#             allowed_vals = self.allowed_vals[key]
#             if val not in allowed_vals:  # + (None, False)
#                 borkmsg = (
#                     'Option %r not recognised for keyword %r. The following values '
#                     'are allowed: %s')
#                 raise ValueError(borkmsg % (kws[key], key, allowed_vals))


class FFTBase:
    """
    Base class for Fast Fourier Transform based spectral analysis
    """

    strict = True

    @classmethod
    def set_strict(cls, b=True):
        """
        Controls behaviour when receiving time stamp arrays that have
        non-constant time step intervals.
        """
        cls.strict = bool(b)

    def __init__(self, t_or_x, signal=None, normalize=None,  unit='',
                 /, dt=1):
        # use TimeSeries class to check and sanitize times / signals
        t, signal, _ = self._ts = TimeSeries(t_or_x, signal)
        dt, signal = self._check_input(signal, t, dt)

        self.signal = signal
        self.dt = dt
        self.T = self.dt * len(signal)
        self.df = 1 / self.T

        # normalization
        self.normalizer = Normalizer(normalize, dt, signal_unit=unit)

    @classmethod
    def _check_input(cls, signal, t, dt):
        emit = wrn.warn
        if np.ma.is_masked(signal):
            msg = ('Your signal contains masked data points. FFT-based spectral'
                   ' estimation methods are not appropriate for time series '
                   'with non-constant time steps. You may wish to first '
                   'interpolate the missing points, although it is probably '
                   'best to use an analysis technique, such as such as the '
                   'Lomb-Scargle periodogram, which is valid for non-constant '
                   'time steps. ')
            if cls.strict:
                emit = raises(ValueError)
                msg += ('If you wish to proceed with the assumption of constant'
                        f' timesteps, use \n >>> {cls}.set_strict(False).\nThis'
                        ' message will then be emitted as a warning instead of '
                        'rasing an exception')
            #
            emit(msg)

        # check timing
        if t is not None:
            # timestamp array
            t = np.squeeze(t)
            if len(t) != len(signal):
                raise ValueError('Timestamps and signal are unequally sized.')

            dt, _, msg = timing.summary(t)
            if msg:
                emit(f'Your timestamp array contains {msg}. The FFT-based '
                     f'methods is not applicable for time series with non-'
                     f'constant time steps.')
        elif not dt:
            # no timestamps
            raise ValueError(txw.dedent(
                '''Please provide one of the following:
                        t - sequence of time stamps
                        dt - constant sample time interval''')
            )

        return dt, np.array(signal)

    @property
    def omega(self):
        """angular frequencies"""
        return 2. * np.pi * self.frq

    def get_ylabel(self, signal_unit=''):
        norm = self.normalizer
        name = norm.name
        power_unit = norm.get_power_unit(signal_unit)
        if power_unit:
            power_unit = power_unit.join('()')
        density = name and (('density' in name) or (name == 'pds'))
        density = 'density ' * density
        return f'Power {density}{power_unit}'

    def get_xlabel(self):
        return 'Frequency (Hz)'


class Periodogram(FFTBase):

    def __init__(self,
                 t_or_x, signal=None,
                 window=None,
                 detrend=None,
                 pad=None,
                 normalize=None,
                 /, dt=1):

        FFTBase.__init__(self, t_or_x, signal, normalize, dt=dt)

        n = len(self.signal)
        self.padding = self.npadded, *_ = resolve_padding(pad, n, self.dt)

        # calculate periodograms
        self.power = self.compute(self.signal, detrend, pad, window)

    def __call__(self,  signal, detrend, pad, window):
        return self.compute(signal, detrend, pad, window)

    def __iter__(self):
        """enable use case: f, P = Spectral(t, s)"""
        return iter((self.frq, self.power))

    @property
    def frq(self):
        # FFT frequencies
        return np.fft.rfftfreq(self.npadded, self.dt)

    def prepare_signal(self, signal, detrend, pad, window):

        # detrend
        method, n, kws = detrending.resolve_detrend(detrend)
        signal = detrending.detrend(signal, method, n, **kws)

        # padding
        if pad:
            npad, method, kws = pad
            extra = npad - self.n

            # this does pre- AND post padding
            #  WARNING: does this mess with the phase??
            div, mod = divmod(extra, 2)
            pad_width = ((0, 0), (div, div + mod))
            # pad_width = ((0, 0),(0, apodise - self.nwindow)
            signal = np.pad(signal, pad_width, mode=method, **kws)

        # apply windowing
        return windowing.windowed(signal, window)

    def compute(self, signal, detrend, pad, window):

        signal = self.prepare_signal(signal, detrend, pad, window)

        # calculate periodograms
        return self.normalizer(FFTpower(signal), signal)

    def plot(self, ax=None, signal_unit=None, dc=False, **kws):
        if ax is None:
            fig, ax = plt.subplots()

        # dict(ls='-')
        # ignore DC component for plotting
        i = int(not dc)
        line, = ax.plot(self.frq[i:], self.power[i:], **kws)

        ax.set(xlabel=self.get_xlabel(),
               ylabel=self.get_ylabel(signal_unit))
        ax.grid()
        ax.figure.tight_layout()
        return line


# synonymns = dict(apodize='window',
#                   apodise='window',
#                   taper='window',
#                   # nfft='nwindow',
#                   normalize='normalise',
#                   norm='normalise',
#                   overlap='noverlap',
#                   nperseg='nwindow',
#                   kct='dt')

# valdict = dict(hours='h', hour='h',
#                seconds='s', sec='s')

# @classmethod
# def translate(cls, kws):
#     nkws = {}
#     for key, val in kws.items():
#         if key in cls.dictionary:
#             key = cls.dictionary[key]
#         nkws[key.lower()] = val
#     return nkws

# def use_ls(self, opt):
#     return opt.lower() in ('lomb-scargle', 'lombscargle', 'ls')
#
# def use_fft(self, opt):
#     return opt.lower() in ('ft', 'fourier', 'fft')


def get_segments(t, signal, nwindow, noverlap):
    # fold
    # if nwindow:
    # step = nwindow - noverlap
    segments = fold.fold(signal, nwindow, noverlap)
    t_seg = fold.fold(t, nwindow, noverlap)
    # padding will happen below for each section
    # t_ = np.arange(nwindow) * dt
    # tstep = np.arange(1, len(segments) + 1) * step * dt
    # t_seg = t_ + tstep[None].T
    return t_seg, segments

    # else:
    # NOTE: unnecessary for uniform sample spacing
    # leftover = (len(t) - noverlap) % step
    # end_time = t[-1] + dt * (step - leftover)
    # t_seg = fold.fold(t, nwindow, noverlap,
    #                   pad='linear_ramp',
    #                   end_values=(end_time,))
    # else:
    #     raise NotImplementedError
    # self.t_seg          = np.split(t, self.opts.split)
    # self.raw_seg        = np.split(signal, self.opts.split)

    # embed()
    # assert t_seg.shape == signal.shape

    # return t_seg, segments


class Spectrogram(Periodogram):
    """
    Spectral estimation routines:

    Periodogram / spectrogram (DFT / STFT) with optional tapering, de-trending, 
    padding, and imputation.
    """

    # @translate(synonymns) # translate keywords

    def __init__(self,
                 t_or_x, signal=None,
                 nwindow=None,
                 noverlap=0,
                 window='hanning',
                 detrend=None,
                 pad=None,
                 split=None,
                 normalize='rms',
                 /, dt=1):
        """
        Compute the spectrogram of a time series. Optional arguments allow for
        signal de-trending, padding (tapering).

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

        # super().__init__(*args, window, detrend, pad, dt,  normalize)

        FFTBase.__init__(self, t_or_x, signal, normalize, dt=dt)

        # t, signal = prepare_signal(signal, t, self.dt, gaps)
        n = len(self.signal)
        self.nwindow = nwindow = resolve_nwindow(nwindow, split, n, dt)
        self.noverlap = noverlap = resolve_overlap(nwindow, noverlap, dt)
        self.padding = self.npadded, *_ = \
            resolve_padding(pad, nwindow, self.dt)

        # fold
        # self.t_seg, segments = get_segments(
        #     self.signal, self.dt, nwindow, noverlap)
        segments = fold.fold(signal, nwindow, noverlap)
        self.t_seg = fold.fold(self._ts.t, nwindow, noverlap)

        # calculate periodograms
        self.power = self.compute(segments, detrend, pad, window)

        # self.n_seg = len(segments)
        # self.raw_seg = segments

        # pad, detrend, window
        # self.segments = self.prepare_signal(segments, detrend, pad, window)

        # # FFT frequencies
        # if pad:
        #     n = pad[0],
        # self.frq = np.fft.rfftfreq(n, dt)

        # # calculate periodograms
        # self.power = periodogram(self.segments, normalize, dt)
        # self.normed = normalize

    
    
    @property
    def fRayleigh(self):
        return 1. / (self.nwindow * self.dt)

    @ftl.cached_property
    def tmid(self):
        # median time for each section
        d, r = divmod(self.nwindow, 2)
        if r:
            # odd size window
            return np.mean(self.t_seg[:, [d, d + 1]], 0)

        return self.t_seg[:, d]

    def plot(self):
        from .tfr import TimeFrequencyRepresentation

        return TimeFrequencyRepresentation(self)
