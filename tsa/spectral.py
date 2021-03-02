"""# TODO:
Tools for frequency spectral estimation (aka Fourier Analysis)
"""

# TODO: separate module
# TODO: subclass for LS TFR
# TODO methods for non-uniform window length??
# TODO: functions for plotting segments etc...
# TODO: unit tests!!!
# TODO: logging


import functools
import warnings

import numpy as np
import scipy
from recipes.dicts import AttrDict

from . import windowing, fold, detrending
from .gaps import fill_gaps, get_delta_t_mode  # , windowed


def periodogram(signal, norm=None, dt=None):
    """
    Compute FFT power (aka periodogram). optionally normalize and or detrend
    """
    pwr = abs(np.fft.rfft(signal)) ** 2  # Power
    return normaliser(pwr, signal, norm, dt)


def FFTpower(y, norm=0, detrend=()):
    """
    Compute FFT power (aka periodogram). optionally normalize and or detrend
    """

    y = detrending.detrend(y, *detrend)
    sp = abs(np.fft.rfft(y)) ** 2  # Power

    if norm:
        sp /= sp.sum()

    return sp


def FFTpowers(data, detrend=None):
    """
    Single-Sided Amplitude Spectrum of y(t). Multiprocessing implementation
    NOTE: This assumes evenly sampled data!
    """
    import multiprocessing as mp
    func = functools.partial(FFTpower, detrend=detrend)

    with mp.Pool() as pool:
        specs = pool.map(func, data)
    pool.join()

    return np.array(specs)


# ====================================================================================================
# def cross_spectrum(signalA, signalB):


########################################################################################################################
class Spectral(object):
    """
    Spectral estimation routines:
    Periodogram / spectrogram (DFT / STFT) with optional tapering, de-trending, padding, and gap-filling
    """

    allowed_vals = dict(  # use=('ls', 'fft'),
            timescale=('h', 's'),
            # pad=('constant', 'mean', 'median', 'minimum',
            #      'maximum', 'reflect', 'symmetric', 'wrap',
            #      'linear_ramp', 'edge'),
            normalise=(True, False, 'rms', 'leahy', 'leahy density'), )

    defaults = AttrDict(use='fft',
                        timescale='s',
                        split=None,
                        detrend=None,
                        pad=None,
                        # 'mean',     # effectively a 0 pad after mean de-trend...
                        gaps=None,
                        window='boxcar',
                        nwindow=None,
                        noverlap=0,
                        dt=None,
                        fs=None,
                        normalise='rms', )
    # translation #                           _alias_map
    dictionary = dict(apodize='window',
                      apodise='window',
                      taper='window',
                      # nfft='nwindow',
                      normalize='normalise',
                      norm='normalise',
                      overlap='noverlap',
                      nperseg='nwindow',
                      kct='dt',
                      sampling_frequency='fs')

    valdict = dict(hours='h', hour='h',
                   seconds='s', sec='s')

    @classmethod
    def translate(cls, kws):
        nkws = {}
        for key, val in kws.items():
            if key in cls.dictionary:
                key = cls.dictionary[key]
            nkws[key.lower()] = val
        return nkws

    # def use_ls(self, opt):
    #     return opt.lower() in ('lomb-scargle', 'lombscargle', 'ls')
    #
    # def use_fft(self, opt):
    #     return opt.lower() in ('ft', 'fourier', 'fft')

    def __init__(self, *args, **kws):
        # TODO: this should be a call method - initialize in __init__.py??
        # TODO: update docstring
        # otherwise sample spacing / frequency will suffice
        """
        Compute frequency power spectrum or Time Frequency representation (TFR).

        Parameters
        ----------
        args    :
            (signal,)      - in which case t will be extrapolated (from dt if given)
            (t, signal)    - in which case
        signal  :       array-like
            values for which to compute power spectrum / TFR
        """

        # If 'split' is a number, split the sequence into that number of roughly
        # equal portions. If split is a list, split the array according to the
        # indices in that list.

        *t, signal = args
        t = np.squeeze(t)
        # self.has_t = ~bool(len(t))
        # embed()

        # translate keywords & check
        opts = self.opts = self.defaults.copy()
        kws = self.translate(kws)
        self.check(t, signal, **kws)
        opts.update(kws)
        # self.opts = AttrDict(opts)

        # timing stats
        self.dt, self.fs = self.check_timing(t, self.opts)
        self.T = self.dt * len(signal)  # NOTE: assumes even sampling
        self.df = 1 / self.T

        # clean masked, fill gaps etc
        # try:
        t, signal = self.prepare_signal(signal, t, self.dt)
        # except Exception as err:
        # embed()
        # raise err
        self.nwindow = resolve_nwindow(self.opts.nwindow, self.opts.split, t,
                                       self.dt)
        self.noverlap = resolve_overlap(self.nwindow, self.opts.noverlap)
        self.fRayleigh = 1. / (self.nwindow * self.dt)

        # fold
        self.t_seg, signal_seg = self.get_segments(signal, t, self.dt,
                                                   self.nwindow, self.noverlap)
        self.n_seg = len(signal_seg)

        self.raw_seg = signal_seg
        # median time for each section
        self.tms = np.median(self.t_seg, 1)

        # pad, detrend, window
        self.segments = self.prepare_segments(signal_seg)

        # FFT frequencies
        nw = self.npadded or self.nwindow
        self.frq = np.fft.rfftfreq(nw, self.dt)
        self.ohm = 2. * np.pi * self.frq  # angular frequencies
        self.n_frq = len(self.frq)

        # calculate spectra
        self.power = self.main(self.segments)
        self.normed = self.opts.normalise

    def check(self, t, signal, **kws):
        """Checks"""
        if len(t) == 0:
            assert len(t) == len(signal)

        allowed_kws = self.defaults.keys()
        for key, val in kws.items():
            assert key in allowed_kws, 'Keyword %r not recognised' % key
            # Check acceptable keyword values
            val = self.valdict.get(val, val)
            if key in self.allowed_vals:
                allowed_vals = self.allowed_vals[key]
                if val not in allowed_vals:  # + (None, False)
                    borkmsg = (
                        'Option %r not recognised for keyword %r. The following values '
                        'are allowed: %s')
                    raise ValueError(borkmsg % (kws[key], key, allowed_vals))

    def check_timing(self, t, opts):  # TODO: as function...

        dt = opts.dt
        fs = opts.get('fs')

        if (dt is None) and (fs is None) and (len(t) == 0):
            raise ValueError(
                    'Please provide one of the following: dt - sample time spacing,'
                    'fs - sampling frequency, t - time sequence')

        if fs and not dt:
            dt = 1. / fs

        if len(t) and (dt is None):  # sample spacing in time units
            Dt = np.diff(t)
            if np.allclose(Dt, Dt[0]):  # TODO: include tolerance value        
                # constant time steps
                dt = Dt[0]
            else:  # non-constant time steps!
                from scipy.stats import mode

                unqdt = np.unique(Dt)  # Fixme: us mr!? mode, counts = mr
                np.diff(t)
                mr = mode(Dt)
                dt = mr.mode
                if len(unqdt) > 10:
                    info = '%i unique values between (%f, %f)' % (
                        len(unqdt), Dt.min(), Dt.max())
                else:
                    info = str(unqdt)
                msg = ('Non-constant time steps: %s. '
                       'Using time-step mode: %f for all further calculations.'
                       '' % (info, dt))
                warnings.warn(msg)
        else:
            ''  # TODO: check if dt same as implied by t??
        fs = 1. / dt
        return dt, fs

    def prepare_signal(self, signal, t, dt):

        is_masked = np.ma.is_masked(signal)
        # if is_masked:
        #     signal = signal[~signal.mask]

        # Fill data gaps
        # NOTE: we have to do this before allocating nwindow since len(t) might change.
        if len(t):
            if self.opts.gaps:
                fillmethod, option = self.opts.gaps
                t, signal = fill_gaps(t, signal, dt, fillmethod, option)

            elif is_masked:
                warnings.warn(
                        'Removing masked values from signal! This may not be a good idea...')
                t = t[~signal.mask]
        else:
            ''

        return t, signal

    def get_segments(self, signal, t, dt, nwindow, noverlap):
        # fold

        if nwindow:
            step = nwindow - noverlap
            signal_seg = fold.fold(signal, nwindow, noverlap)
            # padding will happen below for each section
            if not len(t):
                t_ = np.arange(nwindow) * dt
                tstep = np.arange(1, len(signal_seg) + 1) * step * dt
                t_seg = t_ + tstep[None].T
            else:
                # NOTE: unnecessary for uniform sample spacing
                # leftover = (len(t) - noverlap) % step
                # end_time = t[-1] + dt * (step - leftover)
                # t_seg = fold.fold(t, nwindow, noverlap,
                #                   pad='linear_ramp',
                #                   end_values=(end_time,))

                t_seg = fold.fold(t, nwindow, noverlap)


        else:
            raise NotImplementedError
            # self.t_seg          = np.split(t, self.opts.split)
            # self.raw_seg        = np.split(signal, self.opts.split)

        # embed()
        # assert t_seg.shape == signal.shape

        return t_seg, signal_seg

    def prepare_segments(self, segments):

        detrend_method, detrend_order, detrend_opt = \
            resolve_detrend(self.opts.detrend)

        # conversion factor for dt to timing array passed to this function
        # conv_fact = {'s' : 1,
        # 'h' : 3600}[self.opts.timescale]

        # detrend
        segments = detrending.detrend(segments, detrend_method, detrend_order,
                                      **detrend_opt)

        # padding
        if self.opts.pad:
            npad, pad_method, pad_kws = resolve_padding(self.opts.pad,
                                                        self.nwindow, self.dt)
            self.npadded = npad
            extra = npad - self.nwindow

            # this does pre- AND post padding
            #  WARNING: does this mess with the phase??
            div, mod = divmod(extra, 2)
            pad_width = ((0, 0), (div, div + mod))
            # pad_width = ((0, 0),(0, apodise - self.nwindow)
            segments = np.pad(segments, pad_width, mode=pad_method, **pad_kws)
        else:
            self.npadded = self.nwindow

        # apply windowing
        segments = windowing.windowed(segments, self.opts.window)
        # Nsegs = len(segments)

        return segments

    def main(self, segments):  # calculate_spectra
        # calculate spectra

        # NOTE: you can probs use the periodogram function here

        spec = scipy.fftpack.fft(segments)
        spec = spec[...,
               :len(self.frq)]  # since we are dealing with real signals
        power = np.square(np.abs(spec))
        power = normaliser(power, self.segments, self.opts.normalise, self.dt,
                           self.npadded)
        return power

    # def get_nfft(self, ):

    def __iter__(self):
        """enable use case: f, P = Spectral(t, s)"""
        return iter((self.frq, self.power))


def get_unit(how, unit=None):
    hz_1 = '/ Hz'  # $ '$Hz^{-1}$'
    density_unit = ' '.join(filter(None, (hz_1, unit)))
    units = {'rms': '(rms/mean)$^2$ %s) ' % hz_1,
             'leahy': unit,
             'pds': density_unit,
             'leahy density': density_unit
             }

    u = units.get(how, None)
    if not u:
        raise ValueError('')

    return u


# ====================================================================================================
def normaliser(power, segments, how=None, dt=None, nwindow=None, unit=None):
    """
    Normalise periodogram(s)

    see:
    Leahy 1983: http://adsabs.harvard.edu/full/1983ApJ...272..256L
    """

    # TODO: return unit

    if how is False:
        return power

    # NOTE: First We normalise the fft such that Parceval's theorem holds true.
    # The factor 2 below comes from the fact that the signal is real (one-sided)
    #  - we ignore half the points. However, we do not need to double the DC
    # component, and in the case of even number of frequencies, the last point
    # (which is unpaired Nyquist freq)
    if nwindow is None:
        nwindow = segments.shape[-1]

    end = None if (nwindow % 2) else -1
    power[1:end] *= 2
    # can check Parceval's theorem here

    if how is None:  # default
        return power

    if how is True:
        how = 'rms'

    if not isinstance(how, str):
        raise ValueError('Unknown normalisation %r requested' % how)

    how = how.lower()

    # NOTE: each segment will be normalized individually
    # Nph = signal.sum()
    # #N_{\gamma} in Leahy 83 = DC component of FFT

    Nph = np.c_[segments.sum(-1)]  #
    # N = segments.shape[1]def

    # print(Nph, nwindow, dt)

    # FIXME: are you including the power of the window function?????
    if how == 'leahy':
        return np.squeeze((2 / Nph) * power)

    if dt is None:
        raise ValueError('Require sampling time to normalise as density / rms')

    # TODO: can pass either T, df, or (n, dt)
    # total time per segment    #TODO: what if this changes per segment??
    T = nwindow * dt  # frequency step is 1/T

    if how in ('power density', 'pds'):
        return np.squeeze(T * power)

    if how == 'leahy density':
        return np.squeeze((2 * T / Nph) * power)

    if how == 'rms':
        return np.squeeze((2 * T / Nph ** 2) * power)

    raise ValueError('Unknown normalisation %r requested' % how)


normalizer = normaliser


# ====================================================================================================
def resolve_nwindow(nwindow, split, t, dt):
    if nwindow is None:
        if split is None:
            nwindow = len(t)  # No segmentation
        elif isinstance(split, int):
            nwindow = len(t) // split  # *split* number of segments
        else:  # if split values are passed explicitly as a sequence
            # split at specific indeces
            'check that split is the correct format'
            # NOTE: handling this case complicates things because segments
            # no longer be uniform length. Better to delegate this to case
            # separate mainloop
            raise NotImplementedError
    else:
        if isinstance(nwindow, str):
            if nwindow.endswith('s'):
                nwindow = round(float(nwindow.strip('s')) / dt)
            else:
                raise NotImplementedError
        else:
            'check that nwindow is integer'  # CAN BE HANDELED BY KW OPTIONS
    return nwindow


# ====================================================================================================
def resolve_overlap(nwindow, noverlap):
    """convert overlap to integer value"""
    if not bool(noverlap):
        return 0

    # overlap specified by percentage string eg: 99% or timescale eg: 60s
    if isinstance(noverlap, str):
        if noverlap.endswith('%'):
            frac = float(noverlap.strip('%')) / 100
            noverlap = frac * nwindow

    if isinstance(noverlap, float):
        # ISSUE WARNING??
        noverlap = round(noverlap)

    # negative overlap works like negative indexing! :)
    if noverlap < 0:
        noverlap += nwindow

    if noverlap == nwindow:
        noverlap -= 1  # Maximal overlap!
        warnings.warn('Specified overlap equals window size. Adjusting to '
                      'maximal overlap = %i' % noverlap)
    return noverlap


# ====================================================================================================
def resolve_padding(args, nwindow, dt):
    known_methods = (
        'constant', 'mean', 'median', 'minimum', 'maximum', 'reflect',
        'symmetric',
        'wrap', 'linear_ramp', 'edge')
    if isinstance(args, tuple):
        size, method, *kws = args
        assert method in known_methods
        if isinstance(size, str):
            if size.endswith('s'):
                size = round(float(size.strip('s')) / dt)
            elif size.endswith('%'):
                frac = float(size.strip('%')) / 100
                size = frac * nwindow
        size = round(size)
        if size < nwindow:
            raise ValueError(
                    'Total padded segment length %i cannot be smaller than nwindow %i'
                    '' % (size, nwindow))

        if not len(kws):
            kws = {}

        return size, method, kws
    else:
        raise ValueError(
                'Padding needs to be a tuple containing 1) desired signal size (int)'
                '2) padding method (str), 3) optional arguments for method (dict)')


# ====================================================================================================
def resolve_detrend(detrend):
    # TODO: unify detrend & smoothing into filtering interface
    if detrend is None:
        detrend_method, detrend_order, detrend_opt = None, None, {}

    elif isinstance(detrend, str):
        detrend_method, detrend_order, detrend_opt = detrend, None, {}

    elif isinstance(detrend, tuple):
        detrend_method, detrend_order = detrend
        detrend_opt = {}

    elif isinstance(detrend, dict):
        detrend_method = detrend.pop('method', None)
        detrend_order = None
        detrend_opt = detrend

    else:
        raise ValueError('Detrend kw not understood')

    return detrend_method, detrend_order, detrend_opt


# ====================================================================================================
def show_all_windows(cmap='gist_rainbow'):
    """
    plot all the spectral windows defined in scipy.signal (at least those that
    don't want a parameter argument.)
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    cm = plt.get_cmap(cmap)
    allwin = scipy.signal.windows.__all__
    colours = cm(np.linspace(0, 1, len(allwin)))
    ax.set_color_cycle(colours)

    winge = functools.partial(scipy.signal.get_window, Nx=1024)
    for w in allwin:
        try:
            plt.plot(winge(w), label=w)
        except:
            pass
    plt.legend()
    plt.show()


if __name__ == '__main__':

    def check_parceval(signal, periodogram):
        # Parceval's theorem
        tp_signal = np.square(signal).sum()
        tp_fft = periodogram.sum() / len(signal)
        return np.allclose(tp_signal, tp_fft)


    def check_DC(signal, periodogram):
        # DC component is squared signal sum
        ss = signal.sum()
        return (ss * ss) == periodogram[0]


    def check_var_rms_relation(signal, periodogram):
        """Variance of a real ts is equal the rms of the non-DC power spectrum"""
        n = len(signal)
        rms_pwr = periodogram[1:].sum() / (n * n)
        var = np.var(signal)
        return np.allclose(var, rms_pwr)


    # check parceval for even and odd signals
    off = 1e4
    N = 2 ** 10
    for n in [N, N - 1]:
        signal = np.random.randn(n) + off
        pwr = periodogram(signal)
        check_parceval(signal, pwr)
