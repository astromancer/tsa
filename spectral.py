'''
Tools for spectral estimation
'''

# TODO: unit tests!!!

import functools
import warnings

import numpy as np
import scipy

from recipes.dict import AttrDict
from .gaps import fill_gaps, get_deltat_mode  # , windowed
from . import windowing, fold, detrending

from IPython import embed


# ====================================================================================================
def FFTpower(y, norm=0, detrend=()):
    '''Compute FFT power (aka periodogram). optionally normalize and or detrend'''

    y = detrending.detrend(y, *detrend)
    sp = abs(np.fft.rfft(y)) ** 2  # Power

    if norm:
        sp /= sp.sum()

    return sp


periodogram = FFTpower


# ====================================================================================================
def FFTpowers(data, detrend=None):
    '''
    Single-Sided Amplitude Spectrum of y(t). Multiprocessing implimentation
    NOTE: This assumes evenly sampled data!
    '''
    import multiprocessing as mp
    func = functools.partial(FFTpower, detrend=detrend)

    pool = mp.Pool()
    specs = pool.map(func, data)
    pool.close()
    pool.join()

    return np.array(specs)


# ====================================================================================================
# def cross_spectrum(signalA, signalB):


########################################################################################################################
class Spectral(object):
    '''Spectral estimation routines'''

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
                        pad=None,  # 'mean',     #effectively a 0 pad after mean de-trend...
                        gaps=None,
                        window='boxcar',
                        nwindow=None,
                        noverlap=0,
                        dt=None,
                        fs=None,
                        normalise='rms', )
    # translation
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

    # ====================================================================================================
    # def use_ls(self, opt):
    #     return opt.lower() in ('lomb-scargle', 'lombscargle', 'ls')
    #
    # def use_fft(self, opt):
    #     return opt.lower() in ('ft', 'fourier', 'fft')

    # ====================================================================================================
    def __init__(self, *args, **kws):  # TODO: this should be a call method
        # TODO: update docstring
        # otherwise sample spacing / frequency will suffice
        '''
        Compute frequency power spectrum or Time Frequency representation (TFR).

        Parameters
        ----------
        args    :
            (signal,)      - in which case t will be extrapolated (from dt if given)
            (t, signal)    - in which case
        signal  :       array-like
            values for which to compute power spectrum / TFR
        '''

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

        # clean masked, fill gaps etc

        # embed()
        # try:
        t, signal = self.prepare_signal(signal, t, self.dt)
        # except Exception as err:
            # embed()
            #raise err
        self.nwindow = resolve_nwindow(self.opts.nwindow, self.opts.split, t, self.dt)
        self.noverlap = resolve_overlap(self.nwindow, self.opts.noverlap)
        self.fRayleigh = 1. / (self.nwindow * self.dt)

        # fold
        self.t_seg, signal_seg = self.get_segments(signal, t, self.dt,
                                                   self.nwindow, self.noverlap)

        self.raw_seg = signal_seg
        # median time for each section
        self.tms = np.median(self.t_seg, 1)

        # pad, detrend, window
        self.segments = self.prepare_segments(signal_seg)

        # FFT frequencies
        nw = self.npadded or self.nwindow
        self.frq = np.fft.rfftfreq(nw, self.dt)
        self.ohm = 2. * np.pi * self.frq  # angular frequencies

        # calculate spectra
        self.power = self.main(self.segments)
        self.normed = self.opts.normalise

    # ====================================================================================================
    def translate(self, kws):
        nkws = {}
        for key, val in kws.items():
            if key in self.dictionary:
                key = self.dictionary[key]
            nkws[key.lower()] = val
        return nkws

    # ====================================================================================================
    def check(self, t, signal, **kws):
        '''Checks'''
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
                    borkmsg = ('Option %r not recognised for keyword %r. The following values '
                               'are allowed: %s')
                    raise ValueError(borkmsg % (kws[key], key, allowed_vals))

    # ====================================================================================================
    def check_timing(self, t, opts):

        dt = opts.dt
        fs = opts.get('fs')

        if (dt is None) and (fs is None) and (len(t) == 0):
            raise ValueError('Please provide one of the following: dt - sample time spacing,'
                             'fs - sampling frequency, t - time sequence')

        if fs and not dt:
            dt = 1. / fs

        if len(t) and (dt is None):  # sample spacing in time units
            Dt = np.diff(t)
            if np.allclose(Dt, Dt[0]):  # TODO: include tolerance value         # constant time steps
                dt = Dt[0]
            else:  # non-constant time steps!
                unqdt = np.unique(Dt)
                dt = get_deltat_mode(t)
                if len(unqdt) > 10:
                    info = '%i unique values between (%f, %f). ' % (len(unqdt), Dt.min(), Dt.max())
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

    # ====================================================================================================
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
                warnings.warn('Removing masked values from signal! This may not be a good idea...')
                t = t[~signal.mask]
        else:
            ''

        return t, signal

    # ====================================================================================================
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
                leftover = (len(t) - noverlap) % step
                end_time = t[-1] + dt * (step - leftover)
                t_seg = fold.fold(t, nwindow, noverlap,
                                  pad='linear_ramp',
                                  end_values=(end_time,))
        else:
            raise NotImplementedError
            # self.t_seg          = np.split(t, self.opts.split)
            # self.raw_seg        = np.split(signal, self.opts.split)

        return t_seg, signal_seg

    # ====================================================================================================
    def prepare_segments(self, segments):

        detrend_method, detrend_order, detrend_opt = resolve_detrend(self.opts.detrend)

        # conversion factor for dt to timing array passed to this function
        # conv_fact = {'s' : 1,
        # 'h' : 3600}[self.opts.timescale]

        # detrend
        segments = detrending.detrend(segments, detrend_order, **detrend_opt)

        # padding
        if self.opts.pad:
            npad, pad_method, pad_kws = resolve_padding(self.opts.pad, self.nwindow, self.dt)
            self.npadded = npad
            extra = npad - self.nwindow

            # this does pre- AND post padding # WARNING: does this mess with the phase??
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

    # ====================================================================================================
    def main(self, segments):  # calculate_spectra
        # calculate spectra
        spec = scipy.fftpack.fft(segments)
        spec = spec[..., :len(self.frq)]  # since we are dealing with real signals
        power = np.square(np.abs(spec))
        power = normaliser(power, self.segments, self.opts.normalise,
                           self.npadded, self.dt)
        return power

    # ====================================================================================================
    # def get_nfft(self, ):

    # ====================================================================================================
    def __iter__(self):
        '''enable use case: f, P = Spectral(t, s)'''
        return iter((self.frq, self.power))


# TODO: subclass for LS TFR
# TODO methods for non-uniform window length??
# TODO: functions for plotting segments etc...

# ====================================================================================================
def normaliser(power, segments, how, nwindow, dt):
    ''' '''
    if how is False:
        return power

    if how is True:
        how = 'rms'

    if not isinstance(how, str):
        raise ValueError

    how = how.lower()
    # NOTE: each segment will be normalized individually
    Nph = np.c_[segments.sum(1)]
    # Nph = signal.sum()                 #N_{\gamma} in Leahy 83 = DC component of FFT
    # N = len(signal)

    # NOTE: factor 2 below comes from the fact that the signal is real (one-sided)
    # However, we do not need to double the DC component, and in the case of even
    # number of frequencies, the last point (which is unpaired Nyquist freq)
    end = None if (nwindow % 2) else -1
    power[1:end] *= 2

    # FIXME: are you including the power of the window function?????
    scale = 1
    if how == 'leahy':
        scale = 1 / Nph

    T = nwindow * dt  # total time per segment #TODO: what if this changes per segment??
    if how == 'leahy density':
        scale = T / Nph

    if how == 'rms':
        scale = T / Nph ** 2

    return scale * power


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
            # NOTE: handeling this case complicates things because segments
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
    '''convert overlap to integer value'''
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
    known_methods = ('constant', 'mean', 'median', 'minimum', 'maximum', 'reflect', 'symmetric',
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
            raise ValueError('Total padded segment length %i cannot be smaller than nwindow %i'
                             '' % (size, nwindow))

        if not len(kws):
            kws = {}

        return size, method, kws
    else:
        raise ValueError('Padding needs to be a tuple containing 1) desired signal size (int)'
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
    '''
    plot all the spectral windows defined in scipy.signal (at least those that
    don't want a parameter argument.)
    '''
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
