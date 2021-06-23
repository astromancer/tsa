
import ossaudiodev as sd

import scipy
from scipy.optimize import leastsq
import numpy as np
from IPython.display import Audio

from recipes.misc import is_interactive
from obstools.psf.model import Model

from .spectral import Spectral, normaliser



def rescale(data, interval=(-1, 1)):
    """Linearly rescale data to fall within given interval"""
    data = np.asarray(data)
    dmin, dmax = data.min(), data.max()
    imin, imax = sorted(interval)
    scale = np.ptp(interval) / (dmax - dmin)
    return (data - dmin) * scale + imin


def best_int_dtype(data):
    """get bit depth required to best represent float data as int"""
    d, r = divmod(np.log2(data.ptp()), 8)
    d = max(d, 1)
    i = (2 ** (int(np.log2(d)) + bool(r)))
    return np.dtype('i%d' % i)


def rescale_int(data, dtype=None):
    """Convert to integer array for saving as wav"""
    dtype = best_int_dtype(data) if dtype is None else np.dtype(dtype)
    if not isinstance(dtype.type(), np.integer):
        raise ValueError('Please give valid dtype')

    lims = np.iinfo(dtype)
    interval = (lims.min, lims.max)
    return rescale(data, interval).astype(dtype)


def monotone(f, duration=1, fs=44100):
    """A pure sinusoidal tone"""
    t = np.linspace(0, duration, fs * duration)
    return np.cos(2 * np.pi * f * t)

# def multitone(frqs, duration=1, fs=44100):


def play(signal, rate):
    if is_interactive():
        return Audio(data=signal, rate=rate, autoplay=True)

    with sd.open('w') as dev:
        dev.setfmt(sd.AFMT_S16_LE)
        dev.speed(rate)
        dev.writeall(signal)


class PianoKeys():
    """
    Simple class that returns the frequency of keys on the piano when sliced

    Example
    -------
    piano = PianoKeys()
    piano[40] # 261.625 #(middle C)
    piano['C4'] # 261.625 #(middle C)
    piano.to_name(40),
    piano.to_key_nr(piano[40]),
    piano.freq_to_name(piano[40]),
    piano['A0'],
    piano['C#1'],
    piano['B5']

    See:
    ----
    https://en.wikipedia.org/wiki/Piano_key_frequencies
    """
    A = 440  # Hz
    iA = 49  # key number for middle C

    notes = []
    for n in 'abcdefg'.upper():
        notes.append(n)
        if n not in 'BE':
            notes.append(n + '#')

    def to_freq(self, n):
        # https://en.wikipedia.org/wiki/Piano_key_frequencies
        if n < 1 or n > 88:
            raise ValueError('Key nr not in range')
        return self.A * pow(2, (n - self.iA) / 12)

    def to_key_nr(self, f):
        return int(12 * np.log2(f / self.A) + self.iA)

    def to_name(self, n):
        i = n % 12
        octave = (n + 8) // 12
        return self.notes[i - 1] + str(octave)

    def name_to_key(self, name):
        ix = 1 + ('#' in name)
        note = name[:ix].upper()
        if not note in self.notes:
            raise ValueError('Unrecognized note %s' % name)
        i = self.notes.index(note)

        octave = name[-1]
        if not octave.isdigit():
            octave = (i > 2)
        octave = int(octave)

        n = i + 1 + (octave * 12)
        n -= (n % 12 > 3) * 12
        return n

    def freq_to_name(self, f):
        return self.to_name(self.to_key_nr(f))

    def name_to_freq(self, name):
        return self.to_freq(self.name_to_key(name))

    def freq_of(self, key):
        if isinstance(key, (int, np.integer, float, np.floating)):
            return self.to_freq(key)
        elif isinstance(key, str):
            return self.name_to_freq(key)
        else:
            raise KeyError('Invalid key %s' % key)

    def __getitem__(self, key):
        return self.freq_of(key)

    def play(self, key, duration=1):
        """Produce a monotone signal at frequency of *key*"""
        signal = monotone(self.freq_of(key), duration)
        return Audio(data=signal, rate=44100, autoplay=True)


def FrequencyModulator(data, duration, fs=44.1e3, phase=0, fcarrier=None,
                       fdev=None):
    """
    data : information to be transmitted (i.e., the baseband signal)
    fcarrier : carrier's base frequency
    fdev : frequency deviation (represents the maximum shift away from the carrier frequency)"""

    t = np.linspace(0, duration, fs * duration)
    dmin, dmax = data.min(), data.max()

    if fcarrier is None:
        fcarrier = np.mean((dmin, dmax))
    if fdev is None:
        fdev = (np.max((dmin, dmax)) - fcarrier) / fs

    # normalize the data range from -1 to 1
    rescaled = rescale(data, (-1, 1))

    # generate FM signal:
    return np.cos(
        2 * np.pi * (fcarrier * t + fdev * np.cumsum(rescaled)) + phase)


class AudifySpec(Spectral):
    def main(self, segments):  # calculate_spectra
        # calculate spectra
        spec = scipy.fftpack.fft(segments)
        # since we are dealing with real signals
        spec = spec[..., :len(self.frq)]
        self.spectra = spec
        power = np.square(np.abs(spec))
        power = normaliser(power, self.segments, self.opts.normalise,
                           self.npadded, self.dt)
        return power

    def reconstruct_segment(self, i, duration, rate):

        n = int(duration * rate)
        ifft = scipy.fftpack.ifft(self.spectra[i], n)
