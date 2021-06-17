from pathlib import Path
import numpy as np

from tsa.spectral import Periodogram
from tsa.spectral import TimeFrequencyRepresentation as TFR
import pytest
import re

np.random.seed(123)




def check_parceval(signal, periodogram):
    # Parceval's theorem
    tp_signal = np.square(signal).sum()
    tp_fft = periodogram.sum() / len(signal)
    return np.allclose(tp_signal, tp_fft)


def check_DC(signal, periodogram):
    # DC component is squared signal sum
    ss = signal.sum()
    return (ss * ss) == periodogram[0]


def check_var_rms(signal, periodogram):
    """
    Variance of a real time series is equal the rms of the non-DC power
    spectrum
    """
    n = len(signal)
    rms_pwr = periodogram[1:].sum() / n ** 2
    var = np.var(signal)
    return np.allclose(var, rms_pwr)


def random_signal(n, mean):
    return np.random.randn(n) + mean


@pytest.mark.parametrize(
    'signal',
    [random_signal(2**10, 1e4),
     random_signal(2**10 - 1, 1e4)]
)
def test_periodogram(signal):
    # check parceval for even and odd signals
    frq, pwr = Periodogram(signal)
    check_parceval(signal, pwr)
    check_DC(signal, pwr)
    check_var_rms(signal, pwr)


def test_tfr():
    # generate data
    n = int(1e4)        # number of points
    A = 5               # amplitude
    ω = 25              # angular frequency
    t = np.linspace(0, 2 * np.pi, n)
    signal = A * np.sin(ω * t)
    noise = np.random.randn(n)
    y = signal + noise

    tfr = TFR(t, y, )
