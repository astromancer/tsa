import numpy as np

from tsa.spectral import periodogram


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
    """
    Variance of a real time series is equal the rms of the non-DC power
    spectrum
    """
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