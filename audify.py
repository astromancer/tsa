import numpy as np


def rescale(data, interval=(-1, 1)):
    '''Linearly rescale data to fall within given interval'''
    data = np.asarray(data)
    dmin, dmax = data.min(), data.max()
    imin, imax = sorted(interval)
    scale = np.ptp(interval) / np.ptp(data)
    return (data - dmin) * scale + imin


def FrequencyModulator(data, duration, fs=44.1e3, phase=0, fcarrier=None, fdev=None):
    '''
    data : information to be transmitted (i.e., the baseband signal)
    fcarrier : carrier's base frequency
    fdev : frequency deviation (represents the maximum shift away from the carrier frequency)'''

    t = np.linspace(0, duration, fs * duration)
    dmin, dmax = data.min(), data.max()

    if fcarrier is None:
        fcarrier = np.mean((dmin, dmax))
    if fdev is None:
        fdev = (np.max((dmin, dmax)) - fcarrier) / fs

    # normalize the data range from -1 to 1
    rescaled = rescale(data, (-1, 1))

    # generate FM signal:
    return np.cos(2 * np.pi * (fcarrier * t + fdev * np.cumsum(rescaled)) + phase)


