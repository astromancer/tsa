import numpy as np
import scipy as sp
import scipy.signal


def get_window(window, N=None):
    """
    Return window values of window described by str `window' and length `N'
    ...
    """
    if isinstance(window, (str, tuple)):
        if N is None:
            raise ValueError('Please specify window size N')
        return sp.signal.get_window(window, N)

    # if window values are passed explicitly as a sequence of values
    elif np.iterable(window):
        # if N given, assert that it matches the window length
        if N is not None:
            assert len(window) == N, ('length {} of given window does not match'
                                      'array length {}').format(len(window), N)
        return window
    else:
        raise ValueError('Cannot make window from %s' % window)


def windowed(a, window=None):
    """get window values + apply"""
    if window is not None:
        windowVals = get_window(window, a.shape[-1])
        return a * windowVals
    else:
        return a

