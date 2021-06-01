import numpy as np
import scipy as sp
import scipy.signal
import functools as ftl


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
    if np.iterable(window):
        # if N given, assert that it matches the window length
        if N is not None:
            assert len(window) == N, ('length {} of given window does not match'
                                      'array length {}').format(len(window), N)
        return window
    
    raise ValueError('Cannot make window from %s' % window)


def windowed(a, window=None):
    """get window values + apply"""
    if window is None:
        return a

    return a * get_window(window, a.shape[-1])


def show_all_windows(cmap='gist_rainbow'):
    """
    plot all the spectral windows defined in scipy.signal (at least those that
    don't want a parameter argument.)
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    cm = plt.get_cmap(cmap)
    windows = scipy.signal.windows.__all__
    ax.set_color_cycle(cm(np.linspace(0, 1, len(windows))))

    winge = ftl.partial(scipy.signal.get_window, Nx=1024)
    for w in windows:
        try:
            plt.plot(winge(w), label=w)
        except:
            pass
        
    plt.legend()
    plt.show()