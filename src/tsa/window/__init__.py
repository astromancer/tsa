# std
import contextlib
import functools as ftl

# third-party
import numpy as np
import scipy.signal


# relative
from . import resolve
from .mwa import MovingWindowAnalysis


# ---------------------------------------------------------------------------- #

def product(a, window=None):
    """get window values for array, and multiply."""
    return a if (window is None) else a * get_window(window, a.shape[-1])


# aliases
windowed = product
get = get_window = get_array = resolve.array


# ---------------------------------------------------------------------------- #
# plot

def show_all_windows(cmap='gist_rainbow', size=1024):
    """
    plot all the spectral windows defined in scipy.signal (at least those that
    don't want a parameter argument.)
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    cm = plt.get_cmap(cmap)
    windows = scipy.signal.windows.__all__
    ax.set_color_cycle(cm(np.linspace(0, 1, len(windows))))

    get_window = ftl.partial(scipy.signal.get_window, Nx=size)
    for name in windows:
        with contextlib.suppress(Exception):
            plt.plot(get_window(name), label=name)

    plt.legend()
    plt.show()
