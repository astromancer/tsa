# std
import numbers
import contextlib
import functools as ftl

# third-party
import numpy as np
import scipy as sp
import scipy.signal

# local
from recipes.string import Percentage


# ---------------------------------------------------------------------------- #
def windowed(a, window=None):
    """get window values + apply"""
    return a if (window is None) else a * get_window(window, a.shape[-1])


def get_window(window, n=None):
    """
    Return window values of window described by str `window' and length `n'.
    ...
    """
    if isinstance(window, (str, tuple)):
        if n is None:
            raise ValueError('Please specify window size `n`.')

        if window == 'hanning':
            window = 'hann'

        return sp.signal.get_window(window, n)

    # if window values are passed explicitly as a sequence of values
    if np.iterable(window):
        # if N given, assert that it matches the window length
        if n and len(window) != n:
            raise ValueError(
                f'Length {len(window)} of given window does not match '
                f'array length {n}.'
            )
        return window

    raise ValueError(f'Cannot make window from object: {window!r}.')


# ---------------------------------------------------------------------------- #

def resolve_size(size, n=None, dt=None):

    # overlap specified by percentage string eg: 99% or timescale eg: 60s
    if isinstance(size, str):
        assert n, 'Array size `n` required if `size` given as percentage (str).'

        # percentage
        if size.endswith('%'):
            return round(Percentage(size).of(n))

        return _size_from_unit_string(size, dt)

    if isinstance(size, float):
        if size < 1:
            assert n, 'Array size `n` required if `size` given as percentage (float).'
            return round(size * n)

        raise ValueError('Providing a float value for `size` is only valid if '
                         'that value is smaller than 1, in which case it is '
                         'interpreted as a fraction of the array size.')

    if isinstance(size, numbers.Integral):
        return size

    raise ValueError(
        f'Invalid size: {size!r}. This should be an integer, or a percentage '
        'of the array size as a string eg: "12.4%", or equivalently a float < 1'
        ' eg: 0.124, in which case the array size should be supplied. '
        'Finally, you may also provide the size in units of seconds eg: '
        '"30s", in whic case, the timestep `dt`, should also be provided.'
    )


def _size_from_unit_string(size, dt):
    if size.endswith('s'):
        return round(float(size.strip('s')) / dt)

    raise NotImplementedError


# ---------------------------------------------------------------------------- #

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

    winge = ftl.partial(scipy.signal.get_window, Nx=size)
    for w in windows:
        with contextlib.suppress(Exception):
            plt.plot(winge(w), label=w)

    plt.legend()
    plt.show()
