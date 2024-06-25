# std
import numbers
from warnings import warn

# third-party
import numpy as np
import scipy as sp

# local
from recipes.string import Percentage


# ---------------------------------------------------------------------------- #
# Input resolution

def _size_from_unit_string(size, dt):
    if size.endswith('s'):
        return round(float(size.strip('s')) / dt)

    raise NotImplementedError


def array(window, n=None):
    """
    Return window values of window described by str `window' and length `n'.
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


def size(size, n=None, dt=None):

    # overlap specified by percentage string eg: 99% or timescale eg: 60s
    if isinstance(size, str):
        if not n:
            raise ValueError('Array size `n` required if `size` given as '
                             'percentage (str).')

        # percentage
        if size.endswith('%'):
            return round(Percentage(size).of(n))

        return _size_from_unit_string(size, dt)

    if isinstance(size, float):
        if size < 1:
            if n:
                return round(size * n)

            raise ValueError('Array size `n` required if `size` given as '
                             'percentage (float).')

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


def nwindow(nwindow, nsplit, n, dt):
    """
    Convert semantic `nwindow` value to integer

    Parameters
    ----------
    nwindow : int or str or None
        Input value for calculating window size. Can be ab int, in which case it
        is used directly, or a sting representing a percentage value of the full
        array size, in which case it is converted to an int value which is
        closest to the requested percentage size.
    nsplit : int or None
        Number of segments to use. Cannot be given simultaneously with *size*
        parameter.
    n : int
        Size of the array.
    dt : array-like
        Sample time (time step size).

    Returns
    -------
    int
        Size of the window.
    """
    if nwindow is None:
        return n if nsplit is None else n // int(nsplit)

    if isinstance(nwindow, str):
        return size(nwindow, n, dt)

    return int(nwindow)


def overlap(nwindow, noverlap, dt=None):
    """
    Convert semantic `noverlap` to integer value.

    Parameters
    ----------
    nwindow : [type]
        [description]
    noverlap : [type]
        [description]

    Examples
    --------
    >>> 

    Returns
    -------
    [type]
        [description]
    """
    noverlap = size(noverlap, nwindow, dt)

    if noverlap > nwindow:
        raise ValueError(f'Size cannot be larger than {noverlap} > {nwindow}')

    if noverlap == nwindow:
        noverlap -= 1  # Maximal overlap!
        warn('Specified overlap equals window size. Adjusting to '
             f'maximal {noverlap=}')

    # negative overlap works like negative indexing! :)
    if noverlap < 0:
        noverlap += nwindow

    return noverlap
