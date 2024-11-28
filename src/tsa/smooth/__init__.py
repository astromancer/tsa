# std
import itertools as itt

# third-party
import numpy as np

# relative
from .. import window as wdw
from . import tvr


#alias
tv = tvr


# ---------------------------------------------------------------------------- #

class KernelSmoother:

    def __init__(self, window='hanning', size=11):
        self.window = window
        self.nwindow = size

    def __call__(self, x):
        x, wsize = self._check(x, self.nwindow)

        if wsize < 3:
            # should probably warn
            return x

        # get window values
        window = wdw.resolve.array(self.window, wsize)

        # pad array symmetrically at both ends
        s = np.ma.concatenate([x[wsize - 1:0:-1], x, x[-1:-wsize:-1]])

        # compute lower and upper indices to use such that input array dimensions
        # equal output array dimensions
        div, mod = divmod(wsize, 2)
        if mod:  # i.e. odd window length
            pl, ph = div, div + mod
        else:  # even window len
            pl = ph = div

        # convolve the signal
        # normalize window
        y = np.convolve(window / window.sum(), s, mode='valid')

        # return array that has same size as input array
        return y[pl:-ph + 1]

    def _check(self, x, wsize):

        x = np.asanyarray(x).squeeze()

        if np.ma.is_masked(x):
            raise ValueError('Smoother does not support working with masked data.')

        if x.ndim != 1:
            raise ValueError(f'{type(self).__name__} only accepts 1D arrays.')

        # resolve window size
        wsize = wdw.resolve.size(wsize, len(x))

        if x.size < wsize:
            raise ValueError('Input vector needs to be bigger than window size.')

        return x, wsize


def smooth(x, wsize=11, window='hanning', fill=None, output_masked=None):
    """
    Generic smoothing routine able to handle masked arrays.

    Parameters
    ----------
    x
    wsize
    window
    fill
    output_masked

    Returns
    -------

    """
    # TODO:  Docstring
    # TODO: smooth (filter) in timescale (use astropy.units?)

    # todo: compare astropy smoother ??

    if x.ndim != 1:
        raise ValueError('`smoother` only accepts 1D arrays.')

    if x.size < wsize:
        raise ValueError('Input vector needs to be bigger than window size.')

    if wsize < 3:
        # should probably warn
        return x

    # get the window values
    window = wdw.resolve.array(window, wsize)  # window values

    # pad array symmetrically at both ends
    s = np.ma.concatenate([x[wsize - 1:0:-1], x, x[-1:-wsize:-1]])

    # compute lower and upper indices to use such that input array dimensions
    # equal output array dimensions
    div, mod = divmod(wsize, 2)
    if mod:  # i.e. odd window length
        pl, ph = div, div + mod
    else:  # even window len
        pl = ph = div

    # replace masked values with mean / median.  They will be re-masked below
    if fill and np.ma.isMA(s):
        # s.mask = np.r_[ x.mask[wsize-1:0:-1], x.mask, x.mask[-1:-wsize:-1] ]
        wh = np.where(s.mask)[0]

        idxs = itt.starmap(slice, zip(wh - pl, wh + ph))
        func = getattr(np.ma, fill)  # TODO: error handeling
        fillmap = map(lambda idx: func(s[idx]), idxs)
        fillvals = np.fromiter(fillmap, float)
        s[s.mask] = fillvals

    # convolve the signal
    # normalize window
    y = np.convolve(window / window.sum(), s, mode='valid')

    # return
    if output_masked := (output_masked or np.ma.is_masked(x)):
        # re-mask values
        return np.ma.array(y[pl:-ph + 1], mask=x.mask)

    # return array that has same size as input array
    return y[pl:-ph + 1]


# alias
smoother = smooth
