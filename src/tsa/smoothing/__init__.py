# std
import itertools as itt

# third-party
import numpy as np

# relative
from ..windowing import get_window


def smoother(x, wsize=11, window='hanning', fill=None, output_masked=None):
    # TODO:  Docstring
    # TODO: smooth (filter) in timescale (use astropy.units?)

    # todo: compare astropy smoother ??
    """
    Generic smoothing routine able to handle masked arrays

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

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < wsize:
        raise ValueError("Input vector needs to be bigger than window size.")

    if wsize < 3:
        return x

    # get the window values
    windowVals = get_window(window, wsize)  # window values

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
    w = windowVals / windowVals.sum()  # normalize window
    y = np.convolve(w, s, mode='valid')

    # return
    if output_masked := (output_masked or np.ma.is_masked(x)):
        # re-mask values
        return np.ma.array(y[pl:-ph + 1], mask=x.mask)

    # return array that has same size as input array
    return y[pl:-ph + 1]
