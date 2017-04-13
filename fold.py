import warnings

import numpy as np
from numpy.lib.stride_tricks import as_strided

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def fold(a, wsize, overlap=0, axis=0, **kw):
    """
    segment an array at given wsize, overlap,
    optionally applying a windowing function to each
    segment.

    keywords are passed to np.pad used to fill up the array to the required length.  This
    method works on multidimentional and masked array as well.

    keyword arguments are passed to np.pad to fill up the elements in the last window (default is
    symmetric padding).

    NOTE: When overlap is nonzero, the array returned by this function will have multiple entries
    **with the same memory location**.  Beware of this when doing inplace arithmetic operations.
    e.g.
    N, wsize, overlap = 100, 10, 5
    q = fold(np.arange(N), wsize, overlap )
    k = 0
    q[0,overlap+k] *= 10
    q[1,k] == q[0,overlap+k]  #is True
    """
    if a.size < wsize:
        warnings.warn('Window size larger than data size')
        return a[None]

    a, Nseg = padder(a, wsize, overlap, **kw)
    sa = get_strided_array(a, wsize, overlap, axis)

    # deal with masked data
    if np.ma.isMA(a):
        mask = a.mask
        if not mask is False:
            mask = get_strided_array(mask, wsize, overlap)
        sa = np.ma.array(sa, mask=mask)

    return sa


def rebin(x, binsize, t=None, e=None):
    '''
    Rebin time series data. Assumes data are evenly sampled in time (constant time steps).
    '''
    xrb = fold(x, binsize).mean(1)
    returns = (xrb,)

    if t is not None:
        trb = np.median(fold(t, binsize), 1)
        returns += (trb,)

    if e is not None:
        erb = np.sqrt(np.square(fold(e, binsize)).mean(1))
        returns += (erb,)

    if len(returns) == 1:
        return returns[0]
    return returns

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def gen(a, wsize, overlap=0, axis=0, **kw):
    """
    Generator version of fold.
    """
    a, Nseg = padder(a, wsize, overlap, **kw)
    step = wsize - overlap
    i = 0
    while i < Nseg:
        start = i * step
        stop = start + wsize
        ix = [slice(None)] * a.ndim
        ix[axis] = slice(start, stop)
        yield a[ix]
        i += 1


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def padder(a, wsize, overlap=0, axis=0, **kw):
    """ """
    assert wsize > 0, 'wsize > 0'
    assert overlap >= 0, 'overlap >= 0'
    assert overlap < wsize, 'overlap < wsize'

    mask = a.mask if np.ma.is_masked(a) else None
    a = np.asarray(a)  # convert to (un-masked) array
    N = a.shape[axis]
    step = wsize - overlap
    Nseg, leftover = divmod(N - overlap, step)

    if leftover:
        pad_mode = kw.pop('pad', 'mask')  # default is to mask the "out of array" values
        if pad_mode == 'mask' and (mask is None or mask is False):
            mask = np.zeros(a.shape, bool)

        pad_end = step - leftover
        pad_width = np.zeros((a.ndim, 2), int)  # initialise pad width indicator
        pad_width[axis, -1] = pad_end  # pad the array at the end with 'pad_end' number of values
        pad_width = list(map(tuple, pad_width))  # map to list of tuples

        # pad (apodise) the input signal (and mask)
        if pad_mode == 'mask':
            a = np.pad(a, pad_width, 'constant', constant_values=0)
            mask = np.pad(mask, pad_width, 'constant', constant_values=True)
        else:
            a = np.pad(a, pad_width, pad_mode, **kw)
            if mask not in (None, False):
                mask = np.pad(mask, pad_width, pad_mode, **kw)

    # convert back to masked array
    if not mask is None:
        a = np.ma.array(a, mask=mask)

    return a, int(Nseg)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_strided_array(a, size, overlap, axis=0):
    """
    Fold array `a` along `axis` with given `size` and `overlap`. Use strides (byte-steps) for memory efficiency
    By default, insert the new axis in the position before `axis`.
    """
    if axis < 0:
        axis += a.ndim
    step = size - overlap
    nsegs = (a.shape[axis] - overlap) // step  # number of segments

    new_shape = np.insert(a.shape, axis + 1, size)
    new_shape[axis] = nsegs

    # byte steps
    new_strides = np.insert(a.strides, axis, step * a.strides[axis])

    return as_strided(a, new_shape, new_strides)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_nocc(N, wsize, overlap):
    """
    Return an array of length N, with elements representing the number of
    times that the index corresponding to that element would be repeated in
    the strided array.
    """
    from recipes.list import count_repeats, sortmore

    I = fold(np.arange(N), wsize, overlap).ravel()
    if np.ma.is_masked(I):
        I = I[~I.mask]

    d = count_repeats(I)
    ix, noc = sortmore(*zip(*d.items()))
    return noc
