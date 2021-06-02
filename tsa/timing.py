import itertools as itt

import numpy as np
from scipy import stats

from recipes.lists import flatten


def get_delta_t(t, t_cyc=np.ma.masked):
    """
    Compute 1st order discrete difference (time steps). Set value of the
    first time step if given.

    Parameters
    ----------
    t
    t_cyc

    Returns
    -------

    """
    delta_t = np.ma.zeros(t.shape)  # empty_like
    delta_t[1:] = np.diff(t)
    # set the first value to the kinetic cycle time if known
    delta_t[0] = t_cyc
    return delta_t


def get_delta_t_mode(t):
    """Most commonly occurring time step"""
    return stats.mode(get_delta_t(t)).mode.item()


def summary(t, rtol=1e-5, atol=1e-8):
    """

    Parameters
    ----------
    t
    rtol
    atol

    Returns
    -------

    """
    deltas = np.diff(t)
    if np.allclose(deltas, deltas[0], rtol, atol):
        # constant time steps
        return deltas[0], deltas, ''
    
    # non-constant time steps!
    unqdt = np.unique(deltas)
    mode = stats.mode(deltas)
    dt = mode.mode
    if len(unqdt) > 5:
        info = f'{len(unqdt)} unique values between {deltas.min(), deltas.max()}'
    else:
        info = str(unqdt)

    return dt, unqdt, f'Non-constant time steps: {info}'


def detect_gaps(t, kct=None, ltol=1.9, utol=np.inf, tolerance='relative'):
    """
    Data gap detection based on the most common (mode) time delta value.


    Parameters
    ----------
    kct : float
        time delta value to use for gap detection.  If not given use mode instead.
    ltol : float
        lower detection tolerance - only flag gaps larger than ltol*kct
    utol : float
        upper detection tolerance - only flag gaps smaller than utol*kct
    tolerance : str {'abs', 'rel'}
        how the tolerance values are interpreted.
    """

    if np.ndim(t) > 1:
        t = np.squeeze(t)
        assert np.ndim(t) == 1, 'Array must be 1D'

    if utol is None:
        utol = np.inf

    # absolute time separation between successive values
    delta_t = np.abs(np.diff(t))

    if tolerance.lower().startswith('rel'):
        if kct is None:
            # use most frequently occuring time separation
            kct = stats.mode(delta_t)[0]

        ltol *= kct
        utol *= kct

    gap_idx, = np.where((delta_t > ltol) & (delta_t < utol))
    return gap_idx


def fill_array(x, indices, fillers):
    """intersperse the fillers and flatten to contiguous array."""
    # list with continuous sections of the original array
    cont_secs = np.split(x, indices + 1)
    # TODO: speed checks
    return np.array(flatten(itt.zip_longest(cont_secs, fillers, fillvalue=[])))


def fill_gaps(t, y, kct=None, mode='linear', option=None, fill=True,
              return_index=False):
    # NOTE THIS IS A HACK. Better to model the signal well enough so you
    # can generate data from model to fill gaps
    """ """

    if len(t) != len(y):
        raise ValueError(
            f'Input arrays must have same length. {len(t)=};{len(y)=}'
        )

    # if data is masked: remove masked values (treat as gaps)
    if np.ma.is_masked(y):
        t = t[~y.mask]
        y = y.compressed()

    # gap detection
    gap_idx = detect_gaps(t, kct)

    # fill gaps in original data
    # #return filler values only
    tfill, yfill = [], []
    idx = []

    for i in gap_idx:
        # to handel missing data (single missing point
        npoints = np.floor(round((t[i + 1] - t[i]) / kct, 6))
        # number of points that will be inserted

        t_fill = t[i] + np.arange(1, npoints) * kct
        # always fill time gaps with data at constant time step

        if mode == 'mean':
            mode = 'poly'
            option = (0, option)
            # option gives number of data points adjacent to gap used to fit
            # mean

        if mode.startswith('linear'):
            # option gives number of data values adjacent to gap to do the fit
            mode = 'poly'
            option = (1, 10 or option)
            # default to using 5 values on either side of gap for fitting

        if mode.startswith('poly'):
            # infer data using a polynomial
            if isinstance(option, int):
                n, k = option, 20
            else:
                n, k = option  # use k data values adjacent to gap to do the fit
                k = k or 20  # if k is None

            # n gives degree of polynomial
            i_l = max(0, i - k // 2)
            i_u = min(i + k // 2 + 1, len(t))
            coeff = np.polyfit(t[i_l:i_u], y[i_l:i_u], n)
            y_fill = np.polyval(coeff, t_fill)

        elif mode == 'spline':
            raise NotImplementedError
        else:
            if mode == 'constant':
                val = option  # option gives constant value to use
            if mode == 'edge':
                val = y[i + 1] if option == 'upper' else y[i]  
                # use upper of lower edge value

            if mode == 'median':
                val = np.median(y[i - option // 2:i + option // 2 + 1])
                # option gives number of data points adjacent to gap used to
                # calculate median

                y_fill = val * np.ones(npoints)

            if mode == 'random':
                k = option
                i_l = max(0, i - k // 2)
                i_u = min(i + k // 2 + 1, len(t))
                y_fill = np.random.choice(y[i_l:i_u], npoints - 1)

        # print( len(t_fill), len(y_fill) )

        tfill.append(t_fill)  # fill gaps in original data
        yfill.append(y_fill)

        if return_index:
            idx += list(range(i + 1, i + 1 + len(t_fill)))

    if fill:
        tfill = fill_array(t, gap_idx, tfill)
        yfill = fill_array(y, gap_idx, yfill)

    if return_index:
        return tfill, yfill, idx
    else:
        return tfill, yfill
