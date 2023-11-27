"""
Various de-trending (low-pass filtering) methods.
"""

# TODO: change name of this
# look at scipy.sigtools.detrend

# third-party
import numpy as np

# relative
from .smoothing import smoother


ORDER_NAMES = ['mean', 'linear', 'quadratic', 'cubic', 'quartic', 'quintic']
NAMED_ORDERS = dict(map(reversed, enumerate(ORDER_NAMES)))


# TODO:
# SplineDetrend


# TODO
# def MA():
# Binomial filters
# exponential smooth
# Holt exponential smoother?


def resolve_detrend(method):
    # TODO: unify detrend & smoothing into filtering interface
    if method is None:
        return None, None, {}

    if isinstance(method, str):
        return method, None, {}

    if isinstance(method, tuple):
        # method, order = method
        return *method, {}

    if isinstance(method, dict):
        return method.pop('method', None), None, method

    raise ValueError('Detrend not understood')


def poly(y, n=1, t=None, preserve_energy=False):
    """
    Detrends the time series by fitting a polinomial of degree n and returning
    the fit residuals.
    """
    if n is None:
        return y

    if t is None:
        t = np.arange(len(y))
    coof = np.ma.polyfit(t, y, n)  # y may be masked!!

    yd = y - np.polyval(coof, t)

    # NOTE: is this sensible??
    if preserve_energy and n > 0:  # mean detrending inherently does not preserve energy
        P = (y**2).sum()
        Pd = (yd**2).sum()
        offset = np.sqrt((P - Pd) / len(yd))
        yd += offset
    return yd


def poly_uniform_bulk(Y, deg):
    """
    Detrends multiple time series by fitting a polinomial of degree n and returning
    the fit residuals for each.
    """
    _, k = Y.shape
    x = np.arange(k)
    Ydt = np.empty_like(Y)

    coef = np.polyfit(x, Y.T, deg)
    for i, c in enumerate(coef.T):  # TODO: there might be a faster way of doing this
        Ydt[i] = Y[i] - np.polyval(c, x)

    return Ydt


def smooth(x, wsize=11, window='hanning', fill=None, preserve_energy=False):
    """
    Detrends the time series by smoothing and returning the residuals.
    """
    s = smoother(x, wsize, window, fill, output_masked=None)
    return x - s


def detrend(x, method=None, n=None, t=None, **kws):
    """Convenience method for detrending (multiple) time series"""

    if method in (False, None):
        return x  # TODO: log warning?

    method = method.lower()
    if method in NAMED_ORDERS:
        n = NAMED_ORDERS.get(method)
        method = 'poly'

    if  method not in ('poly', 'smooth'):  # TODO: more methods
        raise ValueError('Unknown method: {}'.format(method))

    if method == 'poly':
        if t is None:
            return poly_uniform_bulk(x, n)
        else:
            # for loop?
            raise NotImplementedError
        # return poly(x, n, t)

    if method == 'smooth':
        return x - smoother(x, n)

        # for loop?
        # return smooth(x, **kws)

    if method == 'spline':
        from . import fold
        binsize = kws.get('binsize', 100)
        rbn = fold.rebin(x, binsize)


