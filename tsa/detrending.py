'''
Various de-trending (low-pass filtering) methods.
'''

#TODO: change name of this
#look at scipy.sigtools.detrend

from .smoothing import smoother

import numpy as np

_named_orders = ['mean', 'linear', 'quadratic', 'cubic', 'quartic', 'quintic']
_odict = dict(zip(_named_orders, range(len(_named_orders))))

def poly(y, n=1, t=None, preserve_energy=False):
    '''
    Detrends the time series by fitting a polinomial of degree n and returning
    the fit residuals.
    '''
    if n is None:
        return y

    if t is None:
        t = np.arange(len(y))
    coof = np.ma.polyfit(t, y, n)               #y may be masked!!

    yd = y - np.polyval(coof, t)

    #NOTE: is this sensible??
    if preserve_energy and n > 0:  #mean detrending inherently does not preserve energy
        P = (y**2).sum()
        Pd = (yd**2).sum()
        offset = np.sqrt((P - Pd) / len(yd))
        yd += offset
    return yd


def poly_uniform_bulk(Y, deg):
    '''
    Detrends multiple time series by fitting a polinomial of degree n and returning
    the fit residuals for each.
    '''
    _, k = Y.shape
    x = np.arange(k)
    Ydt = np.empty_like(Y)

    coef = np.polyfit(x, Y.T, deg)
    for i, c in enumerate(coef.T):      #TODO: there might be a faster way of doing this
        Ydt[i] = Y[i] - np.polyval(c, x)

    return Ydt


def smooth(x, wsize=11, window='hanning', fill=None, preserve_energy=False):
    '''
    Detrends the time series by smoothing and returning the residuals.
    '''
    s = smoother(x, wsize, window, fill, output_masked=None)
    return x - s


def detrend(x, method=None, n=None, t=None, **kws):
    '''Convenience method for detrending (multiple) time series'''


    if method in (False, None):
        return x                    #TODO: log warning?

    method = method.lower()
    if method in _odict:
        n = _odict.get(method)
        method = 'poly'

    if not method in ('poly', 'smooth'):    #TODO: more methods
        raise ValueError('Unknown method: {}'.format(method))

    if method == 'poly':
        if t is None:
            return poly_uniform_bulk(x, n)
        else:
            #for loop?
            raise NotImplementedError
        #return poly(x, n, t)

    if method == 'smooth':
        raise NotImplementedError
        #for loop?
        #return smooth(x, **kws)

    if method == 'spline':
        from . import fold
        binsize = kws.get('binsize', 100)
        rbn = fold.rebin(x, binsize)



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    #polynomial detrend example
    N, k = 100, 10
    q, m, c = np.random.randn(3, k, 1)
    x = np.arange(N)
    Y = q*x*x + m * x + c * N

    Ydt = poly_uniform_bulk(Y, 2)

    fig, (ax1, ax2) = plt.subplots(2,1, sharey=True)
    ax1.plot(Y.T)
    ax2.plot(Ydt.T)


# TODO:
# SplineDetrend


#TODO
#def MA():
#Binomial filters
#exponential smooth
#Holt exponential smoother?