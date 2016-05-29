'''
Various detrending (low-pass filtering) methods.
'''

import numpy as np

def poly(y, n=1, t=None, preserve_energy=False):
    '''
    Detrends the time series by fitting a polinomial of degree n and returning the fit residuals.
    '''
    if n is None:
        return y
    
    if t is None:
        t = np.arange(len(y))
    coof = np.ma.polyfit(t, y, n)               #y may be masked!!      
    
    yd = y - np.polyval(coof, t)
    
    if preserve_energy and n > 0:  #mean detrending inherently does not preserve energy
        P = (y**2).sum()
        Pd = (yd**2).sum()
        offset = np.sqrt((P - Pd) / len(yd))
        yd += offset
    return yd

from .tsa import smoother
def smooth(x, wsize=11, window='hanning', fill=None, preserve_energy=False):
    '''
    Detrends the time series by smoothing and returning the residuals.
    '''
    s = smoother(x, wsize, window, fill, output_masked=None)
    return x - s


def detrend(x, *args, **kws):
    '''Convenience method for detrending time series data'''
    if not len(args):
        #TODO: log warning?
        return x
    
    method, *args = args
    
    #TODO: methods: 'mean', 'linear', 'quadratic', 'cubic', 'quartic', 'quintic'
    
    if method in (False, None):
        #TODO: log warning?
        return x
    
    #TODO: improve this
    if not method.lower() in ('poly', 'smooth'):
        raise ValueError('Unknown smoothing method: {}'.format(x))
    
    f = eval(method.lower())    #HACK??
    return f(x,  *args, **kws)
        
    

#TODO
#def MA():
#Binomial filters
#Holt exponential smoother