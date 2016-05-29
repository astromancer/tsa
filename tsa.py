import numpy as np
import numpy.linalg as la

import scipy as sp
import scipy.signal
import scipy.stats

import itertools as itt

from misc import flatten, accordion, lmap, count_repeats, sortmore

from IPython import embed

#====================================================================================================
def get_deltat(t, kct=None):
    '''compute 1st order discrete difference (time steps). set value of the
    first time step if known.'''
    deltat = t - np.roll(t,1)           #FIXME: np.diff is ~10 times faster
    #set the first value to the kinetic cycle time if known
    deltat[0] = kct
    
    return deltat

#====================================================================================================
def get_deltat_mode(t):
    '''Most commonly occuring time step'''
    return sp.stats.mode(get_deltat(t))[0][0]

#====================================================================================================
def get_window(window, N=None):
    '''
    Return window values of window described by str `window' and length `N'
    ...
    '''
    if isinstance(window, str):
        if N is None:
            raise ValueError( 'Please specify window size N' )
        return sp.signal.get_window( window, N )
    
    #if window values are passed explicitly as a sequence of values
    elif np.iterable(window):
        #if N given, assert that it matches the window length
        if not N is None:
            assert len(window) == N, ('length {} of given window does not match' 
                                      'array length {}').format(len(window), N)
        return window

#====================================================================================================
def detect_gaps(t, kct=None, ltol=1.9, utol=np.inf, tolerance='relative'):
    '''
    Data gap detection based on the most common (mode) time delta value.
    Parameters:
    ----------
    kct : float
        time delta value to use for gap detection.  If not given use mode instead.
    ltol : float
        lower detection tolerance - only flag gaps larger than ltol*kct
    utol : float
        upper detection tolerance - only flag gaps smaller than utol*kct
    tolerance : str {'abs', 'rel'}
        how the tolerance values are interpreted.
    '''
    
    if np.ndim(t) > 1:
        t = np.squeeze(t)
        assert np.ndim(t) == 1, 'Gap detection not supported for multidimentional arrays'
    
    if utol is None:
        utol = np.inf
    
    deltat = np.abs(np.diff(t))      #absolute time separation between successive values
    
    if tolerance.lower().startswith('rel'):
        if kct is None:
            #use most frequently occuring time separation
            kct = sp.stats.mode(deltat)[0]
            
        ltol *= kct
        utol *= kct
        
    gap_inds, = np.where((deltat > ltol) & (deltat < utol))
    return gap_inds

#====================================================================================================
#TODO: Investigate interpolate
#WARNING: DON'T FILL GAPS --- RATHER USE METHODS THAT DO NOT REQUIRE THIS!
def fill_gaps(t, y, kct=None, mode='linear', option=None, fill=True, ret_idx=False):     #ADD NOISE??
    ''' '''
    def make_filled_array(x, fillers):
        ''' intersperse the fillers and flatten to contiguous array. '''
        cont_secs = np.split(x, gap_inds+1)         #list with continuous sections of the original array
        #TODO: speed checks
        return np.array( flatten(itt.zip_longest( cont_secs, fillers, fillvalue=[] )) ) 
    
    assert len(t)==len(y), 'Input arrays must have same length. len(t)={}; len(y)={}'.format( len(t), len(y) )

    #if data is masked: remove masked values (treat as gaps)
    if np.ma.is_masked(y):
        t = t[~y.mask]
        y = y[~y.mask]

    #gap detection
    gap_inds = detect_gaps(t, kct)
    
    #fill gaps in original data                     #return filler values only
    Tfill, Yfill = [], []
    IDX = []
    
    for i in gap_inds:
        #to handel missing data (single missing point
        npoints = np.floor(round((t[i+1]-t[i]) / kct, 6))       #number of points that will be inserted
        t_fill = t[i] + np.arange(1, npoints) * kct      #always fill time gaps with data at constant time step
        
        if mode == 'mean':
            mode = 'poly'
            option = (0, option)                                     #option gives number of data points adjacent to gap used to fit mean

        if mode.startswith('linear'):                                #option gives number of data values adjacent to gap to do the fit
            mode = 'poly'
            option = (1, 10 or option)                                  #default to using 5 values on either side of gap for fitting

        if mode.startswith('poly'):                                        #interpolate data points using a polynomial
            if isinstance(option, int):
                n, k = option, 20
            else:
                n, k = option                                           #use k data values adjacent to gap to do the fit
                k = k   or 20           #if k is None

            i_l = max(0,i-k//2); i_u = min(i+k//2+1, len(t))
            coeff = np.polyfit(t[i_l:i_u], y[i_l:i_u], n)           #n gives degree of polinomial fit
            y_fill = np.polyval(coeff, t_fill)
            
        elif mode == 'spline':
            raise NotImplementedError
        else:
            if mode == 'constant':
                val = option                                        #option gives constant value to use
            if mode == 'edge':
                val = y[i+1] if option is 'upper' else y[i]         #use upper of lower edge value

            if mode == 'median':
                val = np.median(y[i-option//2:i+option//2+1])        #option gives number of data points adjacent to gap used to calculate median

                y_fill = val*np.ones( npoints )
            
            if mode == 'random':
                k = option
                i_l = max(0,i-k//2); i_u = min(i+k//2+1, len(t))
                y_fill = np.random.choice( y[i_l:i_u], npoints-1 )
        
        #print( len(t_fill), len(y_fill) )
        
        Tfill.append(t_fill)                                          #fill gaps in original data
        Yfill.append(y_fill)
        
        if ret_idx:
            IDX += list(range(i+1, i+1+len(t_fill)) )
    
    if fill:
        Tfill = make_filled_array( t, Tfill )
        Yfill = make_filled_array( y, Yfill )

    if ret_idx:
        return Tfill, Yfill, IDX
    else:
        return Tfill, Yfill

         
#====================================================================================================        
def rolling_var(data, wsize, overlap, min_points=2, center=False, samesize=False, ddof=0):
    #NOTE:  might be easier to just use pandas.rolling_var...
    '''   
    (Unbiased) moving variance.
    Parameters
    ----------
    wsize : int
        Size of the moving window. This is the number of observations used for
        calculating the statistic.
    overlap : int
        Size of the overlap between consecutive windows.
    min_points : int, default 2
        Minimum number of observations in window required to have a value
        (otherwise result is masked).
    center : boolean, default False
        Whether to center the window on the observation.
        if True - the window will be centered on the data point - i.e. data point preceding and
        following the observation data point will be used to calculate the statistic. The data
        point on which the window is centered will depend on the values of wsize and overlap.
        In this case the array is prepended and appended by k zeros where k is the maximum between step 
        and overlap.  In this way the first window contains at least overlap unique 
        data points and is centered on either the middle of these unique elements, or the first
        data point in the array, depending on the overlap.
        For overlap>~wsize/2, center=True will only affect the calculation for the first window.
        
        if False - right window edge is aligned with data points - i.e. only the preceding values 
        are used.
        
    samesize : boolean, default False
        Whether the returned array should have the same size as the input array
    ddof : int, default 0
        "Delta Degrees of Freedom": the divisor used in the calculation is
        ``N - ddof``, where ``N`` represents the number of elements.
        

    Returns
    -------
    v : array (possibly masked)
        Variance in each window

    '''

    assert( wsize > 0 )
    assert( overlap >= 0 )
    assert( overlap < wsize )    

    step = wsize - overlap
    N = data.shape[-1]
    
    #print( 'N', N )  
    #print( 'wsize', wsize )
    #print( 'overlap', overlap )
    #print( 'step', step )

    #if min_points is None:
    #    min_points = 0.25*wsize

    nans = np.isnan(data)
    if nans.any():
        data = np.ma.masked_where( nans, data )
        
    if center:
        #pl = max(0, div)  #-mod #will shift the first non-zero value one left if wsize is even, favouring data points in the first window above zeros
        #print( 'step>overlap', step>overlap )
        #print( 'pl', pl )
        #pl = sh    #min(step, sh)
        pw = max(step, overlap)
        #print( 'max(step,overlap)', pw )
        Nseg, leftover = divmod(N-overlap, step)
        #print( 'Nseg, leftover', Nseg, leftover )
        #print( 'step-leftover (end zeros)', step-leftover )
        k = overlap//step if overlap > step else 1
        
        #if leftover:
        #    pw += (step-leftover)
        pw = k*step-leftover
        div, mod = divmod(pw, 2)
        #print( 'divmod(pw,2)', div, mod ) 
        #k = (N+pl-Nseg*step)//step #(step+leftover) // step               #how many steps should still be taken
        
        
        #print( 'step-leftover (end zeros)', step-leftover )
        #print( '(N+pl-Nseg*step)', (N+pl-Nseg*step) ) #(div+mod+leftover)
        #print( 'k', k )     
        #ph = max(0, k*step-leftover)
        pl = div
        ph = div+mod
        #print( 'pl, ph', pl, ph )    
        
        data = np.ma.concatenate( [np.zeros(pl), data, np.zeros(ph)] )  #pad data array with zeros on both sides
        
        if np.ma.is_masked(data):
            z = np.ma.zeros(pl); z.mask = True
            mask = np.ma.concatenate( [z, data] )
       
    a = Div.div( data, wsize, overlap, pad='constant', constant_values=0 )
    
    #print()
    #print( a )
    #print( 'a.shape', a.shape )
    #print()    

    if np.ma.is_masked(a):
        v = np.ma.var( a, ddof=ddof, axis=1 )
        if not min_points is None:
            l = a.mask.sum(1) > wsize-min_points
            v.mask[l] = 1
    else:
        v = np.var( a, ddof=ddof, axis=1 )

    if samesize:
        V = np.empty(N)
        if center:
            q = overlap//2 + overlap%2
            i0 = wsize - pl - q
            ix = [0] + list(range(i0, N+1, step))
            if ix[-1] != N:
                ix += [N+1]
            #IX = accordion(ix)
            #uix = min((i+1)*step, N+1)
            #print( 'IX', IX )
            #print( 'len(IX)', len(IX) )
            for i,j in enumerate(accordion(ix)):#enumerate(a):
                #print( q, '---> idx', list(range(*IX[i])) )
                V[slice(*j)] = v[i]
            v = V
            
    return v
    
    
    
#==================================================================================================== 
def smoother(x, wsize=11, window='hanning', fill=None, output_masked=None):
    #TODO:  Docstring
    #TODO: smooth (filter) in timescale (use astropy.units?)
    ''' '''
    if x.ndim != 1:
        raise ValueError( "smooth only accepts 1 dimension arrays." )

    if x.size < wsize:
        raise ValueError( "Input vector needs to be bigger than window size." )

    if wsize < 3:
        return x

    #get the window values
    windowVals = get_window(window, wsize)      #window values
    
    #pad array symmetrically at both ends
    s = np.ma.concatenate([ x[wsize-1:0:-1], x, x[-1:-wsize:-1] ])

    #compute lower and upper indices to use such that input array dimensions 
    #equal output array dimensions
    div, mod = divmod(wsize, 2)
    if mod: #i.e. odd window length
        pl, ph = div, div + mod
    else:  #even window len
        pl = ph = div

    #replace masked values with mean / median.  They will be re-masked below
    if fill and np.ma.isMA(s):
        #s.mask = np.r_[ x.mask[wsize-1:0:-1], x.mask, x.mask[-1:-wsize:-1] ]
        wh = np.where(s.mask)[0]

        idxs = itt.starmap(slice, zip(wh - pl, wh + ph))
        func = getattr(np.ma, fill)     #TODO: error handeling
        fillmap = map(lambda idx: func(s[idx]), idxs)
        fillvals = np.fromiter(fillmap, float)
        s[s.mask] = fillvals
    
    #convolve the signal
    w = windowVals / windowVals.sum()   #normalization
    y = np.convolve(w, s, mode='valid')
    
    #return
    output_masked = output_masked or np.ma.is_masked(x)
    if output_masked:
        #re-mask values
        return np.ma.array( y[pl:-ph+1], mask=x.mask)
    
    #return array that has same size as input array
    return y[pl:-ph+1]
        
        
#====================================================================================================       

from recipes.array import ArrayFolder
##############################################################################################################################################
class WindowedArrayFolder(ArrayFolder):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @staticmethod
    def fold(a, wsize, overlap=0, axis=0, **kw):
        window = kw.pop('window', None)
        sa = ArrayFolder.fold(a, wsize, overlap, axis, **kw)
        return Div.windowed(sa, window)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @staticmethod
    def gen(a, wsize, overlap=0, axis=0, **kw):
        window = kw.pop('window', None)
        for sub in ArrayFolder.gen(a, wsize, overlap, axis, **kw):
            yield  ArrayFolder.windowed(sub, window)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @staticmethod
    def windowed(a, window=None):
        '''get window values + apply'''
        if window:
            windowVals = get_window(window, a.shape[-1])
            return a * windowVals
        else:
            return a

Div = WindowedArrayFolder
#TODO: issue deprication warning?
Div.div = Div.fold #depricated!
    
##############################################################################################################################################
    
