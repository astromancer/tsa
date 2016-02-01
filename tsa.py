import numpy as np
import numpy.linalg as la
import numpy.core.numeric as _nx
from numpy.lib.stride_tricks import as_strided

import scipy as sp
import scipy.signal
import scipy.stats

import itertools as itt

from misc import flatten, accordion, lmap, count_repeats, sortmore

from IPython import embed

#====================================================================================================
def get_deltat( t, kct=None ):
    deltat = t - np.roll(t,1)
    deltat[0] = kct  #set the first value to the kinetic cycle time if available
    return deltat

#====================================================================================================
def get_deltat_mode(t):
    return sp.stats.mode( get_deltat(t) )[0][0]

#====================================================================================================
def get_window(window, N=None):
    if isinstance(window, str):
        if N is None:
            raise ValueError( 'Please specify window size N' )
        return sp.signal.get_window( window, N )
    
    elif np.iterable(window):   #if window values are passed explicitly as a sequence of values
        if not N is None:
            assert len(window) == N, 'length {} of given window does not match array length {}'.format(len(window), N)
        return window

#====================================================================================================
def detect_gaps(t, kct=None, ltol=1.9, utol=np.inf):
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
    '''
    
    if np.ndim(t) > 1:
        t = np.squeeze(t)
        assert np.ndim(t) == 1, 'Gap detection not supported for multidimentional arrays'

    deltat = np.roll(t,-1) - t                     #time separation between successive values

    if kct is None:
        kct = sp.stats.mode(deltat)[0]                       #most frequently occuring time separation

    #if utol is None:
        #utol = np.inf
    
    #embed()
    
    seq = np.abs( deltat / kct )
    l = (seq > ltol) & (seq < utol)
    gap_inds, = np.where( l )
    return gap_inds[:-1]

#====================================================================================================
#TODO: Investigate interpolate
def fill_gaps(t, y, kct=None, mode='linear', option=None, fill=True, ret_idx=False):     #ADD NOISE??
    ''' '''
    def make_filled_array(x, fillers):
        ''' intersperse the fillers and flatten to contiguous array. '''
        cont_secs = np.split(x, gap_inds+1)         #list with continuous sections of the original array
        return np.array( flatten(itt.zip_longest( cont_secs, fillers, fillvalue=[] )) ) 
    
    assert len(t)==len(y), 'Input arrays must have same length. len(t)={}; len(y)={}'.format( len(t), len(y) )

    #if data is masked use unmasked values only
    if np.ma.is_masked( y ):
        t = t[~y.mask]
        y = y[~y.mask]

    #gap detection
    gap_inds = detect_gaps(t, kct)
    
    #fill gaps in original data                     #return filler values only
    Tfill, Yfill = [], []
    IDX = []
    
    #from IPython import embed
    #embed()
    
    for i in gap_inds:
        #print(i)
        npoints = np.round((t[i+1]-t[i]) // kct)                            #number of points that will be inserted
        t_fill = t[i] + np.arange(1, npoints)*kct                       #always fill time gaps with data at constant time step
        
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
            #print( 'k', k )
            #from misc import make_ipshell
            #ipshell = make_ipshell()
            #ipshell()
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
def detrend(y, n=1, t=None, preserve_energy=True):
    '''
    Detrends the time series by fitting a polinomial of degree n and returning the fit residuals.
    '''
    if n is None:
        return y
    
    if t is None:
        t = np.arange(len(y))
    coof = np.ma.polyfit(t, y, n)               #y may be masked!!      
    
    yd = y - np.polyval(coof, t)
    
    if preserve_energy and n>0:  #mean detrending inherently does not preserve energy
        P = (y**2).sum()
        Pd = (yd**2).sum()
        offset = np.sqrt((P-Pd)/len(yd))
        yd += offset
    return yd
    
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
def smooth(x, wsize=11, window='hanning', fill=None, output_masked=None):
    #TODO:  Docstring
    ''' '''
    if x.ndim != 1:
        raise ValueError( "smooth only accepts 1 dimension arrays." )

    if x.size < wsize:
        raise ValueError( "Input vector needs to be bigger than window size." )

    if wsize<3:
        return x

    #get the window values
    if not isinstance(window, str) and isinstance(window, Iterable):        #if window values are passed explicitly as a sequence of values
        assert(len(window) == wsize)
        windowVals = window
    else:
        windowVals = get_window( window, wsize )                     #window values

    s = np.ma.concatenate([ x[wsize-1:0:-1], x, x[-1:-wsize:-1] ])

    #compute lower and upper indices to use such that input array dimensions equal output array dimensions
    div, mod = divmod( wsize, 2 )
    if mod: #i.e. odd window length
        pl, ph = div, div+mod
    else:  #even window len
        pl = ph = div

    #replace masked values with mean / median
    if fill and np.ma.isMA( s ):
        #s.mask = np.r_[ x.mask[wsize-1:0:-1], x.mask, x.mask[-1:-wsize:-1] ]
        wh = np.where( s.mask )[0]

        idxs = itt.starmap( slice, zip(wh - pl, wh + ph) )
        func = getattr( np.ma, fill )
        fillmap = map( lambda idx: func(s[idx]), idxs )
        fillvals = np.fromiter( fillmap, float )
        s[s.mask] = fillvals
    
    #convolve the signal
    w = windowVals/windowVals.sum()
    y = np.convolve(w, s, mode='valid')
    
    #return
    output_masked = output_masked or np.ma.is_masked(x)
    if output_masked:
        return np.ma.array( y[pl:-ph+1], mask=x.mask)
    
    return y[pl:-ph+1]
        
        
#====================================================================================================       


##############################################################################################################################################
class Div(object):
    
    def __call__(self, a, wsize, overlap=0, axis=0, **kw):
        return self.div( a, wsize, overlap=0, axis=0, **kw )
    
    @staticmethod
    def pad(a, wsize, overlap=0, axis=0, **kw):
        ''' '''
        assert wsize > 0, 'wsize > 0'
        assert overlap >= 0, 'overlap >= 0'
        assert overlap < wsize, 'overlap < wsize'
        
        mask = a.mask if np.ma.is_masked(a) else None
        a = np.asarray(a)
        N = a.shape[axis]
        step = wsize - overlap
        Nseg, leftover = divmod(N-overlap, step)

        if leftover:
            
            mode = kw.pop('pad', 'symmetric')       #default pad mode is symmetric
            pad_end = step-leftover
            pad_width = np.zeros((a.ndim, 2), int)       #initialise pad width indicator
            pad_width[axis,-1] = pad_end            #pad the array at the end with 'pad_end' number of values
            pad_width = lmap(tuple, pad_width)      #map to list of tuples
            
            #pad (apodise) the input signal
            a = np.pad(a, pad_width, mode, **kw)
            if not (mask is None or mask is False):
                mask = np.pad(mask, pad_width, mode, **kw)
        
        if not mask is None:
            a = np.ma.array( a, mask=mask )

        return a, int(Nseg)

    @staticmethod
    def windowed(a, window=None):
        '''get window values + apply'''
        if window:
            windowVals = get_window(window, a.shape[-1])
            return a * windowVals
        else:
            return a

    @staticmethod
    def get_strided_array(a, size, overlap, axis=0):
        if axis<0:
            axis += a.ndim
        step = size - overlap
        other_axes = np.setdiff1d(range(a.ndim), axis) #indeces of axes which aren't stepped along
        
        Nwindows = (a.shape[axis] - overlap) // step
        new_shape = np.zeros(a.ndim)
        new_shape[0] = Nwindows
        new_shape[1:] = np.take(a.shape, other_axes)
        new_shape = np.insert(new_shape, axis+1, size)
        
        new_strides = (step*a.strides[axis],) + a.strides

        return as_strided( a, new_shape, new_strides )

        
    
    @staticmethod
    def div(a, wsize, overlap=0, axis=0, **kw):
        '''
        segment an array at given wsize, overlap, optionally applying a windowing function to each
        segment.  
        
        keywords are passed to np.pad used to fill up the array to the required length.  This 
        method works on multidimentional and masked array as well.
        
        keyword arguments are passed to np.pad to fill up the elements in the last window (default is 
        symmetric padding).
        
        NOTE: When overlap is nonzero, the array returned by this function will have multiple entries
        **with the same memory location**.  Beware of this when doing inplace arithmetic operations.
        e.g. 
        N, wsize, overlap = 100, 10, 5
        q = Div.div(np.arange(N), wsize, overlap )
        k = 0
        q[0,overlap+k] *= 10
        q[1,k] == q[0,overlap+k]  #is True
        '''
        window = kw.pop('window', None)
        a, Nseg = Div.pad(a, wsize, overlap, **kw)
        mask = a.mask if np.ma.is_masked(a) else None
        
        sa = Div.get_strided_array(a, wsize, overlap, axis)
        
        if not mask is None:
            if not mask is False:
                mask = Div.get_strided_array(mask, wsize, overlap)
            sa = np.ma.array(sa, mask=mask)

        return Div.windowed(sa, window)

    @staticmethod
    def gen(a, wsize, overlap=0, axis=0,**kw):
        '''
        Generator version of div.
        '''
        window = kw.pop('window', None)
        a, Nseg = Div.pad(a, wsize, overlap, **kw)
        
        step = wsize - overlap

        get_slice = lambda i: [slice(i*step, i*step+wsize) if j==axis 
                                   else slice(None) for j in range(a.ndim)]
        i = 0
        while i<Nseg:
            yield Div.windowed( a[get_slice(i)], window )
            i += 1
    
    @staticmethod
    def get_nocc(N, wsize, overlap):
        '''
        Return an array of length N, with elements representing the number of times that the index 
        corresponding to that element would be repeated in the strided array.
        '''
        I = Div.div(range(N), wsize, overlap)
        d = count_repeats( I.ravel() )
        ix, noc = sortmore( *zip(*d.items()) )
        return noc
    
##############################################################################################################################################
    
