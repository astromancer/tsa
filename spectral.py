import numpy as np
import multiprocessing as mp

#import scipy as sp
from scipy.fftpack import rfftfreq, fft
#from lombscargle import fasper, getSignificance
from scipy.signal import lombscargle

#from pynfft import NFFT, Solver

from copy import copy

import itertools as itt

from myio import warn
from magic.dict import TransDict
from .tsa import Div, detrend, fill_gaps, get_deltat_mode

from IPython import embed

#====================================================================================================
def FFTpower(y, norm=0, dtrend=None):
    ''' fft computing and normalization '''
    if dtrend:
        y = detrend(y, dtrend)
    
    sp = abs( np.fft.rfft(y) ) ** 2     #Power
    if norm:
        sp /= sp.sum()
    return sp
    
        
#====================================================================================================
def FTspectra(data, dtrend=None):
    '''
    Single-Sided Amplitude Spectrum of y(t). Multiprocessing implimentation 
    NOTE: This assumes evenly sampled data!
    '''
    func = functools.partial( FFT, dtrend=dtrend )
    
    pool = mp.Pool()
    specs = pool.map( func, data )
    pool.close()
    pool.join()
    
    return np.array(specs)

#====================================================================================================
def NFFTspect(nodes, values, N, M):
    nfft = NFFT( N=[N, N], M=M )
    nfft.x = np.c_[ nodes, np.zeros_like(nodes) ]
    nfft.precompute()

    #f     = np.empty( M,       dtype=np.complex128)
    #f_hat = np.empty( (N, N), dtype=np.complex128)

    infft = Solver(nfft)
    infft.y = values         # '''right hand side, samples.'''

    #infft.f_hat_iter = initial_f_hat # assign arbitrary initial solution guess, default is 0
    #print( infft.r_iter )# etc...all internals should still be uninitialized

    infft.before_loop()       # initialize solver internals
    #print( 'infft.r_iter', infft.r_iter ) # etc...all internals should be initialized

    nIter = 0
    maxIter = 50                    # completely arbitrary
    threshold = 1e-6
    while (infft.r_iter.sum() > threshold):
        if nIter > maxIter:
            raise RuntimeError( 'Solver did not converge, aborting' )
        infft.loop_one_step()
        nIter += 1
    
    return infft.f_hat_iter[:,0]


##############################################################################################################################################
class SpectralDataWrapper(object):
    #====================================================================================================
    def __init__(self):
         self.f = []
         self.A = []
         self.signif = []
    
    #====================================================================================================
    def __getitem__(self, key):
        return self.f[key], self.A[key], self.signif[key]
            
    #====================================================================================================
    def __setitem__(self, key, vals):   
        self.f[key], self.A[key], self.signif[key] = vals

##############################################################################################################################################
class Spectral(object):
    '''...'''
    #CompKW = ['split', 'detrend', 'pad', 'gaps', 'window', 'nwindow', 'noverlap']
    #WINDOWS = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
    #====================================================================================================
    @staticmethod
    def show_all_windows(cmap='gist_rainbow'):
        '''plot all the spectral windows defined in scipy.signal (at least those that don't want a 
        parameter argument.)'''
        fig, ax = plt.subplots()
        cm = plt.get_cmap(cmap)
        allwin = sp.signal.windows.__all__
        ax.set_color_cycle( cm(np.linspace(0,1,len(allwin))) )
        
        winge = fct.partial(sp.signal.get_window, Nx=1024)
        for w in allwin:
            try:
                plt.plot(winge(w), label=w)
            except:
                pass
        plt.legend()
        plt.show()
        
    #====================================================================================================
    def __init__(self, t, signal, **kw):
        '''Do a LS spectral density map.   
        If 'split' is a number, split the sequence into that number of roughly equal portions.
        If split is a list, split the array according to the indeces in that list.'''
        print( 'Spectral.compute' )
        #print( kw )
        
        #Check acceptable keyword combinations
        #allkeys = kw.keys()
        #disallowedkwargs = {
            #'split': ['nwindow', 'noverlap'],
            #'nwindow': ['split'],
            #'noverlap': ['split']       }
        
        #kwneeded = {'noverlap': 'nwindow'}
        #kwdefaults = {
            #'which': 'raw',
            #'split': 1,
            #'detrend': 0 ,
            #'pad': None,
            #'gaps': None,
            #'window': lambda x: 1.,
            #'nwindow': 2**8,
            #'noverlap': 2**7,
            #}
        
        #kwoptions = {}
        #else: raise KeyError('Invalid option %s for keyword %s') 
        
        # Make sure have allowed kw combinations
        #for key in kw:
            #if key in disallowedkwargs:
                #if [k in disallowedkwargs[key]
                #raise ValueError('{} keyword not in allowed keywords {}'.format( key, np.setdiff1d(allkeys, disallowedkwargs[key]) ) )
        
        # Set kw defaults
        #for key in np.setdiff1d(allkeys, disallowedkwargs[key]):
            #kw.setdefault(key, kwdefaults[key])
        
        #kw = TransDict(**kw)
        #translations = ((('frequency', 'freq', 'frq'),                  'f'),
                        #(('lomb-scargle', 'lombscargle'),               'ls'),
                        #(('noverlap',),                                 'overlap'))
                        ##(('noverlap',),                                 'overlap'))
        #transmap = dict(itt.chain(*( itt.zip_longest( k, (v,), fillvalue=v ) for k,v in m )))
        #kw.translate( transmap )
        #====================================================================================================
        use =           kw.setdefault( 'use',           'fft'           )
        timescale =     kw.setdefault( 'timescale',     's'             )          #the timescale of the passed time variable.  Will be used to convert to s timescale.
        split =         kw.setdefault( 'split',         None            )
        dtrend =        kw.setdefault( 'detrend',       None            )       
        pad =           kw.setdefault( 'pad',           'symmetric'     )        #'constant', 'mean', ,'median', 'minimum', 'maximum', 'reflect', 'symmetric', 'wrap', 'linear_ramp', 'edge' 
        gaps =          kw.setdefault( 'gaps',          None            )        #tuple with fill method and option
        window =        kw.setdefault( 'window',        'boxcar'        )
        nwindow =       kw.setdefault( 'nwindow',       None            )
        noverlap =      kw.setdefault( 'overlap',       0               )
        kct =           kw.setdefault( 'kct',           None            )
        apodise =       kw.setdefault( 'apodise',       None            )
        normalise =     kw.setdefault( 'normalise',     None            )
        f =             kw.setdefault( 'f',             None            )
        #====================================================================================================
        
        #print( kw )
        
        if kct is None:
            warn( 'Determining KCT from time-step mode stat.' )
            kct = get_deltat_mode( t )
        
        #if (use.lower() in ('fft', 'fourier') or not gaps is None) and kct is None:
            #kct = Spectral.get_delta( t )
            #desc = 'FFT analysis' if use.lower in ('fft', 'fourier') else 'Gap detection'
            #warn('{} analysis requested, no KCT supplied.  KCT={} assumed from the most frequently occuring time delta (mode)...'.format(desc, kct))
            
        if nwindow is None:
            if split is None:
                nwindow = len(t)                #No segmentation
            elif isinstance(split, int):
                nwindow = len(t) // split       #*split* number of segments
            else:
                #split at specific indeces
                'check that split is the correct format'                                    #if split values are passed explicitly as an integer or iterable
        else:
            if isinstance(nwindow, str):
                #if nwindow.endswith('s'):
                print( 'nwindow', nwindow )
                nwindow = round( float(nwindow.strip('s')) / kct )
                print( 'nwindow', nwindow )
            else:
                'check that nwindow is integer'                                 #CAN BE HANDELED BY KW OPTIONS
        
        noverlap = self.get_overlap(nwindow, noverlap)
        
        if timescale=='h':
            conv_fact = 3600.                                                       #conversion factor for kct to timing array passed to this function
        elif timescale=='s':
            conv_fact = 1
        
        #Fill data gaps
        if gaps:
            fillmethod, option = gaps
            print( fillmethod, option )
            t, signal = fill_gaps(t, signal, kct, fillmethod, option)
        elif np.ma.is_masked(signal):
            warn( 'Removing masked values' )
            t = t[~signal.mask]
            signal = signal[~signal.mask]
            
        #====================================================================================================
        print( nwindow, noverlap )
        
        segments, Power = [], []
        
        if nwindow:
            step = nwindow - noverlap
            leftover = (len(t)-noverlap) % step
            end_time = t[-1] + kct * (step-leftover) ,
            self.t_seg = Div.div( t, nwindow, noverlap, 
                                  pad='linear_ramp', 
                                  end_values=end_time )
            self.raw_seg = Div.div( signal,  nwindow, noverlap, window=window ) #padding will happen below
        else:
            self.t_seg             = np.split(t, split)
            self.raw_seg        = np.split(signal, split)

        self.tms = np.median( self.t_seg, 1 )                #mid time for each section
        
        #FFT frequencies
        #embed()
        if f is None:
            f = np.fft.rfftfreq(apodise or nwindow, kct)
        self.frq = f
        
        if use.lower() in ('ls', 'lomb-scargle', 'lombscargle'):
            if 0 in f:
                self.frq = f = f[1:]
        self.ohm = 2*np.pi*f         #angular frequencies
        
        #TODO: MULIPROCESS!
        for t_s, seg in zip(self.t_seg, self.raw_seg):
            
            t = (t_s - t_s[0]) * conv_fact                                                 #time in seconds          #NOTE:  THESE CONVERSION VALUES WILL CHANGE DEPENDING ON THE TIME VARIABLE PASSED TO  ls()
            #T = t[-1]  #length of this segment in sec
            seg = detrend( seg, dtrend, t, preserve_energy=False )
            #NOTE THAT OUTLIERS WILL ADVERSELY INFLUENCE POLINOMIAL DETRENDING!!!!
            #print( seg )
            
            if apodise:                 #pad each segment so that it ends up being this size
                #TODO: MOVE OUT OF LOOP
                div, mod = divmod(apodise - len(t), 2)
                padwidth = (div, div+mod)   
                
                #embed()
                
                seg = np.pad( seg, (div, div+mod), mode='constant', constant_values=0 )
            
            #print( 'Calculating ls for section {}: t in [{},{}]'.format(i, t_s[0], t_s[-1])  )
            
            if use.lower() in ('ls', 'lomb-scargle', 'lombscargle'):
                #embed()
                P = lombscargle(t, seg, self.ohm)  #LS periodogram
                #P = np.r_[np.nan, P]    #DC signal cannot be measure by LS
                
            if use.lower() in ('fft', 'fourier'):
                P = FFTpower( seg, norm=False, dtrend=False )
            
            #FIXME: appending is SLOWWWWWWWWW
            segments.append( seg )
            Power.append( P )
        
        self.power = np.array(Power)
        self.segments = np.array(segments)
        
        if normalise is True:
            normalise = 'rms'
    
        if isinstance(normalise, str):
            self.normed = normalise = normalise.lower()
            Nph = np.c_[ self.raw_seg.sum(1) ]
            #Nph = signal.sum()                 #N_{\gamma} in Leahy 83 = DC component of FFT
            #N = len(signal)
            T = nwindow * kct
            if normalise=='leahy':
                self.power = 2 * self.power / Nph
            elif normalise=='leahy density':
                self.power = 2 * T * self.power / Nph
            elif normalise=='rms':
                self.power = 2 * T * self.power / Nph**2
    
    
    #====================================================================================================
    @staticmethod
    def get_overlap(nwindow, noverlap):
        '''convert overlap to integer value'''
        if not bool(noverlap):
            return 0
        
        if isinstance(noverlap, str):       #overlap specified by percentage string eg: 99% or timescale eg: 60s
            if noverlap.endswith('%'):
                frac = float(noverlap.strip('%')) / 100
            noverlap = frac * nwindow
            
        if isinstance(noverlap, float):
            noverlap = round(noverlap)
            
        if noverlap<0:                      #negative overlap works like negative indexing! :)
            noverlap += nwindow
            
        if noverlap==nwindow:
            noverlap -= 1                   #Maximal overlap!
        
        return noverlap

    #====================================================================================================
    def __iter__(self):
        return self.frq, self.power#, self.segments

        



        
        
        
        
#Orphans        
#def running_sigma_clip(x, sig=3., nwindow=100, noverlap=0, iters=None, cenfunc=np.ma.median, varfunc=np.ma.var):

    ##SLOWWWWWWWWW...................
    
    #step = nwindow - noverlap
    #o = (nwindow // step) + int(bool(nwindow % step))

    #if not np.ma.is_masked(x):
        #x = np.ma.array(x, mask=np.isnan(x))
        
    #sections = Div.div( x, nwindow, noverlap )
    #filtered_data = []
    #for i, sec in enumerate(sections):
        
        #filtered_sec = sec.copy()
        
        #if iters is None:
            #lastrej = filtered_sec.count() + 1
            #while (filtered_sec.count() != lastrej):
                
                #lastrej = filtered_sec.count()
                #secdiv = filtered_sec - cenfunc(filtered_sec)
                
                #flagged = np.ma.greater( secdiv*secdiv, varfunc(secdiv)*sig**2 )
                #w, = np.where(flagged)
                #wmin, wmax = w.min(), w.max()
                #flagger = sections[i-o+wmin+1:i+1].mask
                #for j in range(wmax-wmin):
                    #flagger[j, nwindow-j-1:] |= flag[wmin:wmin+j+1]
                

                #filtered_sec.mask |= False
                ##print( filtered_sec.mask )
            ##iters = i + 1
        #else:
            #for k in range(iters):
                #secdiv = filtered_sec - cenfunc(filtered_sec)
                #filtered_sec.mask |= np.ma.greater( secdiv*secdiv, varfunc(secdiv)*sig**2 )

        #filtered_data.append( filtered_sec )

    #return np.ma.concatenate(filtered_data)[:len(x)]

    
#def rejection(i):
    #global sections
    #sec = sections[i]
    #last_rej_count = sec.mask.sum()  #non masked items
    #print( 'section ', i, sec )
    #for j in range(1):
        #print( '\tj =', j )
        #secdev = sec - cenfunc(sec)
        #secstd = np.sqrt( varfunc(secdev) )
        #flagged = (-sig*secstd > secdev) & (secdev > sig*secstd)
        ##flagged = np.ma.greater( secdev*secdev, varfunc(secdev)*sig**2 )  #This produces a masked array if data in sec has masked values, hence
        #flagged = flagged.data & flagged.mask                                               #Produces a numpy array (not masked) 
        #print( 'flagged', flagged )
        #print( 'last_rej_count, flagged.sum()',  last_rej_count, flagged.sum() )
        #if last_rej_count == flagged.sum():            #No new rejections on this pass
            #print( 'returning\n' )
            #return
        
        ##because of the overlap we have to modify all the preceding sections that contain the
        ##now-rejected data points, also masking them in these segments and recomputing the variance
        ##for those segments, and checking if any new points need to be rejected in those segments 
        ##based on the new variance
        #w, = np.where(flagged)                          #indeces of rejected points based on criterion above
        #print( 'flagged @', w )
        #wmin, wmax = w.min(), w.max()                   
        #flagger = sections[i-o+wmin+1:i+1].mask         #a slice of the lagged array containing new-rejected points
        #for k in range(wmax-wmin):
            #print( '\t\tk =', k )
            #flagger[k, nwindow-k-1:] |= flagged[wmin:wmin+k+1]    #mask all the new-rejected points in the preceding segments of the lagged array
            ##recompute variance
            #print( 'recurring' )
            #print( )
            #rejection( i-o+wmin+k+1 )


