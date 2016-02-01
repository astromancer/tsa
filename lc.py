import numpy as np
from .tsa import get_deltat_mode

class LightCurve(object):
    def __init__(self, t,r,e):
        self.t = t
        self.r = r
        self.e = e
    
    def _get_parts(self):
        return self.t, self.r, self.e
    
    parts = property( _get_parts )
    
    def  __str__(self):
        pass
    
    def __len__(self):
        return len(self.t)
    
    def __iter__(self):
        pass
    
    def __neg__(self):
        return LightCurve( self.t, -self.r, self.e )
        
    def __pos__(self):
        return self
    
    def __abs__(self):
         return LightCurve( self.t, abs(self.r), self.e )
    
    def __add__(self,o):
        t,r,e = self.parts
        to,ro,eo = o.parts

        if len(t) != len(to):
            #can't add directly...have to rebin? 
            NOTE: this also induces uncertainty in T
            tn, deltat = self._retime(o)
            t = tn + deltat/2                    #bin centers
            i = np.searchsorted( t, tn )        #indeces to use
            io = np.searchsorted( to, tn )
            r, e = r[i], e[i]
            ro, eo = ro[io], eo[io]
            
        R = r + ro
        E = np.sqrt(np.square((e,eo)).sum(0)) #add errors in quadrature
        
        return LightCurve(t, R, E)
    
    
    def __sub__(self, o):
        return self + -o
        
    
    def _retime(self,o):
        t,r,e = self.parts
        to,ro,eo = o.parts
        
        mode0 = get_deltat_mode(t)
        mode1 = get_deltat_mode(to)
         
        if mode0==mode1:  #cool, time steps are at least identical
            #now find the overlapping bits
            t0 = min(t.min(), to.min())
            t1 = max(t.max(), to.max())
            
            first = np.argmin((t[0], to[0])) #which has the earliest start?
            if first:
                t, to = to, t                #swap them around, so that t starts first
                r, ro = ro, r
                e, eo = eo, e
            
            #choose bins in such a way that the times from both arrays in a single bin have minimal seperation
            offset = abs(t-to[0]).argmin()
            b0 = t[0] - (to[0]-t[offset])/2 #first bin starts here
            Nbins = np.ceil( (t1-t0) / mode0 )
            bin_edges = b0 + np.arange(Nbins+1)*mode0

            h, b = np.histogram( np.r_[t,to], bin_edges )
            tn = b[h==2]                        #bin edges for new time bins
        else:
            'rebin one of them??'

        return tn, mode0
    
    def __mul__(self,o):
        pass
    def __truediv__(self,o):
        pass
    def __pow__(self,o):
        pass
    
    def __mul__(self,o):
        pass
    def __truediv__(self,o):
        pass
    def __pow__(self,o):
        pass
    
    def __lt__(self, b):
        pass
    def __le__(self, b):
        pass
    def __eq__(a, b):
        pass
    def __ne__(a, b):
        pass
    def __ge__(a, b):
        pass
    def __gt__(a, b):
        pass
    
    