import numpy as np
import pylab as plt

from misc import Gaussian
from pynfft import NFFT, Solver
import functools

np.random.seed( 123456 )
x = np.linspace(-0.5, 0.5, 1000) # 'continuous' time/spatial domain; -0.5<x<+0.5

# function we want to reconstruct
#a = 50
#p = (1, a, 0)
#g = functools.partial(Gaussian, p)              # 'true' underlying function
#y = g(x)
#Fp = (np.sqrt(np.pi/a), np.pi**2/a, 0)          # coefficients for analytical solution to FT[g]
#Fg = functools.partial(Gaussian, Fp )           # Fourier transform of g (analytical solution)

def g(x, k):
    k = np.atleast_2d(k).T
    return np.sin(2*np.pi*x*k).sum(0)

    
# we should sample at a rate of >2*~max(k)
M = 128                   # number of nodes
N = 64                   # number of Fourier coefficients

k = [1,5,23,30]
nodes = np.random.rand(M) - 0.5 # non-uniform oversampling
values = g(nodes, k)            # nodes&values will be used below to reconstruct original function using the pyNFFT solver
y = g(x, k)

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

#nfft.trafo()
#P = np.abs( nfft.f_hat[:,0] )**2

#Power spectra
#ret2 = nfft.adjoint()
s = NFFTspect(nodes, values, N, M )
P = np.abs(s)**2
#Rn = R/R.sum()*y.sum()
#Padj = np.abs(ret2[:,0])**2
#Padj /= Padj.sum()
K = np.arange(-N/2, N/2)#np.linspace(-0.5, 0.5, N)

#Plots
#-----
fig, ax = plt.subplots( figsize=(18,6), tight_layout=1 )
ax.plot( x, y, 'r', label='g' )
ax.plot( nodes, values, 'rx', mew=2, label='nodes' )


ax.legend()
ax.grid()



#nfft.f_hat = infft.f_hat_iter
#f = nfft.trafo()
#plt.figure()
#f = nfft.trafo()
#R = np.abs(f)**2
#R -= R.mean()
#plt.plot( nodes, R, 'bo'  )

fig, ax = plt.subplots( figsize=(18,6), tight_layout=1 )
for q in k:  ax.axvline(q, color='g')

ax.plot( K, P, 'gx', mew=1.5, label='NFFT Solver' )
#ax.plot( K, Padj, 'gx', label='NFFT adjoint' )
#ax.plot(x, Fg(x), 'g', label='Analytical')
#ax.plot(K, R, 'gx', label='Reconstruction')
ax.legend()
ax.grid()


#fig, ax = plt.subplots( figsize=(18,6), tight_layout=1 )

#ax.plot(x, Fg(x), 'g')
#ax.plot(np.abs(values), f, 'gx')
#ax.plot(nodes, values, 'bo')
##ax.set_xlim(-0.5, 0.5)
#ax.grid()

plt.show()

