"""
Some functions to generate time series data
"""

import numpy as np

class Harmonic:
    
    def __init__(self, amplitudes, frequencies, phases=0):
        self.a = np.atleast_2d(amplitudes).T
        self.ω = np.atleast_2d(frequencies).T * 2 * np.pi
        self.φ = np.atleast_2d(phases).T
            
    def __call__(self, t, noise=1):
        t = np.atleast_2d(t)
        signal = (self.a * np.sin(self.ω * t + self.φ)).sum(0)
        return signal + noise * np.random.rand(t.size)
    
# class FM:
    
