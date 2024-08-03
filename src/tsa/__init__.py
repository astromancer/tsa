"""
Time Series Analysis Tools
"""

from . import detrend, window, smooth, outliers
from .spectral import tfr
from .ts import TimeSeries


# aliases
windowing = window
smoothing = smooth