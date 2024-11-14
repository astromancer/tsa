"""
Time series objects
"""

# third-party
import numpy as np

# local
from recipes.oo.property import Alias

# relative
from ..smooth import KernelSmoother, tv
from .interface import Interface
from .plotting import TimeSeriesPlot
from .ms import MeasurementSequence, MultiVariate


# ---------------------------------------------------------------------------- #
# import uncertainties.unumpy as unp  # linear uncertainty propagation
# since this library is tracking correlations between rvs operations like mean
# on an array is n^2 or n^3 which means its actually too slow to be useful.

# need class that tracks uncertainties but with correlation tracking
# optionally disabled
# Options: Linearized uncertainty propagation via `uncertainties`
#        : Full distribution calculation with rv arithmetic via `pacal`
# if _handle_uncertainties == 0:

# for now calculations on uncertainty has to be done manually which is a pain..

# ---------------------------------------------------------------------------- #

# def mean(x, w):
#     """Weighted Mean"""
#     return np.ma.average(x, w)

# def cov(x, y, w):
#     """Weighted Covariance"""
#     return np.ma.average((x - mean(x, w)) * (y - mean(y, w)), w)

# def corr(x, y, w):
#     """Weighted Correlation"""
#     return cov(x, y, w) / np.sqrt(cov(x, x, w) * cov(y, y, w))

# ---------------------------------------------------------------------------- #
# class WindowedAnalysis:
#     def __init__(nwindow=None, noverlap=0, njobs=-1, **kws):


#     def __call__(self, x, wsize=11, window='hanning'):
#         return KernelSmoother(window, wsize)(x)

#     def tv(self, smoothing=None, nwindow=None, noverlap=0, λ0=1, njobs=-1, **kws):

#         t, x, u = self.get_data(())
#         # x = x[(..., *[np.newaxis] * (x.ndim == 1))].T
#         y = np.empty_like(x)

#         if nwindow:
#             njobs = (njobs, )
#             smoother = tv.MovingWindowSmoother(nwindow, noverlap, **kws)
#             name = 'tv.MovingWindowSmoother'
#         else:
#             # no windowing. might bork for long ts
#             njobs = ()
#             smoother = tv.smooth
#             name = 'tv.smooth'

#         if (m := x.shape[1]) > 1:
#             self.logger.debug('Looping over {} variates.', m)

#         self.optima = []
#         for i, xx in enumerate(x.T):
#             smoother.jobname = f'{name} ({i + 1}/{m})'
#             self.logger.debug('Running {} with njobs={} on {} array, λ = {}.',
#                               smoother.jobname, njobs, xx.shape, smoothing)
#             result = smoother(t, xx, smoothing, λ0, *njobs)

#             if smoothing:
#                 y[:len(result), i] = result
#             else:
#                 result, optimum = result
#                 y[:len(result), i] = result
#                 self.optima.append(optimum)

#         #     return y, np.reshape(optima, (-1, i + 1))

#         return TimeSeries(t, y)


# ---------------------------------------------------------------------------- #
class Smoothing(Interface):

    def __call__(self, x, wsize=11, window='hanning'):
        return KernelSmoother(window, wsize)(x)

    def tv(self, smoothing=None, nwindow=None, noverlap=0, λ0=1, njobs=-1, **kws):

        t, x, u = self.get_data(())
        # x = x[(..., *[np.newaxis] * (x.ndim == 1))].T
        y = np.empty_like(x)

        if nwindow:
            njobs = (njobs, )
            smoother = tv.MovingWindowSmoother(nwindow, noverlap, **kws)
            name = 'tv.MovingWindowSmoother'
        else:
            # no windowing. might bork for long ts
            njobs = ()
            smoother = tv.smooth
            name = 'tv.smooth'

        if (m := x.shape[1]) > 1:
            self.logger.debug('Looping over {} variates.', m)

        self.optima = []
        for i, xx in enumerate(x.T):
            smoother.jobname = f'{name} ({i + 1}/{m})'
            self.logger.debug('Running {} with njobs={} on {} array, λ = {}.',
                              smoother.jobname, njobs, xx.shape, smoothing)
            result = smoother(t, xx, smoothing, λ0, *njobs)

            if smoothing:
                y[:len(result), i] = result
            else:
                result, optimum = result
                y[:len(result), i] = result
                self.optima.append(optimum)

        #     return y, np.reshape(optima, (-1, i + 1))

        return TimeSeries(t, y)


# class OutlierDetection(Interface):
#     def __call__(self, nwindow, noverlap):

# ---------------------------------------------------------------------------- #

class TimeSeries(MeasurementSequence):
    """
    A basic univariate time series with optional uncertainties.
    """

    # TODO
    # --------
    #  * automatic (linearized) uncertainty propagation (assumes normally
    #    distributed data)
    #  *
    #   interface with
    #    -statmodels
    #   - pandas
    #   - pacal
    #   - uncertainties
    #     ~ Approximate (linear) uncertainty propagation. Will only yield
    #       accurate results when used in computation if uncertainties are small
    #       compared to variation of the (non-linear) function through which it is
    #       mapped.

    #       note:
    #       "Error estimates for non-linear functions are biased on account of
    #       using a truncated series expansion. The extent of this bias depends
    #       on the nature of the function. For example, the bias on the error
    #       calculated for log(1+x) increases as x increases, since the expansion
    #       to x is a good approximation only when x is near zero."
    #       from: https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Caveats_and_warnings

    #  * spectral estimation techniques
    #  * sampled time series + visualizations.

    #  Not supported
    #  - units.  Make sure you use compatible units when doing arithmetic

    # Interfaces
    # ------------------------------------------------------------------------ #
    plot = TimeSeriesPlot(xlabel='Time (s)',
                          ylabel='Signal')
    #
    smooth = Smoothing()
    # mwa
    # ------------------------------------------------------------------------ #

    # Time
    # ------------------------------------------------------------------------ #
    t = Alias('index')
    x = Alias('value')
    u = Alias('sigma')

    # Spectral estimators
    # ------------------------------------------------------------------------ #

    def periodogram(self, window=None, detrend=None, pad=None, norm=None, **kws):
        from tsa.spectral import Periodogram

        return Periodogram.fit(self.t, self.x, window=window, detrend=detrend,
                               pad=pad, norm=norm, **kws)

    def spectrogram(self, nwindow, noverlap=0, window='hanning', detrend=None,
                    pad=None, split=None, norm=False, **kws):
        from tsa.spectral import Spectrogram

        return Spectrogram.fit(self.t, self.x,
                               nwindow=nwindow, noverlap=noverlap,
                               window=window, detrend=detrend,
                               pad=pad, split=split, norm=norm, **kws)

    def correlogram(self, max_lag=None, method=None, njobs=-1):
        from tsa.spectral import Correlogram

        Correlogram(max_lag, method, njobs)

    #
    acf = Alias('correlogram')

    # ------------------------------------------------------------------------ #
    # def fold(self, eph):


class MultiVariateTimeSeries(MultiVariate, TimeSeries):
    """Multivariate Time Series"""

    def corner(self, *args, **kws):
        return corner(self.values, *args, **kws)
    
    

# alias
MultivariateTimeSeries = MultiVariateTimeSeries
