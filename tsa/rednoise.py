"""
Implements significance tests for periodic signals in presence of red
noise form Vaughan 2005A&A...431..391V
"""

# import math
import warnings
import operator
import functools as ftl
import itertools as itt
import multiprocessing as mp

import numpy as np
from scipy.integrate import quad, dblquad
from scipy.optimize import brentq

from recipes.parallel.synced import SyncedArray

# NOTE: can be optimized more for better performance?
#  powerlaw variance is symmetrical about 1 in log frequency
#       ==> 10**3 and 10**-3 Hz has same variance

# integrand is infinite at 0, which leads to multiple warnings being emitted.
# To avoid this, set the lower limit of the interval to very small positive
# value
zero = 1e-20  # todo: remove

# a few global variables defined for optimization
LN10 = np.log(10)
LN10SQ = LN10 ** 2
SQRT2 = np.sqrt(2)
SQRT1PI = np.sqrt(np.pi)
SQRT8PI = np.sqrt(8 * np.pi)
VAR_LN_P = (np.pi / LN10) ** 2 / 6


# variance of the log-periodogram about true spectrum (Geweke & Porter-Hudak
# 1983) (\sigma^2 in Vaughan 2005)


def powerlaw(N, alpha, x):
    return N * np.power(x, -alpha)


def powerlaw_variance(frq):
    """
    Uncertainties on the slope and normalisation estimates from LS fit
    """

    a = np.ma.log10(np.atleast_1d(frq))
    n = len(a)  # - 1
    # n is the number of frequencies used in the fitting.
    # Normally n = n − 1 since only the Nyquist frequency is ignored
    # (because the periodogram at the Nyquist frequency does
    # not follow the same distribution as at the other frequencies)

    if n == 1:
        raise ValueError('Powerlaw variance is ill-defined for single '
                         'frequency point')

    a2s = (a * a).sum()
    asum = a.sum()
    delta = n * a2s - asum ** 2
    # delta = n * delta_n

    # var slope
    var_alpha = n * VAR_LN_P / delta
    # var normalization factor
    var_logN = VAR_LN_P * a2s / delta
    # covariance
    cov = VAR_LN_P * asum / delta
    return var_logN, var_alpha, cov


def periodogram_variance(frq):
    """
    The expected uncertainties and covariance of the two model parameters can
    be combined to give an estimate of the uncertainty of the logarithm of
    the model, log[P̂ j ], at a frequency f j , using the standard error
    propagation formula.
    """

    a = np.ma.log10(frq)
    var_logN, var_alpha, cov = powerlaw_variance(frq)
    var_logP = var_alpha * a * a + var_logN - 2 * cov * a
    # NOTE: the log-normal distribution is conventionally defined in terms of
    #  the natural logarithm, whereas previously the results were given in
    #  terms of base 10 logarithm. The uncertainty on the model log-powers
    #  needs to be corrected:
    return var_logP * LN10SQ  # S^2_j =


def _integrand(w, z, s2):
    return np.exp(_f(w, z, s2))


def _f(w, z, s2):
    return -0.5 * (np.square(np.log(w)) / s2 + z * w)


def pdf_gamma(z, s2j):
    # The ratio γ̂ j = 2I j /P̂ j is really the ratio of two random variables;
    # the PDF of this would allow us to calculate the probability of observing
    # a given value of γ̂ j taking full account of the uncertainty in the
    # model fitting. of the uncertainty in the model fitting.
    # 2I j will follow a rescaled χ22 distribution about the true spectrum.
    # In the case of the LS fitting discussed in Sect. 3 the model P̂ j has a
    # log-normal distribution. The probability density of the power in the
    # fitted model at frequency f j is therefore:
    A, Ae = quad(_integrand, zero, np.inf, (z, s2j))
    return A / (SQRT8PI * np.sqrt(s2j))


def cdf_gamma(z, s2j):
    A, Ae = dblquad(_integrand, 0, z, lambda _: zero, lambda _: np.inf, (s2j,))
    return A


def _gamma_worker(f, args):
    i, (z, sj2) = args
    _shared[i] = f(z, sj2)


def _bulk_compute(frq, Z, grid=True, **kws):
    """Calculate probability surface for pdf / cdf"""

    pdf = kws.get('pdf', not kws.get('cdf', False))
    func = pdf_gamma if pdf else cdf_gamma
    worker = ftl.partial(_gamma_worker, func)

    frq = np.atleast_1d(frq)
    Z = np.atleast_1d(Z)

    # make sure we have 1D arrays
    if (np.ndim(frq) > 1) or (np.ndim(Z) > 1):
        raise ValueError('higher dims not supported')

    Sj2 = periodogram_variance(frq)

    if grid:  # compute on grid formed by tensor product of vectors
        args = map(tuple,
                   itt.starmap(zip, itt.product(*map(enumerate, (Z, Sj2)))))
        # returns tuples of ((i,j), (z, sj2)) for outer product
    else:
        args = enumerate(zip(Z, Sj2))
        # returns tuples of (i, (z, sj2)) for inner product

    _init_shared_memory(Z, Sj2, grid)
    with mp.Pool() as pool:
        pool.map(worker, args)
    pool.join()

    return np.array(_shared)


import tempfile


def _init_shared_memory(Z, Sj2, grid):
    global _shared

    shape = (len(Z),)
    if grid:
        shape += (len(Sj2),)

    id_, tmp = tempfile.mkstemp()
    _shared = np.lib.format.open_memmap(tmp, 'w+', np.float, shape)


# @memoize.to_file(cachePath)
def _confidence_solver(sj2, c):
    # The calculation of γ depends only on p_γ_j (z), from Eq. (21), which in
    # turn depends only on Sj, from Eq. (13), and this is calculated using
    # the the abscissae (frequencies f_j) with no dependence on the ordinates
    # (periodogram powers I_j ). The critical value γ can be evaluated
    # using only the frequencies of the periodogram.

    def _solve(z):
        return cdf_gamma(z, sj2) - c

    return brentq(_solve, 0, 1e3)


def global_confidence(frq, percent=99.9, local=False):
    # The probability of obtaining a value of γ̂ j higher than γ_e can be
    # computed by integrating this PDF: This can be evaluated numerically to
    # find γ_e for a given e1 . Equivalently, we can find the value of γ_j at
    # the corresponding 1 − e significance level:

    # For example, using e = 0.05 (i.e. a 95 per cent significance test) we
    # find γ0.05 = 5.99. This means that if the null hypothesis is true the
    # probability of the ratio γˆj being higher than 5.99 is only 0.05. We
    # can therefore define our 95 (and 99) per cent confidence limits
    # on the log-periodogram as the model P̂(f) = N̂ f −α̂  multiplied by
    # the  appropriate γ_e/2.
    # (In log-space we simply add the appropriate log[γ_e /2] to the model.)

    # Finally, we need to correct for the number of frequencies examined.
    # The probability that a peak will be seen given that n' frequencies were
    # examined is pe_n = 1 − (1 − ep_1)^n. One can find the global
    # (1 − ep_n)100 per cent confidence level by finding the value γ_e that
    # satisfies:
    #  where n is again the number of frequencies examined.

    # TODO: interpolate for speed ...

    assert (0 < percent < 100)

    frac = percent / 100.  # 1 - epsilon
    if local:
        c = frac
    else:
        c = pow(frac, 1 / len(frq))

    func = ftl.partial(_confidence_solver, c=c)
    with mp.Pool() as pool:
        gamma_e = pool.map(func, periodogram_variance(frq))
    pool.join()

    return np.array(gamma_e)


def global_confidence2(frq, percent=99.9, local=False):
    # this one should be much faster since it involves only a single
    # numerical integration due to some clever calculus (my own work - not in
    # Vaughan (2005))

    assert (0 < percent < 100), 'Significance should be a percentage value ' \
                                'between 0 and 100 (exclusive).'
    c = percent / 100.  # 1 - epsilon
    if not local:
        c = pow(c, 1 / len(frq))

    func = ftl.partial(solver2, c=SQRT1PI * (1 - c))
    s = periodogram_variance(frq)

    # TODO: progress bar here since this can take some minutes for
    with mp.Pool() as pool:
        result = pool.map(func, s.compressed())
    pool.join()

    gamma_e = np.zeros_like(s)
    gamma_e[~gamma_e.mask] = result
    return gamma_e


def integrand2(x, gamma, sj):
    return np.exp((-x * x - 0.5 * gamma * np.exp(SQRT2 * sj * x)))


def eq2(gamma, sj, c):
    inf = 1e2
    val, err = quad(integrand2, -inf, inf, (gamma, sj))
    return val - c


def solver2(sj, c):
    return brentq(eq2, 0, 1e3, (sj, c))


# import inspect
# from pathlib import Path
# from decor.misc import memoize
# # get coordinate cache file
# here = inspect.getfile(inspect.currentframe())
# moduleDir = Path(here).parent
# cacheName = '.cache.confidence_solver'
# cachePath = moduleDir / cacheName


def test_memoize(frq, percent=99.9):
    assert (0 < percent < 100)

    frac = percent / 100.  # 1 - epsilon
    if local:
        c = frac
    else:
        n = len(frq)
        c = pow(frac, 1 / n)


from scipy.optimize import leastsq

from .spectral import Spectral
from obstools.modelling.core import Model


class Linear(Model):
    """A straight line"""

    def __call__(self, p, x):
        'P_j = N f_j^-\alpha ln(f_j)'
        return p[0] * x + p[1]


class PowerLawSpec(Spectral):
    def __init__(self, *args, **kws):

        Spectral.__init__(self, *args, **kws)

        # suppress warnings for zeros
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            # log-log for fitting powerlaw
            self.logfrq = np.log10(self.frq)
            self.logPwr = np.log10(self.power) + 0.25068  # unbiased

    def fit_powerlaws_lsq(self, ignore=None, return_masked=False,
                          convert_log=True):
        """
        Fit periodogram ordinates (power) with power-law following:
        Vaughan 2005A&A...431..391V
        """
        # The logarithm of the periodogram ordinate, with the bias removed
        # (i.e. constant added) is an unbiased estimator of the logarithm of
        # the spectrum, and is iid (about the underlying spectrum) at each
        # frequency. It is important that the datum at Nyquist frequency be
        # ignored in the least-squares procedure since it is chi2 with 1 dof.
        # (not identical to other frequencies)

        if ignore is None:
            ignore = np.zeros_like(self.frq, dtype=bool)
            ignore[-1] = True  # ignore nyquist
        ignore[self.frq == 0] = True  # ignore DC (infinite in logspace)

        logfrq = self.logfrq[~ignore]
        logPwr = self.logPwr[..., ~ignore] + 0.25068
        # unbiased spectral estimator (c.f. Eq. (26.4.36) of
        # Abramowitz & Stegun (1964))

        # fitting (multiprocessed)
        lin = Linear()  # Model
        p0 = (1, 1)
        solver = ftl.partial(leastsq, lin.rs, p0)
        args = zip(logPwr, itt.repeat(logfrq))
        with mp.Pool() as pool:
            res = pool.map(solver, args)
        pool.join()

        p, success = zip(*res)
        p = np.array(p).T

        if return_masked:
            mask = False if np.all(success) else ~np.array([success] * 2,
                                                           dtype=bool)
            p = np.ma.array(p, mask=mask)

        # compute model
        X = np.c_[self.logfrq, np.ones(self.n_frq)]
        self.models = (X @ p).T

        if convert_log:
            alpha = -p[0]
            N = np.pow(10, p[1] - 0.25068)
            return N, alpha, success
        else:
            return p, success

    def plot_spectrum(self, k=0, use=..., gamma_e=(), log=True, model='full',
                      clabels=(), **kws):
        """

        :param k:       index of segment
        :param use:     indices of frequencies to use
        :param gamma_e: confidence limit curves
        :param log:     boolean - whether to plot in log scale
        :param model:
        :param clabels:
        :param kws:
        :return:
        """
        g = np.asarray(gamma_e) / 2
        # used = ~ignore  # points used for fit

        if model == 'full':
            muse = slice(None)
        else:
            muse = use
        m = self.models[k, muse]

        if log:
            x = self.logfrq
            y = self.logPwr
            g = np.log10(g)
            op = operator.add
        else:
            x = self.frq
            y = self.power
            m = np.pow(10, m)
            op = operator.mul

        fig, ax = plt.subplots(figsize=(14, 9), tight_layout=True)
        ax.plot(x, y[k], 'r-', label='Periodogram')  # the periodogram
        if not use is Ellipsis:
            ax.plot(x[use], y[k, use], 'x',
                    label='Fit points')  # points used for fit

        # plot model
        ax.plot(x[muse], m)
        # plot confidence uppper-limits for spectrum
        # kws.get('clabels', )

        # from IPython import embed
        # embed()

        for i, (gg, lbl) in enumerate(itt.zip_longest(g, clabels)):
            ax.plot(x[::10], op(m[::10], gg), 'k--', dashes=(2 * (i + 1),) * 2,
                    label=lbl)

            # ax.plot(x, m + np.log10(g / 2), 'k:' )

            # caption
            #


#     def detrend_powerlaw(self, N, alpha):
#         if np.size(N) > 1:
#             N = np.array(10, ndmin=2).T
#         if np.size(alpha) > 1:
#             alpha = np.array(10, ndmin=2).T

#         return self.power[i] - powerlaw(N, alpha, self.frq[None])

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import colorConverter


def fancy_3D(Frq, Z, P, log=False, gamma_e=None):
    def shade3D(ax, k, ge=None, color='g', alpha=0.4):
        """3D shaded area"""
        if ge is None:
            ge = Z[:, k].max()

        seg = Z[:, k] <= ge

        fk, zk, pgk = Frq[seg, k], Z[seg, k], P[seg, k]
        fk = np.r_[fk, fk[-1]]
        zk = np.r_[zk, ge]
        pgk = np.r_[pgk, pgk[-1]]

        verts = np.c_[np.c_[[fk, zk, pgk]],
                      [fk[-1], zk[-1], zero],
                      [fk[0], zk[0], zero]].T
        colour = colorConverter.to_rgba(color, alpha=alpha)
        polygon = Poly3DCollection([verts], facecolors=colour)
        ax.add_collection3d(polygon)

        ax.plot3D(fk, zk, pgk, color=color, ls='-')
        ax.plot3D(fk, zk, zero, color=color, ls=('--' if log else '-'))

        # TODO: squiggly line at bottom?
        return polygon

    # Fancy 3D plot of pdf_gamma_j
    fig, ax = plt.subplots(figsize=(14, 8),
                           subplot_kw=dict(projection='3d', ),
                           # axisbg='None'),
                           # facecolor='none',
                           tight_layout=True)

    ix = np.multiply(Frq.shape[-1], [0.01, 0.75]).astype(int)
    if log:
        P = np.log10(P)
        brackets = r'$\log(', ')$'
        zero = -6.6

        # fline = (Frq[0, ix[1]], Frq[0, -1] + 2)
        seg = ...

        txtposge = 0.81, 0.3
        txtpos = (0.18, 0.5)
        txtrot = 0
        lhs = ''

        ax.elev, ax.azim = (48.5, -48.2)  # pick magic viewing angle

    else:
        brackets = '$$'
        zero = 0
        seg = ...  # slice(0, 10)

        txtpos = (0.05, 0.85)  # (0.15, 0.82)  #(0.5, 0.67)
        txtrot = 0  # 22 #-12
        lhs = 'F_{%(gj)s}(%(ge)s) = '

        ax.elev, ax.azim = (34, 34)  # (46.3, 31.7)  # pick magic viewing angle

    # 3D plot
    wire = ax.plot_wireframe(Frq[seg], Z[seg], P[seg])

    if gamma_e is not None:  # plot_gamma_line
        ax.set_autoscale_on(False)
        pol1 = shade3D(ax, ix[0], gamma_e[ix[0]], 'g')
        pol2 = shade3D(ax, ix[1], gamma_e[ix[1]], 'orange')

        txtposge = (0.52, 0.037)

        frq = Frq[0, :]
        pge = pdf_gamma_bulk(frq, gamma_e, grid=False)
        line, = ax.plot3D(frq, gamma_e, pge, 'r', ls='-')
        #
        s = r'$\gamma_{\epsilon} := F^{-1}_{%(gj)s}(0.95)$' \
            % dict(ge=r'\gamma_{\epsilon}', gj=r'\gamma_j')
        txt = ax.text(0, np.mean(gamma_e), 0.05,
                      s,
                      fontsize=18,
                      )
        # txt_ge = ax.text2D(*txtposge, '$\gamma_{\epsilon}$',
        #                    fontsize=18,
        #                    transform=ax.transAxes)
    else:
        pol1 = shade3D(ax, ix[0], None, 'g')
        pol2 = shade3D(ax, ix[1], None, 'orange')

    # text
    Psym = r'\mathrm{\mathbb{P}}'  # symbol to use for Probability
    lhs += r'%(P)s(\hat{%(gj)s} < %(ge)s)'
    rhs = r'\int\limits_{0}^{%(ge)s} %(P)s_{%(gj)s}(z) \,dz'
    txt = '$${}={}$$'.format(lhs, rhs) % dict(P=Psym, ge=r'\gamma_{\epsilon}',
                                              gj=r'\gamma_j')
    # option 1: rotated to match axes
    # eq = ax.text2D(0.175, 0.475, txt,
    #           rotation=25,
    #           fontsize=17,
    #           transform=ax.transAxes)
    # option 2: horizontal
    eq = ax.text2D(*txtpos, txt,
                   rotation=txtrot,
                   fontsize=18,
                   transform=ax.transAxes)

    # labels
    Pg = r'%s_{\gamma}(z)' % Psym
    axlbls = ax.set(xlabel='$f_j$', ylabel='$z$',
                    zlabel='{1}{0}{2}'.format(Pg, *brackets))
    return fig, ax


if __name__ == '__main__':
    from matplotlib import rcParams
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.colors import colorConverter

    from recipes.decor.profile.timers import timer_dev

    rcParams["text.latex.preamble"].append(r'\usepackage{amsfonts}')
    rcParams["text.usetex"] = True
    rcParams["font.size"] = 14

    # symbol to use for Probability
    Psym = r'\mathrm{\mathbb{P}}'


    @timer_dev
    def test_sequential(Z, Sj2):
        igrl = np.empty((len(Z),) + Sj2.shape)
        for (i, z), (j, sj2) in itt.product(*map(enumerate, (Z, Sj2))):
            igrl[i, j] = pdf_gamma(z, sj2)
        return igrl


    @timer_dev
    def test_mp(Z, Sj2):
        return pdf_gamma_bulk(Z, Sj2)


    @timer_dev
    def test_cdf(Sj2):
        cdf_inf = [cdf_gamma(np.inf, sj2) for sj2 in Sj2]
        return np.allclose(cdf_inf, 1)


    def plot_pgram_var(frq, Sj2):
        # plot periodogram variance (including model uncertainty) with freq
        fig, ax = plt.subplots()
        ax.loglog(frq, Sj2)
        ax.grid()


    def plot_integrand(frq):
        # plot integrand for a bunch of freq
        fig, ax = plt.subplots()
        # ix = np.random.randint(0, len(frq), 6)
        w = np.linspace(1e-5, 3, 250)
        ix = range(0, len(frq), len(frq) // 5)
        for i in ix:
            ax.plot(w, _integrand(w, 2, Sj2[i]))
        ax.grid()


    def plot_integrand_3D(w, frq, Sj2):
        # 3D plot
        I = _integrand(w[None].T, 1, Sj2[None, :])
        Frq = np.tile(w, (len(frq), 1)).T
        W = np.tile(frq, (len(w), 1))

        fig, ax = plt.subplots(figsize=(14, 8),
                               subplot_kw=dict(projection='3d', ),
                               # axisbg='None'),
                               # facecolor='none',
                               tight_layout=True)

        ax.plot_wireframe(Frq, W, I)


    def test_profile():
        # from line_profiler import LineProfiler
        from recipes.decor.profile import HLineProfiler

        # from recipes.io.tracewarn import warning_traceback_on
        # warning_traceback_on()

        profiler = HLineProfiler()
        for func in [cdf_gamma, pdf_gamma, _integrand, _f]:
            profiler.add_function(eval(func))
        profiler.enable_by_count()

        conf95 = global_confidence(frq[::20], 95, local=True)

        profiler.print_stats()
        profiler.rank_functions()


    # test computation
    *frng, n = 1e-3, 10, 256
    *zrng, nz = 0, 15, 15
    Z, Frq = np.mgrid[slice(*zrng, complex(nz)),
                      slice(*frng, complex(n))]
    z = Z[:, 0]
    frq = Frq[0]
    # Sj2 = periodogram_variance(frq)
    # igrl = pdf_gamma_bulk(z, Sj2)

    # sequential
    # sq = test_sequential(z, Sj2)

    # parallel
    # sp = test_mp(z, Sj2)

    # test cdf integrates to unity
    # test_cdf(Sj2)

    # profile
    test_profile()

    # plots
    # fancy_3D(Frq, Z, igrl)
    # plt.show()
