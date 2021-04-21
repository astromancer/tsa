"""
Versatile functions for plotting time-series data
"""

from mpl_toolkits.axes_grid1 import make_axes_locatable
from cycler import cycler
from matplotlib import ticker
from matplotlib.transforms import Affine2D
import numbers

from .dualaxes import DualAxes, DateTimeDualAxes
from recipes.logging import logging, get_module_logger
from attr import attrs, attrib as attr  # , astuple
from recipes.dicts import AttrDict
import matplotlib.pyplot as plt
import itertools as itt

import numpy as np
import warnings as wrn
import matplotlib as mpl

from .utils import get_percentile_limits
from .draggable import DraggableErrorbar
from .ticks import OffsetLocator

# TODO:
#  alternatively make `class timeseriesPlot(Axes):` then ax.errorbar()
# NOTE: you can probs use the std plt.subplots machinery if you register your
#  axes classes

# mpl.use('Qt5Agg')
# from matplotlib import rcParams

# import colormaps as cmaps
# plt.register_cmap(name='viridis', cmap=cmaps.viridis)


# from recipes.string import minlogfmt

# from IPython import embed

# from dataclasses import dataclass


# module level logger
logger = get_module_logger()
logging.basicConfig()
logger.setLevel(logging.INFO)


# Set parameter defaults
DEFAULTS = AttrDict(
    # labels=(),
    # title='',
    #
    timescale='s',
    start=None,

    axes_labels=(('t (s)', ''),  # bottom and top x-axes
                 'Counts'),
    # TODO: axes_label_position: (left, right, center, <,>,^)
    twinx=None,  # Top x-axis display format
    # xscale='linear',
    # yscale='linear',
    plims=((0, 100),  # x-axis
           (-1, 101)),  # y-axis

)  # TODO: x, y, upper, lower

# Default options for plotting related stuff
default_opts = AttrDict(
    errorbar=dict(fmt='o',
                      # TODO: sampled lcs from distribution implied by
                      #  errorbars ?  simulate_samples
                      ms=2.5,
                      mec='none',
                      capsize=0,
                      elinewidth=0.5),
    spans=dict(label='filtered',
               alpha=0.2,
               color='r'),
    hist=dict(bins=50,
              alpha=0.75,
              # color='b',
              orientation='horizontal'),
    legend=dict(loc='upper right',  # TODO: option for no legend
                    fancybox=True,
                    framealpha=0.25,
                    numpoints=1,
                    markerscale=3)
)

allowed_kws = list(DEFAULTS.keys())
allowed_kws.extend(default_opts.keys())


TWIN_AXES_CLASSES = {'sexa': DateTimeDualAxes}
N_MAX_TS_PLOT = 50


class TooManyToPlot(Exception):
    """
    Exception that occurs when user accidentally requests too many plots. The
    number at which this occurs is determined by the module variable
    `N_MAX_TS_PLOT`
    """
    pass


# TODO: indicate more data points with arrows????????????
#       : Would be cool if done while hovering mouse on legend

# from astropy import units as u
# from astropy.coordinates.angles import Angle

# TODO: support for longer scale time JD, MJD, etc...
# h,m,s = Angle((self.t0 * u.Unit(self.timescale)).to(u.h)).hms
# #start time in (h,m,s)
# hms = int(h), int(m), int(round(s))Ep
# #datetime hates floats#

# FIXME: self.start might be None
# if self.start is None:

# ymd = tuple(map(int, self.start.split('-')))
# start = ymd + hms


@attrs
class DataPercentileAxesLimits(object):
    lower = attr(-0.05)
    upper = attr(+1.05)

    def get(self, data, e=()):
        return get_percentile_limits(data, (self.lower, self.upper), e)


def _set_defaults(props, defaults):
    for k, v in defaults.items():
        props.setdefault(k, v)


def check_kws(kws):
    # these are AttrDict!

    dopts = default_opts.copy()

    invalid = set(kws.keys()) - set(allowed_kws)
    if invalid:
        raise KeyError('Invalid keyword{}: {}.\n'
                       'Only the following keywords are recognised: {}'
                       ''.format('s' if len(invalid) > 1 else '',
                                 invalid, allowed_kws))

    for key, val in kws.items():
        # deal with keyword args for which values are dict
        if key in dopts:
            dopts[key].update(val)

    return {**DEFAULTS, **kws}, dopts


def is_null(x):
    if (x is None) or (len(x) == 0):
        return True

    # check that errors are not all masked. This sometimes happens
    # when data is read into fields where uncertainties are expected
    if np.ma.getmask(x).all():
        logger.info('All data in vector are masked. Ignoring.')
        return True

    return False


def is_uniform(x):
    if is_null(x) or is1d(x):
        return True
    return len(set(map(len, x))) == 1


def is1d(x):
    return isinstance(x[0], numbers.Real)


def get_labels(labels, signals):
    # check labels
    if is_null(labels):
        return []
    if isinstance(labels, str):
        return [labels]  # check if single label given
    if len(labels) != len(signals):
        wrn.warn('Number of labels does not match number of time series.')
    return labels


def _parse_data(data, labels):

    n = len(data)
    if 4 < n < 1:
        raise ValueError('Invalid number of arguments: %i' % n)

    # signals only
    if n == 1:
        # Assume here each row gives individual signal for a TS
        signals = data[0]

        # check for structured data (dict keyed on labels and containing data)
        if isinstance(signals, dict):
            # labels = signals.keys()
            # data = list(zip(*signals.values()))
            yield from _parse_data(list(zip(*signals.values())),
                                   list(signals.keys()))
            return

        data = (), signals

    # times, signals, [y-errors, x-errors] given
    for i, d in enumerate(data, 1):
        yield [d, ()][is_null(d)]
        # if is_null(d):
        #     yield ()
        # else:
        #     yield d

    for _ in range(i, 4):
        yield ()

    yield labels


def auto_transpose(array, like):
    time_axis = np.argmax(like.shape)
    assert time_axis == 1

    array = np.atleast_2d(array)
    assert array.ndim == 2
    if np.argmax(array.shape) != 1:
        return array.T
    return array


def get_data(data, labels):
    """parse data arguments"""

    times, signals, y_err, x_err, labels = _parse_data(data, labels)
    #
    if (is_uniform(times) and is_uniform(signals)):
        # auto transpose
        times = np.atleast_2d(times)
        # NOTE: atleast_2d may return masked array
        signals = auto_transpose(signals, times)
    elif (len(times) != len(signals)):
        # ragged signal list, explicit time stamps
        raise ValueError('Number of time and signal vectors do not correspond. Please provide explicit time stamps for all signal vectors when plotting ragged time series.'
                         )

    # safety breakout for erroneous arguments that can trigger very slow
    # plotting loop
    n = len(signals)
    if n > N_MAX_TS_PLOT:
        raise TooManyToPlot(
            'Received %i time series to plot. This is probably not what you '
            'wanted. Refusing since safety limit is currently set to %i to '
            'avoid accidental compute intensive commands from overwhelming '
            'system resources.' % (n, N_MAX_TS_PLOT))

    t0 = times if is1d(times) else times[0]
    yield check_data('time', times, signals, t0)
    yield signals
    yield check_data('y_err', y_err, signals)
    yield check_data('x_err', x_err, signals)
    # for name, array in dict(y_err=y_err, x_err=x_err).items():
    # yield check_data(name, array, signals)
    yield get_labels(labels, signals)


def check_data(name, array, signals, fill=None):
    # for name, vector in kws.items():
    if is_null(array):
        yield None
        return

    if is_uniform(signals) and is_uniform(array):
        array = auto_transpose(array, signals)

    n = len(signals)
    if n < len(array):
        raise ValueError(
            f'Superfluous {name} vector(s). Received {len(array)}, expected {n}')

    for vector, signal in itt.zip_longest(array, signals, fillvalue=fill):
        n = len(signal)
        if not is_null(vector) and (len(vector) != n):
            raise ValueError(f'Unequal number of points between signal '
                             f'({n}) and {name} vectors ({len(vector)}).')
        yield vector


def sanitize_data(t, signal, y_err, x_err):
    """
    clean up data for single time series before plot

    Parameters
    ----------
    t
    signal
    y_err
    x_err
    relative_time

    Returns
    -------

    """

    # mask nans
    signal = np.ma.MaskedArray(signal, ~np.isfinite(signal))
    if is_null(t):
        t = np.arange(len(signal))
    return (t, signal, y_err, x_err)


# def sanitize_data(t, signal, y_err, x_err, show_errors, relative_time):
#     """
#     clean up data for single time series before plot
#
#     Parameters
#     ----------
#     t
#     signal
#     y_err
#     x_err
#     show_errors
#     relative_time
#
#     Returns
#     -------
#
#     """
#     n = len(signal)
#     stddevs = []
#     for yx, std in zip('yx', (y_err, x_err)):
#         if std is not None:
#             if show_errors:
#                 size = np.size(std)
#                 if size == 0:
#                     # TODO: these could probably be info
#                     logger.warning(f'Ignoring empty uncertainties in {yx}.')
#                     std = None
#                 elif size != n:
#                     raise ValueError(f'Unequal number of points between data '
#                                      f'({n}) and {yx}-stddev arrays ({size}).')
#                 else:
#                     std = np.ma.masked_where(np.isnan(std), std)
#
#                 # check that errors are not all masked. This sometimes happens
#                 # when data is read into fields where uncertainties are expected
#                 if std.mask.all():
#                     logger.warning(f'All uncertainties in {yx} are masked.  '
#                                    f'Ignoring.')
#                     std = None
#
#             else:
#                 logger.warning(f'Ignoring uncertainties in {yx} since '
#                                '`show_errors = False`.')
#                 std = None
#         # aggregate
#         stddevs.append(std)
#
#     # plot by frame index if no time
#     if (t is None) or (len(t) == 0):
#         t = np.arange(len(signal))
#     else:
#         if len(t) != len(signal):
#             raise ValueError('Unequal number of points between data and time '
#                              'arrays.')
#         # Adjust start time
#         if relative_time:
#             t = t - t[0]
#
#     # mask nans
#     signal = np.ma.MaskedArray(signal, ~np.isfinite(signal))
#     return (t, signal) + tuple(stddevs)


def get_line_colours(n, colours, cmap):
    # Ensure we plot with unique colours

    # rules here are:
    # `cmap` always used if given
    # `colours` always used if given, except if `cmap` given
    #   warning emitted if too few colours - colour sequence will repeat
    too_few_colours = len(mpl.rcParams['axes.prop_cycle']) < n
    if (cmap is not None) or ((colours is None) and too_few_colours):
        cm = plt.get_cmap(cmap)
        colours = cm(np.linspace(0, 1, n))  # linear colour map for ts

    elif (colours is not None) and (len(colours) < n):
        wrn.warn('Colour sequence has too few colours (%i < %i). Colours '
                 'will repeat' % (len(colours), n))
    return colours


# def get_axes_limits(data, whitespace, offsets=None):
#     """Axes limits"""
#
#     x, y, u = data
#     xf, yf = duplicate_if_scalar(whitespace)  # Fractional white space in figure
#     xl, xu = axes_limit_from_data(x, xf)
#     yl, yu = axes_limit_from_data(y, yf, u)
#
#     if offsets is not None:
#         yl += min(offsets)
#         yu += max(offsets)
#
#     return (xl, xu), (yl, yu)


def get_axes_labels(axes_labels):
    if (axes_labels is None) or (len(axes_labels) == 0):
        return DEFAULTS.axes_labels

    if len(axes_labels) != 2:
        raise ValueError('Invalid axes labels')

    xlabels, ylabel = axes_labels
    if isinstance(xlabels, str):
        xlabels = (xlabels, '')
    return xlabels, ylabel


def uncertainty_contours(ax, t, signal, stddev, styles, **kws):
    # NOTE: interpret uncertainties as stddev of distribution
    from tsa.smoothing import smoother

    # preserve colour cycle
    sigma = 3
    c, = ax.plot(t, smoother(signal + sigma * stddev), **kws)
    colour = styles.errorbar['color'] = c.get_color()
    return ax.plot(t, smoother(signal - sigma * stddev), colour, **kws)


def get_axes(ax, figsize=None, twinx=None, **kws):

    if ax is not None:
        return ax.figure, ax

    # get axes with parasite (sic)
    if twinx is not None:
        axes_cls = TWIN_AXES_CLASSES.get(twinx)
        if axes_cls:
            # make twin
            fig = plt.figure(figsize=figsize)
            ax = axes_cls(fig, 1, 1, 1, **kws)
            ax.setup_ticks()
            fig.add_subplot(ax)
            return fig, ax

        #
        wrn.warn('Option %r not understood for argument `twinx`. '
                 'Ignoring.', twinx)

    return plt.subplots(figsize=figsize)


def setup_figure(ax, show_hist):
    """Setup figure geometry"""

    # FIXME:  leave space on the right of figure to display offsets
    fig, ax = get_axes(ax)

    # Add subplot for histogram
    hax = None
    if show_hist:
        divider = make_axes_locatable(ax)
        hax = divider.append_axes('right', size='25%', pad=0.,
                                  sharey=ax)
        hax.grid()
        hax.yaxis.tick_right()

    # # NOTE: #mpl >1.4 only
    # if colours is not None:
    #     ccyc = cycler('color', colours)
    #     ax.set_prop_cycle(ccyc)
    #     if show_hist:
    #         hax.set_prop_cycle(ccyc)

    ax.grid(b=True)  # which='both'
    return fig, ax, hax


class TimeSeriesPlot:
    """
    A time series plotting class
    """
    # TODO: evolve to multiprocessed TS plotter.
    # TODO: Keyword translation?

    def __init__(self, ax=None, title='', hist=(), plims=DEFAULTS.plims):

        self.fig, self.ax, self.hax = setup_figure(ax, hist)
        self.art = []
        self.hist = []
        self._linked = []
        # _proxies = []

        # axes limits
        self.plims = plims
        self.xlim = np.array([np.inf, -np.inf])
        self.ylim = np.array([np.inf, -np.inf])

        self.zorder0 = 10

        self.styles = {}

        # default layout for pretty figures
        # left, bottom, right, top = [0.025, 0.01, 0.97, .98]
        # fig.tight_layout(rect=rect)
        # return fig, ax

    def plot(self, *data, masked=False, hist=False, relative_time=False,
             labels=(), colors=None, cmap=None,
             draggable=False, offsets=(),
             **kws):
        """
        Plot time series

        Parameters
        ----------
        data: tuple of array-likes
            (signal,)   -   in which case t is implicitly the integers up to
                            len(signal).
            (t, signal) -   in which case uncertainty is ignored.
            (t, signal, uncertainty)

        t: array-like or tuple of array-likes or None, optional
            Time stamps. If None or empty, the *signal* will be plotted over
            an index array. If tuple or multi-dimensional array, the array axes
            are interpreted as for *signal*. If *t* is 1D and signal is 
            multi-dimensional, the same timesteps will be used for all time 
            series in signal - ie. multivariate time series.  
        signal: array-like or dict
            Time series data values. For multivariate data, first dimension
            indexes the different variables. ie. For an input array, the shape
            should be (n, m) where n is the number of time series and m is the
            number of points.
        uncertainty : array-like or tuple of array-likes, optional
            Standard deviation uncertainty associated with signal.

        """

        # FIXME: get this to work with astropy time objects
        # TODO: docstring
        # TODO: astropy.units ??
        # TODO: max points = 1e4 ??

        # Check keyword argument validity
        kws, styles = check_kws(kws)
        show_hist = bool(len(kws.get('hist', {})))

        # from IPython import embed
        # embed(header="Embedded interpreter at 'ts.py':536")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # parse data args: times, signals, y_err, x_err
        # *data, _labels = get_data(data, labels)
        # n = len(data[1])  # signals

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # colours = get_line_colours(n, colors, cmap)

        # print('before zip:', len(times), len(signals), len(errors))
        # Do the plotting

        # d = y, t, uy, ux, lbls = get_data(data, labels)
        # from IPython import embed
        # embed(header="Embedded interpreter at 'ts.py':568")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # zip_longest in case errors or times are empty sequences
        for i, (t, y, σy, σt, label) in enumerate(itt.zip_longest(
                *get_data(data, labels))):
            # print(np.shape(t), np.shape(y), np.shape(σy), np.shape(σt))
            # if yo:
            #     y = y + yo
            self.errorbar(t, y, σy, σt, label, masked,
                          show_hist, relative_time, styles)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # add text labels
        # self.set_labels(title, kws.axes_labels,
        #                kws.twinx, relative_time)

        # for lim in (self.xlim, self.ylim):
        #     lim += np.multiply([-1, 1], (np.ptp(lim) * kws.whitespace / 2))

        # set auto-scale limits
        # print('setting lims: ', self.ylim)
        for xy in 'xy':
            lim = getattr(self, f'{xy}lim')
            if np.isfinite(lim).all():
                self.ax.set(**{f'{xy}lim': lim})

        # xlim=self.xlim, ylim=self.ylim,
        # ax.set(xscale=self.kws.xscale, yscale=self.kws.yscale)

        # self.set_axes_limits(data, kws.whitespace, (kws.xscale, kws.yscale),
        #                     kws.offsets)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Setup plots for canvas interaction

        # FIXME: offsets should work even when not draggable!!

        if draggable and not show_hist:
            # FIXME: maybe warn if both draggable and show_hist
            # make the artists draggable

            self.plots = DraggableErrorbar(self.art, offsets=kws.offsets,
                                           linked=self._linked,
                                           **styles.legend)
            # TODO: legend with linked plots!

        else:
            self.ax.legend(self.art, labels, **styles.legend)
            # self._make_legend(ax, self.art, labels)

        return self

    def errorbar(self, x, y, y_err, x_err, label,
                 show_masked, show_hist, relative_time, styles):

        # TODO maxpoints = 1e4  opt

        # if (y_err is not None) & (show_errors == 'contour'):
        #     uncertainty_contours(self.ax, x, y, y_err, styles, lw=1)

        # NOTE: masked array behaves badly in mpl < 1.5.
        # see: https://github.com/matplotlib/matplotlib/issues/5016/
        # x = x.filled(np.nan)

        # main plot
        x, y, y_err, x_err = data = sanitize_data(x, y, y_err, x_err)

        ebar = self.ax.errorbar(x, y, y_err, x_err,
                                label=label, zorder=self.zorder0,
                                **styles.errorbar)
        self.art.append(ebar)

        self.set_limits(x, y, x_err, y_err)

        if relative_time:
            self.ax.xaxis.major.formatter.set_useOffset(x[0])
            self.ax.xaxis.set_major_locator(OffsetLocator())

        # plot masked values with different style if requested
        if show_masked:
            self.plot_masked_points(x, y, show_masked)

        # Histogram
        if show_hist:
            self.plot_histogram(y, **styles.hist)

        self.zorder0 = 1

        return ebar

    def plot_masked_points(self, t, signal, marker='x', color=None, **kws):
        # Get / Plot GTIs

        # msk_art = None
        # if how == 'span':
        #     self.plot_masked_intervals(ax, t, unmasked.mask)

        if color is None:
            last = line, *_ = self.art[-1]
            color = line.get_color()

        # invert mask
        unmasked = np.array(signal[signal.mask])

        # NOTE: using errorbar here so we can easily convert to
        #  DraggableErrorbarContainers
        # FIXME: Can fix this once DraggableLines are supported
        # TODO: manage opts in styles.....
        # if how == 'x':
        ebar = self.ax.errorbar(t[signal.mask], unmasked, color=color,
                                marker=marker, ls='None', alpha=0.7,
                                label='_nolegend_')
        if ebar:
            self.art.append(ebar)
            self._linked.append((last, ebar))

        # raise NotImplementedError

    def plot_histogram(self, signal, **props):
        #
        self.hist.append(
            self.hax.hist(np.ma.compressed(signal), **props)
        )

        self.hax.grid(True)

    def set_limits(self, x, y, x_err, y_err):
        # set axes view limits
        lims = []
        for xy, p, e in zip((x, y), self.plims, (x_err, y_err)):
            lims.append(get_percentile_limits(xy, p, e))

        for lim, xy in zip(lims, 'xy'):
            l, u = getattr(self, f'{xy}lim')
            new_lim = np.array([min(lim[0], l), max(lim[1], u)])

            # check compat with scale
            scale = getattr(self.ax, f'get_{xy}scale')()
            if scale == 'log':
                neg = ([x, y][xy == 'y'] <= 0)
                # if neg.any():
                #     logger.warning(
                #             'Requested logarithmic scale, but data contains '
                #             'negative points. Switching to symmetric log '
                #             'scale')
                #     self.kws[f'{xy}scale'] = 'symlog'
                if new_lim[0] <= 0:  # FIXME: both could be smaller than 0
                    wrn.warn('Requested negative limits on log scaled axis. '
                             'Using smallest positive data element as lower '
                             'limit instead.')
                    new_lim[0] = y[~neg].min()
            # print('new', new_lims)
            # set new limits
            setattr(self, f'{xy}_lim', new_lim)
            # print('YLIMS', ylims)
            # print('lims', 'x', self.xlim, 'y', self.ylim)

    def set_labels(self, title, axes_labels, twinx, relative_time, t0=''):
        """axis title + labels"""
        ax = self.ax
        title_text = ax.set_title(title, fontweight='bold')

        xlabels, ylabel = get_axes_labels(axes_labels)
        xlb, xlt = xlabels
        ax.set_xlabel(xlb)
        ax.set_ylabel(ylabel)

        if twinx:
            # make space for the tick labels
            title_text.set_position((0.5, 1.09))
            if xlt:
                ax.parasite.set_xlabel(xlt)

        # display time offset
        if relative_time:
            ax.xoffsetText = ax.text(1, ax.xaxis.labelpad,
                                     '[{:+.1f}]'.format(t0),
                                     ha='right',
                                     transform=ax.xaxis.label.get_transform())

    def loglog(self, *data, **kws):
        self.ax.set_xscale('log')
        self.ax.set_yscale('log')
        return self.plot(*data, **kws)

    # def animate():
        # simulated_samples from normal distribution given uncertainties


def convert_mask_to_intervals(a, mask=None):
    """Return index tuples of contiguous masked values."""
    if mask is None:
        mask = a.mask
        # NOTE: If a is a masked array, this function returns masked values!!!

    if ~np.any(mask):
        return ()

    from recipes.iter import interleave
    w, = np.where(mask)
    l1 = w - np.roll(w, 1) > 1
    l2 = np.roll(w, -1) - w > 1
    idx = [w[0]] + interleave(w[l2], w[l1]) + [w[-1]]
    return a[idx].reshape(-1, 2)


def time_phase_plot(P, toff=0, **figkws):

    fig = plt.figure(**figkws)

    aux_trans = Affine2D().translate(-toff, 0).scale(P)
    ax = DualAxes(fig, 1, 1, 1, aux_trans=aux_trans)
    ax.setup_ticks()
    fig.add_subplot(ax)

    ax.parasite.yaxis.offsetText.set_visible(False)
    ax.parasite.set_xlabel('Time (s)')
    ax.set_xlabel('Orbital Phase')

    return fig, ax


def phase_time_plot(P, toff=0, **figkws):
    fig = plt.figure(**figkws)

    aux_trans = Affine2D().translate(-toff, 0).scale(1 / P)
    ax = DualAxes(fig, 1, 1, 1, aux_trans=aux_trans)
    ax.setup_ticks()
    fig.add_subplot(ax)

    ax.set_xlabel('Orbital Phase')

    return fig, ax


# TODO: PeriodicTS(t, data, p).fold_plot(mean, std, extrema, style='|')
#  this would make a neater API

def plot_folded_lc(ax, phase, stats, p, twice=True, sigma=1., orientation='h',
                   colours=('b', '0.5', '0.5')):
    """
    plot folded lc mean/max/min/std

    Parameters
    ----------
    ax
    phase
    stats:
        mean, min, max, std
    p: float
        Period in seconds
    twice
    orientation

    Returns
    -------

    """

    from matplotlib.patches import Rectangle

    mean, mini, maxi, std = np.tile(stats, (twice + 1))
    line_data = (mean, mini, maxi)
    if twice:
        phase = np.r_[phase, phase + 1]

    t = phase * p
    std = mean + std * sigma * np.c_[1, -1].T

    # get appropriate fill command / args
    v = orientation.startswith('v')
    args = zip((itt.repeat(t), line_data)[::(1, -1)[v]])
    fill_between = getattr(ax, f'fill_between{"x" * v}')

    lines = []
    for a, colour in zip(args, colours):
        pl, = ax.plot(*a, color=colour, lw=1)
        lines.append(pl)
    plm, plmn, plmx = lines

    # fill uncertainty contour
    fill_between(t, *std, color='grey')

    # add axis labels  set limits
    xy = 'xy'[v]
    ax.set(**{f'{xy}lim': (twice + 1) * p,
              f'{xy}label': 't (s)'})

    # rectangle proxy art for legend.
    r = Rectangle((0, 0), 1, 1, fc='grey', ec='none')
    leg = ax.legend((plm, plmn, r), ('mean', 'extrema', r'$1\sigma$'))

    ax.grid()
    ax.figure.tight_layout()
    # return fig


# def plot_masked_intervals(self, ax, t, mask):
#     """
#     Highlight the masked values within the time series with a span across
#      the axis
#      """
#     spans = convert_mask_to_intervals(t, mask)
#     for s in spans:
#         ax.axvspan(*s, **self.dopts.spans)
#
#     self.mask_shown = True
#     # bool(bti)
#     # #just so we don't make a legend entry for this if it's empty


# def _make_legend(self, ax, plots, labels):
#     """Legend"""
#
#     # print( labels, '!'*10 )
#
#     if len(labels):
#         if self.mask_shown:
#             from matplotlib.patches import Rectangle
#             span_label = self.span_props.pop('label')
#             r = Rectangle((0, 0), 1, 1,
#                           **self.span_props)  # span proxy artist for legend
#
#             plots += [r]
#             labels += [span_label]
#
#         ax.legend(plots, labels, **self.dopts.legend)
#
#

#
# def sexa(h, pos=None):
#     m = abs((h - int(h)) * 60)
#     sign = '-' if h < 0 else ''
#     return '{}{:2,d}ʰ{:02,d}ᵐ'.format(sign, abs(int(h)), int(m))


def make_twin(ax, tick_label_angle=0, period=1):
    from scrawl.ticks import SexagesimalFormatter

    # make transform
    axp = ax.twin(Affine2D().translate(0, 0).scale(1 / period / 86400))  # / 24
    # make ticks
    axp.xaxis.set_major_locator(ticker.MultipleLocator(30 * 60))
    axp.xaxis.set_major_formatter(
        SexagesimalFormatter(precision='m0', unicode=True))
    ax.xaxis.set_ticklabels([])
    axp.yaxis.set_ticks([])

    if tick_label_angle:
        axp.tick_params(pad=0)
        ticklabels = axp.xaxis.get_majorticklabels()
        for label in ticklabels:
            label.set_ha('left')
            label.set_rotation(tick_label_angle)
            label.set_rotation_mode('anchor')

    return axp


def phased_multi_axes(times, data, std, ephemeris, thin=1,
                      colours='midnightblue', ylim_shrink=0.8,
                      subplot_kw=None, gridspec_kw=None, **kws):
    """

    Parameters
    ----------
    times
    data
    std
    ephemeris
    thin
    colours
    subplot_kw
    gridspec_kw

    Returns
    -------

    """
    from mpl_toolkits.axes_grid1.parasite_axes import SubplotHost

    # sharex=True, # not sharing x since it shares
    # all the ticks which is NOT desired here.
    # instead set range for all
    # NOTE: could try:
    # for tck in ax.xaxis.get_major_ticks():
    #       tck.label1.set_visible(True)

    n = len(times)
    fig, axes = plt.subplots(n, 1,
                             sharey=True,

                             subplot_kw=subplot_kw,
                             gridspec_kw=gridspec_kw
                             )

    # hack to get dual axes on topmost
    pos = axes[0].get_position()
    axes[0].remove()

    ax = fig.axes[0] = axes[0] = SubplotHost(fig, n, 1, 1, **subplot_kw)
    axp = make_twin(ax, 45, ephemeris.P)
    fig.add_subplot(ax)
    ax.set_position(pos)

    # get colours
    if not isinstance(colours, (list, tuple, np.ndarray)):
        colours = [colours] * n

    # plot options
    opts = dict(fmt='o', ms=1, alpha=0.75, clip_on=False)
    opts.update(**kws)

    # do plotting
    s = np.s_[::thin]
    xlim = [np.inf, -np.inf]
    ylim = [np.inf, -np.inf]
    for i, (ax, t, y, u) in enumerate(zip(axes, times, data, std)):
        first = (i == 0)
        last = (i == n - 1)

        #
        phase = ephemeris.phase(t)
        phase -= max(np.floor(phase[0]) + 1, 0)
        if np.all(phase < 0):
            phase += 1

        ebc = ax.errorbar(phase[s], y[s], u if u is None else u[s],
                          color=colours[i], **opts)

        xlim = [min(xlim[0], phase[0]),
                max(xlim[1], phase[-1])]
        ylim = [min(ylim[0], y.min()),
                max(ylim[1], y.max())]

        # ticks
        ax.tick_params('y', which='minor', length=2.5, left=True, right=True)
        ax.tick_params('y', which='major', length=5, left=True, right=True)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

        if last:
            ax.tick_params('x', which='minor', length=2.5, bottom=(not first),
                           top=(not last))
            ax.tick_params('x', which='major', length=5, bottom=(not first),
                           top=(not last))
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        else:
            ax.tick_params('x', length=0)

        # remove top & bottom spines
        if not first:
            ax.spines['top'].set_visible(False)

        if not last:
            ax.spines['bottom'].set_visible(False)
            ax.xaxis.set_ticklabels([])

        ax.tick_params(labelright=True, labelleft=True)
        ax.grid(True)

    # axes limits
    stretch = np.ptp(xlim) * 0.025
    xlim = np.add(xlim, [-stretch, stretch])
    ylim[1] *= ylim_shrink
    for ax in axes:
        ax.set(xlim=xlim, ylim=ylim)

    # axes[0].set_ylim(-0.15, 1.65)

    # x label
    axes_label_font_spec = dict(weight='bold', size=14)
    ax.set_xlabel('Orbital Phase', fontdict=axes_label_font_spec)

    # y label
    y_middle = 0.5  # (fig.subplotpars.top - fig.subplotpars.bottom) / 2
    for x, va in zip((0.01, 1), ('top', 'bottom')):
        fig.text(x, y_middle, 'Relative Flux', axes_label_font_spec,
                 rotation=90, rotation_mode='anchor',
                 ha='center', va=va)

    # top ticks
    # axp.xaxis.set_ticks(np.r_[-2.5:3.5:0.5])
    axp.set_xlabel('Time (hours)', fontdict=dict(weight='bold'))
    axp.tick_params('x', which='minor', length=2.5, bottom=False,
                    top=True)
    return fig
