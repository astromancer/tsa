"""
Versatile functions for plotting time-series data
"""


# std
import numbers
import itertools as itt
from warnings import warn

# third-party
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.transforms import Affine2D, blended_transform_factory as btf
from loguru import logger
from mpl_toolkits.axes_grid1 import make_axes_locatable

# local
from recipes import api, dicts
from recipes.config import ConfigNode
from recipes.string import named_items
from scrawl.moves import MovableErrorbar
from scrawl.utils import get_percentiles
from scrawl.dualaxes import DateTimeDualAxes, DualAxes
from scrawl.ticks import (OffsetLocator, SexagesimalFormatter,
                          _rotate_tick_labels)

# relative
from .interface import Interface


# ---------------------------------------------------------------------------- #

# TODO:
#  alternatively make `class TimeSeriesPlot(Axes):` then ax.errorbar()
# NOTE: you can probs use the std plt.subplots machinery if you register your
#  axes classes


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


# ---------------------------------------------------------------------------- #
# Module config
CONFIG = ConfigNode.load_module(__file__)

# array printing (logging)
np.set_printoptions(threshold=100, precision=3, linewidth=120)

STYLE_KWS = {
    'line',
    'errorbar',
    'spans',
    'hist',
    'legend'
}

ALLOWED_KWS = {
    'timescale',
    'start',
    'axes_labels',
    'twinx',
    'plims',
    *STYLE_KWS
}

TWIN_AXES_CLASSES = {'sexa': DateTimeDualAxes}

# ---------------------------------------------------------------------------- #
N_MAX_TS_PLOT = 50


class TooManyToPlot(Exception):
    """
    Exception that occurs when user accidentally requests too many plots. The
    number at which this occurs is determined by the module variable
    `N_MAX_TS_PLOT`
    """


# ---------------------------------------------------------------------------- #
# @dataclass
# class DataPercentileAxesLimits:

#     lower: float = -0.05
#     upper: float = +100.05

#     def get(self, data, e=()):
#         return get_percentiles(data, (self.lower, self.upper), e)


# ---------------------------------------------------------------------------- #

def _set_defaults(props, defaults):
    for k, v in defaults.items():
        props.setdefault(k, v)


def resolve_kws(kws, strict=False):

    if strict and (invalid := set(kws.keys()) - set(ALLOWED_KWS)):
        raise KeyError(
            f"Invalid {named_items(invalid, 'keyword', fmt=repr)}.\n"
            f"Only the following keywords are recognised: {ALLOWED_KWS}."
        )

    kws, user_styles = dicts.split(kws, *STYLE_KWS)
    styles = dict(zip(STYLE_KWS, map(CONFIG.get, STYLE_KWS)))

    # deal with nested dicts
    for key, val in user_styles.items():
        styles[key].update(val)

    return kws, ConfigNode(styles)


# ---------------------------------------------------------------------------- #

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
    return (is_null(x) or is1d(x)) or (len(set(map(len, x))) == 1)


def is1d(x):
    return isinstance(x[0], (numbers.Real, np.ma.core.MaskedConstant))


# ---------------------------------------------------------------------------- #

def resolve_labels(labels, signals):
    # check labels
    if is_null(labels):
        return []

    if isinstance(labels, str):
        return [labels]  # check if single label given

    if len(labels) != len(signals):
        warn('Number of labels does not match number of time series.')

    return labels


def _parse_input(data, labels):
    # data := [times], signals, [y-errors, x-errors]
    n = len(data)
    if 1 > n > 4:
        raise ValueError(f'Invalid number of arguments: {n}')

    # signals only
    if n == 1:
        # Assume here each row gives individual signal for a TS
        signals = data[0]

        # check for structured data (dict keyed on labels and containing data)
        if isinstance(signals, dict):
            #
            yield from _parse_input(list(zip(*signals.values())),
                                    list(signals.keys()))
            return

        data = (), signals

    # times, signals, [y-errors, x-errors] given
    for i, d in enumerate(data, 1):
        yield (d, ())[is_null(d)]

    for _ in range(i, 4):
        yield ()

    yield labels


def auto_transpose(array, shortest=1):

    # NOTE: atleast_2d may return masked array
    array = np.atleast_2d(array)
    assert array.ndim == 2

    if np.argmax(array.shape) != shortest:
        logger.info('Transposing input data to column-variate form: {}',
                    array.shape[::-1])
        return array.T

    return array


def get_data(data, labels, thin=1, max_points=None, t0=None, tscale=None):

    # parse input args: times, signals, y_err, x_err
    data = resolve_data(data, labels)

    # zip_longest in case errors or times are empty sequences
    data = itt.zip_longest(*data, fillvalue=())

    # performance tradeoff: thin data since plotting many point is a bottleneck
    data = _thin_data(data, thin, max_points)

    #
    return _scale_time_vectors(data, t0, tscale)


def resolve_data(data, labels):
    """parse data arguments"""

    times, signals, y_err, x_err, labels = _parse_input(data, labels)
    #
    if is_uniform(signals):
        # auto transpose so columns are variates, rows are data
        signals = auto_transpose(signals)
        times = auto_transpose(times) if len(times) else np.arange(signals.shape[1])

    elif (len(times) != len(signals)):
        # ragged signal list, explicit time stamps
        raise ValueError(
            'Number of time and signal vectors do not correspond. Please '
            'provide explicit time stamps for all signal vectors when plotting '
            'multiple time series with unequal size.'
        )

    # safety breakout for erroneous arguments that can trigger very slow
    # plotting loop
    m = len(signals)
    if m > N_MAX_TS_PLOT:
        raise TooManyToPlot(
            f'Received {m} time series to plot. This is probably not what you '
            'wanted. Stopping since safety limit is currently set to '
            f'{N_MAX_TS_PLOT}. This is to avoid accidental compute intensive '
            'commands from overwhelming system resources.'
        )

    t0 = times if is1d(times) else times[0]
    yield check_data('time', times, signals, t0)
    yield signals
    yield check_data('y_err', y_err, signals)
    yield check_data('x_err', x_err, signals)
    yield resolve_labels(labels, signals)


def check_data(name, array, signals, fill=()):
    # for name, vector in kws.items():
    if is_null(array):
        yield fill
        return

    if is_uniform(signals) and is_uniform(array):
        array = auto_transpose(array)
        signals = auto_transpose(signals)

    n = len(signals)
    if n < (m := len(array)):
        raise ValueError(
            f'Superfluous {name} vector(s). Received {m}, expected {n}.'
        )

    for vector, signal in itt.zip_longest(array, signals, fillvalue=fill):
        n = len(signal)
        if not is_null(vector) and ((m := len(vector)) != n):
            raise ValueError(
                f'Unequal number of points between signal ({n}) and {name} '
                f'vectors ({m}).'
            )
        yield vector


def _resolve_data_step(n, thin, max_points):

    if thin == 1 and max_points and (max_points > 0) and n > max_points:
        return n // max_points

    return thin


def _thin_data(data, thin, max_points):

    if (thin == 1 and max_points):
        data = list(data)
        n = sum(len(v[0]) for v in data)

        max_points = int(max_points)
        thin = _resolve_data_step(n, thin, max_points)
        if thin != 1:
            logger.debug('Thinning plot data by {} so we have fewer than {} '
                         'plot points.', thin, max_points)

    #
    thin = int(thin)
    if thin == 1:
        yield from data
        return

    if not max_points and (thin != 1):
        logger.debug('Thinning plot data by {}.', thin)

    for *vectors, label in data:
        yield (*_thinner(thin, *vectors), label)


def _thinner(thin, x, y, y_err, x_err):
    # thin out
    yield x[::thin]
    yield y[::thin]
    yield y_err[::(thin, 1)[is_null(y_err)]]
    yield x_err[::(thin, 1)[is_null(x_err)]]


def _scale_time_vectors(data, t0, tscale):
    if t0 is None and tscale is None:
        yield from data
        return

    for t, *rest in data:
        if t0 is not None:
            if isinstance(t0, list):
                t0 = t[t0[0]]
            if isinstance(t0, numbers.Real):
                t = t - t0
            else:
                raise ValueError(f'Bad {t0 = }')

        if tscale is not None:
            t = t * tscale

        yield t, *rest


def sanitize_data(t, signal, y_err, x_err):
    """
    Clean up data for single time series before plotting.

    Parameters
    ----------
    t
    signal
    y_err
    x_err

    Returns
    -------

    """

    # mask nans
    signal = np.ma.MaskedArray(signal, ~np.isfinite(signal))
    if is_null(t):
        t = np.arange(len(signal))

    y_err = None if is_null(y_err) else y_err
    x_err = None if is_null(x_err) else x_err

    return (t, signal, y_err, x_err)


def get_line_colours(n, colours, cmap):
    # Ensure we plot with unique colours

    # rules here are:
    # `cmap` always used if given
    # `colours` always used if given, except if `cmap` given
    #   warning emitted if too few colours - colour sequence will repeat
    too_few_colours = len(plt.rcParams['axes.prop_cycle']) < n
    if (cmap is not None) or ((colours is None) and too_few_colours):
        cm = plt.get_cmap(cmap)
        colours = cm(np.linspace(0, 1, n))  # linear colour map for ts

    elif (colours is not None) and (len(colours) < n):
        warn('Colour sequence has too few colours (%i < %i). Colours '
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
        return CONFIG.axes_labels.values()

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
        if axes_cls := TWIN_AXES_CLASSES.get(twinx):
            # make twin
            fig = plt.figure(figsize=figsize)
            ax = axes_cls(fig, 1, 1, 1, **kws)
            ax.setup_ticks()
            fig.add_subplot(ax)
            return fig, ax

        #
        warn('Option %r not understood for argument `twinx`. Ignoring.', twinx)

    return plt.subplots(figsize=figsize)


# ---------------------------------------------------------------------------- #
class TimeSeriesPlot(Interface):
    """
    Multivatiate time series plotting.
    """

    # TODO: evolve to multiprocessed TS plotter.

    def __init__(self, title='', hist=(), plims=CONFIG.plims,
                 colors=None, cmap=None, max_points=1e4, **kws):

        self.parent = None
        self.title = str(title)
        self.fig = self.ax = self.hax = None
        self.colors = colors
        self.cmap = cmap

        self.art = []
        self._show_hist = bool(hist)
        self.hist = []
        self._linked = []
        # _proxies = []

        # max number of plot points
        self.max_points = int(max_points)

        # axes limits
        plims = np.array(plims)
        if plims.shape == (2, ):
            plims = np.array([plims, plims])

        assert plims.shape == (2, 2), f'{plims.shape}'
        self.plims = plims
        self.zorder0 = 10

        self.styles = {}
        self.kws = kws

        # default layout for pretty figures
        # left, bottom, right, top = [0.025, 0.01, 0.97, .98]
        # fig.tight_layout(rect=rect)
        # return fig, ax

    def __iter__(self):
        yield self.fig
        yield self.ax

    @api.synonyms({'(histogram)|(marginal)': 'hist',
                   'time0': 't0',
                   't(ime)?_?scale': 'tscale'})
    def __call__(self, *data, ax=None,
                 t0=None, tscale=None,
                 hist=False, show_masked=False, thin=1,
                 labels=(), offsets=(), draggable=False,
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

        # TODO: docstring
        # TODO: get this to work with astropy time objects
        # TODO: astropy.units ??

        # Check keyword argument validity
        kws, styles = resolve_kws(kws)
        show_hist = bool(hist)

        # setup figure if needed
        self.fig, self.ax, self.hax = self.setup_figure(ax, self._show_hist)

        self.logger.info(f'{self.xlim = }, {self.ylim = }')

        # parse input args: times, signals, y_err, x_err
        data = self.get_data(data)
        data = get_data(data, labels, thin, self.max_points, t0, tscale)

        # Plot
        for x, y, σy, σx, label in data:
            # note: errors or times are empty sequences here if not user provided
            logger.opt(lazy=True).debug(
                '{}', lambda: (f'Now plotting {label or ""}:'
                               f'\n{x = },\n {y = },\n {σy = },\n {σx = }')
            )

            self.plot(x, y, σy, σx, label, thin, show_masked, show_hist,
                      styles=styles, **kws)

        # set auto-scale limits
        for xy in 'xy':
            lim = getattr(self, f'{xy}lim')
            lim = np.where(np.isfinite(lim), lim, [None, None])
            self.ax.set(**{f'{xy}lim': lim})

        # add text labels
        # self.set_labels(title, kws.axes_labels,
        #                kws.twinx, relative_time)

        # -------------------------------------------------------------------- #
        # Setup canvas interaction

        # FIXME: offsets should work even when not draggable!!
        if draggable and not show_hist:
            # FIXME: maybe warn if both draggable and show_hist
            # make the artists draggable
            self.plots = MovableErrorbar(self.art, offsets=offsets,
                                         linked=self._linked,
                                         **styles.legend)
            # TODO: legend with linked plots!

        elif labels:
            self.ax.legend(self.art, labels, **styles.legend)
            # self._make_legend(ax, self.art, labels)

        return self

    plot = __call__

    def setup_figure(self, ax, show_hist, **kws):
        """Setup figure geometry"""

        if ax is None:
            self.xlim = np.array([np.inf, -np.inf])
            self.ylim = np.array([np.inf, -np.inf])
        # else:

        # get / create figure, axes
        fig, ax = get_axes(ax, kws.pop('figsize', None))

        # Add subplot for histogram
        # FIXME: leave space on the right of axes for offsets if draggable
        hax = None
        if show_hist:
            divider = make_axes_locatable(ax)
            hax = divider.append_axes('right', size='25%', pad=0.,
                                      sharey=ax)
            hax.grid()
            hax.yaxis.tick_right()

        # Set axes props
        ax.grid()            # which='both' b=True
        ax.set(**kws)

        return fig, ax, hax

    def plot(self, x, y, y_err, x_err, label, thin=1,
             show_masked=False, show_hist=False, relative_time=False,
             styles=None, **kws):

        # if (y_err is not None) & (show_errors == 'contour'):
        #     uncertainty_contours(self.ax, x, y, y_err, styles, lw=1)

        # clean
        data = sanitize_data(x, y, y_err, x_err)

        # thin out
        if (thin := int(thin)) > 1:
            data = _thinner(thin, *data)

        # plot
        kws = dict(label=(label or None), zorder=self.zorder0, **kws)
        if len(y_err) or len(x_err):
            # plot errorbars
            art = self.ax.errorbar(*data, **{**kws, **styles.errorbar})
        else:
            # plot line
            art = self.ax.plot(*data[:2], **{**kws, **styles.line})

        # collect art
        self.art.append(art)

        # update axes limits
        self.set_limits(*data)

        # time axes offet
        if relative_time:
            self.ax.xaxis.major.formatter.set_useOffset(x[0])
            self.ax.xaxis.set_major_locator(OffsetLocator())

        # plot masked values with different style if requested
        if show_masked:
            self.plot_masked_points(x, y, show_masked)

        # plot histograms
        if show_hist:
            self.plot_histogram(y, **styles.hist)

        # update zorder for future plots to be behind first
        self.zorder0 = 1

        return art

    def plot_masked_points(self, t, signal, marker='x', color=None, **kws):
        # Get / Plot GTIs

        # msk_art = None
        # if how == 'span':
        #     self.plot_masked_intervals(ax, t, unmasked.mask)
        if marker is True:
            marker = 'x'
            color = color or 'r'

        #
        last = line, *_ = self.art[-1]
        if color is None:
            color = line.get_color()

        # plot masked points
        if ebar := self.ax.plot(t[signal.mask], signal[signal.mask].data,
                                color=color, marker=marker,
                                ls='None',  label='_nolegend_',
                                alpha=0.7):
            self.art.append(ebar)
            self._linked.append((last, ebar))

        # raise NotImplementedError

    def plot_histogram(self, signal, **props):
        #
        self.hist.append(
            self.hax.hist(np.ma.compressed(signal), **props)
        )
        self.hax.grid(True)

    def set_limits(self, x, y, y_err, x_err):
        # set axes view limits
        for xy, v, p, e in zip('xy', (x, y), self.plims, (x_err, y_err)):
            datalim = get_percentiles(v, p, e)
            current = getattr(self, f'{xy}lim')
            low, hi = zip(datalim, current)
            new_lim = [min(low), max(hi)]

            # check compat with scale,
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
                    warn('Requested negative limits on log scaled axis. '
                         'Using smallest positive data element as lower '
                         'limit instead.')
                    new_lim[0] = y[~neg].min()

            # set new limits
            self.logger.debug('plims = {}, lim = {}', p, new_lim)
            setattr(self, f'{xy}lim', new_lim)

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

    def acf(self, *data, **kws):
        data = self.get_data(data)

        self.plims = np.array([(-0.1, 100.1), (-0.2, 100)])
        tsp = self.plot(*data,
                        errorbar={'ms': 1},
                        **kws)
        ax = tsp.ax

        scales = [1.959963984540054, 2.5758293035489004]
        ci = np.array([[-1], [1]]) * scales / np.sqrt(len(data[0]))
        ls = [':', '--'] * 2
        ax.hlines(ci.ravel(), 0, 1, ['0.65'], ls=ls, lw=1,
                  transform=btf(ax.transAxes, ax.transData))
        ax.set(xlabel='Time Lag (s)', ylabel='Auto-Correlation')

        return self


# ---------------------------------------------------------------------------- #

def convert_mask_to_intervals(a, mask=None):
    """Return index tuples of contiguous masked values."""
    if mask is None:
        mask = a.mask
        # NOTE: If a is a masked array, this function returns masked values!!!

    if ~np.any(mask):
        return ()

    import more_itertools as mit
    w, = np.where(mask)
    l1 = w - np.roll(w, 1) > 1
    l2 = np.roll(w, -1) - w > 1
    idx = [w[0]] + mit.interleave(w[l2], w[l1]) + [w[-1]]
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

# def axes_utc_


def make_twin_relative(ax, offset=0, scale=1, tick_label_angle=0, **kws):
    #  date=None,

    # make transform
    axp = ax.twin(Affine2D().translate(-offset, 0).scale(scale).inverted())
    # == Affine2D().scale(1/scale).translate(offset, 0)

    # make tick locs / format
    axp.xaxis.set_major_locator(ticker.MultipleLocator(30 * 60))
    axp.xaxis.set_major_formatter(
        SexagesimalFormatter(**{'precision': 'm0', 'unicode': True, **kws})
    )

    for axx in (ax, axp):
        axx.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        axx.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        # axx.xaxis.offsetText.set_visible(False)

    # ticks appearance
    ax.tick_params('x', which='both', bottom=True, labelbottom=True)
    ax.tick_params('y', which='both', left=True, labelleft=True)

    axp.tick_params('x', which='both', top=True, bottom=False,
                    labeltop=True, labelbottom=False)
    axp.tick_params('y', which='both', left=False,
                    right=True, labelright=True)

    # if date is not None:
#

    if tick_label_angle:
        _rotate_tick_labels(axp, tick_label_angle, False)

    return axp


def make_twin_phased(ax, period=1, phoff=0, tick_label_angle=0):
    return make_twin_relative(ax, -phoff, 1 / period / 86400, tick_label_angle)


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
    axp = make_twin_phased(ax, 45, ephemeris.P)
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
