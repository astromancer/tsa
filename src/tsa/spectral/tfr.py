"""
Plotting interactive Time Frequency Representations of Time Series data
"""

# third-party libs
from recipes.dicts import AttrDict, AttrReadItem
import scipy
import numpy as np
import more_itertools as mit
import matplotlib.pyplot as plt
import matplotlib.image as mimage
from matplotlib import gridspec, ticker
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory as btf

# local libs
from scrawl.ticks import ReciprocalFormatter
from scrawl.utils import get_percentile_limits
from scrawl.connect import ConnectionMixin, mpl_connect

# relative libs
from ..smoothing import smoother
from . import Spectrogram, resolve_nwindow, resolve_overlap


# TODO: limit frequency axes at 0 when zooming / panning etc
# TODO: redo ticklabel precision when zoomed ???
# TODO: plot inset of lc on hover
# FIXME: unintended highlight when zooming.
# FIXME: immediately new highlight after selection

# FIXME: repeat code!
def format_coord_spec(x, y):
    # x = ax.format_xdata(x)
    # y = ax.format_ydata(y)

    p = 1. / y
    # f = self.time_axis.get_major_formatter().format_data_short(x)

    # FIXME: not formatting correctly
    return 'f = {:.3f}; p = {:.3f};\tPwr = {:.3g}'.format(y, p, x)
    # return 'f = {}; p = {};\ty = {}'.format( x, p, y )


class AxesContainer(AttrReadItem):
    pass

class ArtistContainer(AttrReadItem):
    pass


class TimeFrequencyMapBase:
    """Base class for Time Frequency plots"""

    ts_props = dict(color='g', marker='.', ms=1.5)
    pg_props = dict(color='g')
    cb_props = {}

    def __init__(self, spectrogram, ts=True, pg=True, info=True, cmap=None,
                 **kws):
        """ """
        assert isinstance(spectrogram, Spectrogram)
        self.spec = spectrogram

        if isinstance(ts, dict):
            ts_props = {**self.ts_props, **ts}

        if isinstance(pg, dict):
            pg_props = {**self.pg_props, **ts}

        # cmap = plt.get_cmap(cmap)
        self.spec_quant = np.array(kws.pop('spec_quant', (0.25, 0.50, 0.75)))

        self.figure, axes = self.setup_figure(bool(ts), bool(pg), bool(info))
        self.axes = AxesContainer(axes)
        # self.infoText = self.info_text()

        art = self.plot(spectrogram, cmap, ts_props, pg_props)
        self.art = ArtistContainer(art)

        self.background = None
        self._need_save = False

    def setup_figure(self, show_lc=True, show_spec=True, show_info=True,
                     figsize=(16, 8)):
        """Setup figure geometry"""

        # TODO limit axes to lower 0 Hz??  OR hatch everything below this somehow
        # NOTE: You will need a custom transformation to implement this.
        # Something that does transAxes for the lower limit, but transData for the upper
        # MAYBE ask on SO??

        fig = plt.figure(figsize=figsize)
        rows_cols = (100, 100)
        gs = gridspec.GridSpec(*rows_cols,
                               hspace=0.02,
                               wspace=0.005,
                               top=0.96,
                               left=0.05,
                               right=0.94 if show_spec else 0.94,
                               bottom=0.07)

        # if show_spec:
        # w1, w2 = 80, 20
        # else:
        # w1, _ = 98, 2

        axes = AttrDict()

        # optionally display various axes
        if show_spec and show_lc:
            w1, w2 = 80, 20
            h1, h2, h3 = 30, 69, 1
            axes.map = fig.add_subplot(gs[h1:-h3, :w1])
            axes.cbar = fig.add_subplot(gs[-h3:, w1:])
            axes.ts = fig.add_subplot(gs[:h1, :w1], sharex=axes.map)
            axes.spec = fig.add_subplot(gs[h1:-h3, w1:], sharey=axes.map)

            if show_info:
                axes.info = fig.add_subplot(gs[:h1, w1:])
                # axes.info.set_visible(False)
                axes.info.patch.set_visible(False)
                axes.info.tick_params(left=False, labelleft=False,
                                      bottom=False, labelbottom=False)
                for _, spine in axes.info.spines.items():
                    spine.set_visible(False)

        elif show_spec:
            w1, _ = 80, 20
            h1, h2, h3 = 0, 99, 1
            axes.map = fig.add_subplot(gs[h1:-h3, :w1])
            axes.cbar = fig.add_subplot(gs[-h3:, w1:])
            axes.ts = fig.add_subplot(gs[:h1, :w1],
                                      sharex=axes.map) if show_lc else None
            axes.spec = fig.add_subplot(gs[h1:h2, w1:],
                                        sharey=axes.map) if show_spec else None
            fig.subplots_adjust(right=0.95)  # space for cbar ticks

        elif show_lc:
            w1, _ = 98, 2
            h1, h2, h3 = 20, 80, 0
            axes.map = fig.add_subplot(gs[h1:-h3, :w1])
            axes.cbar = fig.add_subplot(gs[h1:, w1:])
            axes.ts = fig.add_subplot(gs[:h1, :w1],
                                      sharex=axes.map) if show_lc else None
            axes.spec = fig.add_subplot(gs[h1:h2, w1:],
                                        sharey=axes.map) if show_spec else None
        else:
            w1, _ = 98, 2
            h1, h2, h3 = 0, 100, 0
            axes.map = fig.add_subplot(gs[h1:-h3, :w1])
            axes.cbar = fig.add_subplot(gs[h1:, w1:])
            axes.ts = fig.add_subplot(gs[:h1, :w1],
                                      sharex=axes.map) if show_lc else None
            axes.spec = fig.add_subplot(gs[h1:h2, w1:],
                                        sharey=axes.map) if show_spec else None

            # add the subplots
            # axes.map  = fig.add_subplot(gs[h1:-h3, :w1])
            # axes.cbar   = fig.add_subplot(gs[h2:, w1:])
            # axes.ts   = fig.add_subplot(gs[:h1, :w1],
            # sharex=axes.map)    if show_lc    else None
            # axes.spec = fig.add_subplot(gs[h1:h2, w1:],
            # sharey=axes.map)    if show_spec  else None

        # options for displaying various parts
        minorTickSize = 8
        axes.map.tick_params(which='both', labeltop=False, direction='out')
        axes.map.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        axes.map.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        # MajForm = ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(x))
        # MajForm = formatter_factory( MajForm, tolerance=0.1 )
        # axes.map.yaxis.set_major_formatter(MajForm)
        # MinForm = ticker.FuncFormatter(lambda x, pos: '{:.1f}'.format(x).strip('0') )
        # axes.map.yaxis.set_minor_formatter( MajForm )
        # #ticker.ScalarFormatter()
        axes.map.yaxis.set_tick_params('minor', labelsize=minorTickSize)

        # Coordinate display format
        axes.map.format_coord = self.mapCoordDisplayFormatter

        # setup_ticks
        if axes.ts:
            # set major/minor xticks invisible on light curve plot
            axes.ts.tick_params(axis='x', which='both',
                                labelbottom=False, labeltop=True, top=True,
                                direction='inout', pad=0)
            # axes.ts.tick_params(axis='y', which='both', top=True)

            axes.ts.set_ylabel('Signal')   # TODO: units!!!!
            axes.ts.grid()

        # Get label for power values
        cb_lbl = self.spec.get_ylabel()

        if axes.spec:
            # set yticks invisible on frequency spectum plot
            axes.spec.xaxis.set_tick_params(labelbottom=False, labelleft=False,
                                            direction='inout', right=True)
            axes.spec.xaxis.offsetText.set_visible(False)

            # show Period as ticks on right spine of y
            self._parasite = axp = axes.spec.twinx()

            # FIXME: ticks WRONG after zoom FUCK!
            axp.yaxis.set_major_formatter(ReciprocalFormatter())
            axp.yaxis.set_tick_params(left=False, labelleft=False)
            axes.spec.yaxis.set_tick_params(left=False, labelleft=False)
            # TODO: same tick positions

            axes.spec.yaxis.set_label_position('right')
            axes.spec.set_ylabel('Period (s)', labelpad=40)

            # axes.spec.set_xlabel(cb_lbl, labelpad=25)
            axes.spec.grid()
            axes.spec.format_coord = format_coord_spec
            axes.cbar.set_xlabel(cb_lbl)

        else:
            axes.cbar.yaxis.set_label_position('right')
            axes.cbar.set_ylabel(cb_lbl)

        axes.map.set_xlabel('Time (s)')
        axes.map.set_ylabel('Frequency (Hz)')

        return fig, axes

    def plot(self, spec, cmap, ts_props=None, pg_props=None):
        """ """
        spec = self.spec
        t, signal = spec.t, spec.signal
        frq, pwr = spec.frq, spec.power
        valid = frq > spec.fRayleigh
        # NOTE: we intentionally do not mask values below fRayleigh, even though
        # they are not physicaly meaningful because this often leads to the
        # appearance of a false "peak" at low frequencies.  Instead, we  hatch
        # everything below the Rayleigh frequency.

        # guess reasonable colour limits
        plim = (0.25, 99.9)  # colour limits as percentile of power value
        clim = np.percentile(pwr[:, valid], plim)

        art = AttrDict()

        # Plot TFR image
        tlims = spec.t_seg[[0, -1], [0, -1]]
        flims = frq[[0, -1]]
        extent = np.r_[tlims, flims]
        art.image = image = mimage.NonUniformImage(self.axes.map,
                                                   origin='lower',
                                                   extent=extent,
                                                   cmap=cmap,
                                                   clim=clim)

        # data = im.cmap(P.T)
        # im.set_figure( fig_map )

        # detect_gaps
        # narrow_gap_inds = Spectral.detect_gaps(t, self.KCT*128/3600, 2)
        # narrow_gap_inds = narrow_gap_inds
        ##print( narrow_gap_inds )

        # fade gaps
        # alpha = data[...,-1]
        # alpha[:,narrow_gap_inds] = 0.25
        # alpha[:,narrow_gap_inds+1] = 0.25

        image.set_data(self.spec.tmid, frq, pwr.T)
        self.axes.map.images.append(image)
        # self.axes.map.set_xlim( t[0],t[-1] )
        # self.axes.map.set_ylim( frq[0],frq[-1] )

        # hatch anything below self.fRayleigh
        polycol = self.axes.map.fill_between(tlims, self.spec.fRayleigh,
                                             facecolor='none',
                                             edgecolors='r',
                                             linewidths=1,
                                             hatch='\\')

        if self.axes.ts:
            # Plot light curve
            art.ts, = self.axes.ts.plot(t, signal, **(ts_props or {}))
            # TODO: Uncertainties
            self.axes.ts.set(xlim=t[[0, -1]],
                             ylim=get_percentile_limits(signal, (0, 101)))

        if self.axes.spec:
            self.plot_pgram(pg_props)

            # show colourbar
            tmp = self.axes.cbar.get_xlabel()
            art.cbar = self.figure.colorbar(image,
                                            ticks=self.axes.spec.get_xticks(),
                                            cax=self.axes.cbar,
                                            orientation='horizontal',
                                            **self.cb_props)
            # redo the labels (which the previous line would have killed)
            self.axes.cbar.set_xlabel(tmp)

        else:
            # TODO: 'setup alt cbar'
            tmp = self.axes.cbar.get_ylabel()
            art.cbar = self.figure.colorbar(image, cax=self.axes.cbar,
                                            **self.cb_props)
            # ticks=self.axes.spec.get_xticks(),
            # set the labels (which the previous line killed)
            self.axes.cbar.set_ylabel(tmp)

        # TODO: MOVE TO SUBCLASS ?
        self.axes.map.callbacks.connect('xlim_changed', self.save_background)
        self.axes.map.callbacks.connect('ylim_changed', self.save_background)
        self.axes.map.callbacks.connect('ylim_changed', self._set_parasite_ylim)
        
        return art

    def plot_pgram(self, pg_props=None, smoothing=5):
        # Plot spectrum (median & inter-quartile range)
        pwr, frq = self.spec.pwr, self.spec.frq
        quantiles = np.percentile(pwr, self.spec_quant * 100, 0)
        self.pwr_p25, self.pwr_p50, self.pwr_p75 = quantiles

        self.axes.spec.plot(smoother(self.pwr_p25, smoothing), frq, ':',
                            smoother(self.pwr_p50, smoothing), frq, '-',
                            smoother(self.pwr_p75, smoothing), frq, ':',
                            **(pg_props or {}))
        # HACK
        # self.axes.spec.plot(smoother(self.pwr_p75, 5), frq, '-', **pg_props)

        self.axes.spec.set(xlim=self.art.image.get_clim(),
                           ylim=frq[[0, -1]])
        self.axes.spec.parasite.set_ylim(frq[[0, -1]])  # FIXME

        # self.axes.spec.set_xscale('log')

        # hatch anything below self.fRayleigh
        rinv = Rectangle((0, 0), 1, self.spec.fRayleigh,
                         facecolor='none', edgecolor='r',
                         linewidth=1, hatch='\\',
                         transform=btf(self.axes.spec.transAxes,
                                       self.axes.spec.transData))
        self.axes.spec.add_patch(rinv)

        # connect callbacks for limit change
        self.axes.spec.callbacks.connect('xlim_changed', self._set_clim)
        self.axes.spec.callbacks.connect('ylim_changed', self._set_parasite_ylim)

    def save_background(self, _ignored=None):
        # save_background
        # print('SAVING BG')
        self.background = self.figure.canvas.copy_from_bbox(self.figure.bbox)

    def _set_parasite_ylim(self, ax):
        # FIXME: obviate by using TimeFreqDualAxes
        # print('_set_parasite_ylim', ax.get_ylim())
        self.axes.spec.parasite.set_ylim(ax.get_ylim())
        # print('YO')
        # print(self._parasite.get_ylim())
        # print('!!')

        self._need_save = True
        # print('Done')

    def _set_clim(self, ax):
        # print('_set_clim', ax.get_ylim())
        xlim = ax.get_xlim()
        self.art.image.set_clim(xlim)
        self.art.cbar.set_ticks(self.axes.spec.get_xticks())
        self._need_save = True
        # self.save_background()

    def info_text(self):

        spec = self.spec
        nwin, novr = spec.nwindow, spec.noverlap
        info = (
            f'$\Delta t = {spec.dt:.3f}$ s ($f_s = {spec.fs:.3f}$ Hz)',
            f'window = {spec.window}',
            f'$n_w = {nwin:d}$ ({nwin * spec.dt:.1f} s)',
            f'$n_{{ovr}} = {novr:d}$ ({novr / nwin:.0f%%})'
        )
        if spec.pad:
            info += ('pad = %s' % str(spec.pad),)
        if spec.detrend:
            info += ('detrend = %s' % str(spec.detrend),)

        txt = '\n'.join(info)
        return self.axes.info.text(0.05, 1, txt, va='top',
                                   transform=self.axes.info.transAxes)

    def mapCoordDisplayFormatter(self, x, y):
        pwr = self.spec.power
        frac = np.divide((x, y), (self.spec.tmid[-1], self.spec.frq[-1]))
        col, row = np.round(frac * pwr.shape, 0).astype(int)
        nrows, ncols = pwr.shape
        if (0 < col < ncols) and (0 < row < nrows):
            z = pwr[row, col]
            return 't=%1.3f,\tf=%1.3f,\tPwr=%1.3g' % (x, y, z)
        return 'x=%1.3f, y=%1.3f' % (x, y)


class TimeFrequencyMap(TimeFrequencyMapBase, ConnectionMixin):
    """
    Time Frequency Representation (aka Power Spectral density map)
    Interactive plot elements live in this class
    """
    color_cycle = 'c', 'b', 'm', 'g', 'y', 'orange'

    _span_ts_props = dict(alpha=0.35)
    _span_map_props = dict(facecolor='none',
                           lw=1,
                           ls='--')
    _ispec_prop = dict(alpha=0.65,
                       lw=1.5)

    def __init__(self, spec, **kws):
        """ """
        # smoothing for displayed segment spectrum
        self.smoothing = kws.pop('smoothing', 0)

        TimeFrequencyMapBase.__init__(self, spec, **kws)

        # scale segment spectrum to sum of median (for display)
        # self.scaling = 1. / self.pwr_p50.sum()

        # initialize auto-connect
        ConnectionMixin.__init__(self, self.figure)

        # save background for blitting
        # self.canvas.draw()
        # self.save_background()

        # TODO: can you subclass widgets.cursor to emulate  desired behaviour??
        self.hovering = False
        self.icolour = iter(self.color_cycle)
        self.spans_ts = []  # container for highlighted segments
        self.spans_map = []
        self.spectra = []

    def plot(self, spec, cmap, ts_props=None, pg_props=None):

        TimeFrequencyMapBase.plot(self, spec, cmap, ts_props, pg_props)

        # Initiate elements for interactive display
        # TODO: only really need these upon connect
        # instantaneous spectrum
        self.ispec, = self.axes.spec.plot([], [], 'r', **self._ispec_prop)

        # window indicator on light curve axes
        # TODO:  Alpha channel for window shape
        self.span_lc_transform = btf(self.axes.ts.transData,
                                     self.axes.ts.transAxes)
        self.span_lc = Rectangle((0, 0), 0, 1,
                                 color='r',
                                 **self._span_ts_props,
                                 transform=self.span_lc_transform)
        self.axes.ts.add_patch(self.span_lc)

        # window indicator on map axes
        self.span_map_transform = btf(self.axes.map.transData,
                                      self.axes.map.transAxes)
        self.span_map = Rectangle((0, 0), 0, 1,
                                  edgecolor='r',
                                  **self._span_map_props,
                                  transform=self.span_map_transform)
        self.axes.map.add_patch(self.span_map)

    def get_spectrum(self, ix, smoothing=0, scaling=1.):  # misnomer?
        """ """
        data = self.spec.power[ix]
        if smoothing:
            data = smoother(data, smoothing)
        if scaling:
            data = data * scaling

        return np.array([data, self.spec.frq])

    def update(self, ix):

        # update spectrum
        spectrum = self.get_spectrum(ix, self.smoothing, )
        self.ispec.set_data(spectrum)

        # update rectangle for highlighted span
        tspan = self.spec.t_seg[ix, (0, -1)]
        # NOTE padded values not included here
        # TODO: maybe some visual indicator for padding??
        x = tspan[0]
        width = tspan.ptp()
        self.span_lc.set_x(x)
        self.span_lc.set_width(width)

        self.span_map.set_x(x)
        self.span_map.set_width(width)

    def highlight_section(self):
        # persistent spectrum for this window
        colour = next(self.icolour)
        spectrum = self.ispec.get_xydata().T  # instantaneous spectum
        line, = self.axes.spec.plot(*spectrum, color=colour, **self._ispec_prop)
        self.spectra.append(line)

        # persistent highlight this window
        span_lc = Rectangle(self.span_lc.xy, self.span_lc.get_width(), 1,
                            color=colour,
                            **self._span_ts_props,
                            transform=self.span_lc_transform)
        self.axes.ts.add_patch(span_lc)
        self.spans_ts.append(span_lc)

        span_map = Rectangle(self.span_map.xy, self.span_map.get_width(), 1,
                             edgecolor=colour,
                             **self._span_map_props,
                             transform=self.span_map_transform)
        self.axes.map.add_patch(span_map)
        self.spans_map.append(span_map)

        self.span_map.set_visible(False)
        self.span_lc.set_visible(False)
        self.ispec.set_visible(False)
        self.draw_blit()

        renderer = self.figure._cachedRenderer
        line.draw(renderer)
        span_lc.draw(renderer)
        span_map.draw(renderer)
        self.save_background()
        self.draw_blit()

    def draw_blit(self):
        renderer = self.figure._cachedRenderer
        self.canvas.restore_region(self.background)
        self.ispec.draw(renderer)
        self.span_lc.draw(renderer)
        self.span_map.draw(renderer)

        self.canvas.blit(self.figure.bbox)

        # @mpl_connect( 'draw_event' )
        # def _on_draw(self, event):
        # print( 'drawing:', event )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def ignore_hover(self, event):
        if event.inaxes != self.axes.map:
            return True

        return any(self.canvas.manager.toolbar._actions[button].isChecked()
                   for button in ('pan', 'zoom'))

    @mpl_connect('motion_notify_event')
    def _on_motion(self, event):
        if event.inaxes != self.axes.map:
            return

        ix = abs(self.spec.tmid - event.xdata).argmin()
        self.update(ix)
        self.draw_blit()

    @mpl_connect('axes_enter_event')
    def _enter_axes(self, event):
        # print('enter')
        # self.canvas.draw()

        if not self.ignore_hover(event):
            # NOTE:  if the cursor "leaves" the axes directly onto another window,
            # the axis leave event is not triggered!!
            if not self.hovering:  # i.e legitimate axes enter
                # NOTE: Need to save the background here in case a zoom happened
                self.save_background()

            self.hovering = True

            self.span_lc.set_visible(True)
            self.span_map.set_visible(True)
            self.ispec.set_visible(True)
            self.draw_blit()

    @mpl_connect('axes_leave_event')
    def _leave_axes(self, event):
        # print('leave')
        if event.inaxes == self.axes.map:
            self.span_lc.set_visible(False)
            self.span_map.set_visible(False)
            self.ispec.set_visible(False)
            self.draw_blit()

            self.hovering = False

    @mpl_connect('button_press_event')
    def _on_button(self, event):
        # print(event.button)

        if event.button == 1 and event.inaxes == self.axes.map:
            self.highlight_section()

        if event.button == 2:  # restart on middle mouse
            self.restart()

    @mpl_connect('key_press_event')
    def _on_key(self, event):
        print(event.key)

    @mpl_connect('draw_event')
    def _on_draw(self, event):
        # print('DRAWING', event)
        if self._need_save:
            self.save_background()
            # DOESN'T work since it gets executed before the draw is complete
            self._need_save = False

    def restart(self):
        self.span_lc.set_visible(False)
        self.span_map.set_visible(False)
        self.ispec.set_visible(False)

        # reset colour cycle
        self.icolour = iter(self.color_cycle)
        for art in mit.flatten([self.spans_ts, self.spans_map, self.spectra]):
            art.remove()

        self.spans_ts = []
        self.spans_map = []
        self.spectra = []

        self.canvas.draw()
        self.background = self.canvas.copy_from_bbox(self.figure.bbox)

        # def connect(self):
        # self.canvas.mpl_connect( 'motion_notify_event', self._on_motion )
        # self.canvas.mpl_connect( 'axes_enter_event', self._enter_axes )
        # self.canvas.mpl_connect( 'axes_leave_event', self._leave_axes )


TimeFrequencyRepresentation = TimeFrequencyMap


# class SpectralAudio(TimeFrequencyMap):
#     def __init__(self, t, signal, **kws):
#         TimeFrequencyMap.__init__(self, spec, **kws)

#     def main(self, segments):  # calculate_spectra
#         # calculate spectra
#         spec = scipy.fftpack.fft(segments)
#         # since we have real signals
#         self.spectra = spec = spec[..., :len(self.frq)]
#         power = np.square(np.abs(spec))
#         power = normaliser(power, self.segments, self.opts.normalise, self.dt,
#                            self.npadded)
#         return power

#     def resample_segment(self, ix, duration):
#         ''

#     def play_segment(self, ix):
#         ix = abs(self.tmid - t).argmin()


# class SpectralCoherenceMap(TimeFrequencyMapBase):

#     def __init__(self, t, signalA, signalB, **kws):
#         show_lc = kws.pop('show_lc', False)  # or ('ts_props' in kws)
#         ts_props = TimeFrequencyMapBase.ts_props.copy()
#         ts_props.update(kws.pop('ts_props', {}))
#         show_spec = kws.pop('show_spec', True)  # or ('pg_props' in kws)
#         pg_props = TimeFrequencyMapBase.pg_props.copy()
#         pg_props.update(kws.pop('pg_props', {}))

#         cmap = kws.pop('cmap', 'viridis')

#         self.t = t
#         self.signalA = signalA
#         self.signalB = signalB

#         # TODO: base method that does this for TFR & CSD
#         self.opts = SpectralCoherenceMap.defaults
#         self.check(t, signalA, **kws)

#         # timing stats
#         dt = self.check_timing(t, self.opts.dt)
#         self.dt = dt
#         self.fs = 1. / dt

#         # clean masked, fill gaps etc
#         t, signalA = self.prepare_signal(t, signalA, self.dt)
#         t, signalB = self.prepare_signal(t, signalB, self.dt)

#         self.nwindow = resolve_nwindow(self.opts.nwindow, self.opts.split, t,
#                                        self.dt)
#         self.noverlap = resolve_overlap(self.nwindow, self.opts.noverlap)
#         self.fRayleigh = 1. / (self.nwindow * dt)

#         # fold
#         self.t_seg, self.segAraw = self.get_segments(t, signalA, dt,
#                                                      self.nwindow,
#                                                      self.noverlap)
#         _, self.segBraw = self.get_segments(t, signalB, dt,
#                                             self.nwindow, self.noverlap)

#         # median time for each section
#         self.tms = np.median(self.t_seg, 1)

#         # FFT frequencies
#         nw = self.opts.apodise or self.nwindow
#         self.frq = np.fft.rfftfreq(nw, self.dt)
#         self.ohm = 2. * np.pi * self.frq  # angular frequencies

#         # pad, detrend, window
#         self.segmentsA = self.prepare_segments(self.segAraw)
#         self.segmentsB = self.prepare_segments(self.segBraw)

#         # calculate spectra
#         self.power = self.main(self.segmentsA, self.segmentsB)

#         # plot stuff
#         fig, axes = self.setup_figure(show_lc, show_spec)
#         self.axes.map, self.axes.ts, self.axes.spec, self.axes.cbar = axes
#         self.plot(axes, t, signalA, cmap, ts_props,
#                   pg_props)  # FIXME: signalB

#     def main(self, segA, segB):
#         # since we are dealing with real signals
#         specA = scipy.fftpack.fft(segA)
#         specA = specA[..., :len(self.frq)]
#         specB = scipy.fftpack.fft(segB)
#         specB = specB[..., :len(self.frq)]

#         csd = np.conjugate(specA) * specB
#         coh = np.abs(csd) ** 2 / (np.abs(specA) * np.abs(specB))

#         T = self.t.ptp()
#         power = T ** 2 * coh / (self.signalA.sum() * self.signalB.sum())

#         return power
