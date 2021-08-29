"""
Plotting interactive Time Frequency Representations of Time Series data
"""


# third-party
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mimage
from matplotlib import gridspec, ticker
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory as btf

# local
from recipes.dicts import AttrDict, AttrReadItem
from scrawl.ticks import ReciprocalFormatter
from scrawl.connect import ConnectionMixin, mpl_connect

# relative
from ..smoothing import smoother


# TODO: limit frequency axes at 0 when zooming / panning etc
# TODO: redo ticklabel precision when zoomed ???
# TODO: plot inset of lc on hover
# FIXME: unintended highlight when zooming.
# FIXME: immediately new highlight after selection

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


class TimeFrequencyBase:
    """Base class for Time Frequency plots"""

    ts_props = dict(color='g', ms=1.5)
    pg_props = dict(color='g')
    cb_props = {}
    hatch_props = dict(facecolor='none',
                       edgecolor='r',
                       alpha=0.65,
                       linewidth=1,
                       hatch='//')

    def __init__(self, spectrogram, ts=True, pg=True, info=False, cmap=None,
                 percentiles=(25, 50, 75)):
        """ """
        # assert isinstance(spectrogram, Spectrogram)
        self.spec = spectrogram

        ts_props = {**self.ts_props,
                    **(ts if isinstance(ts, dict) else {})}
        pg_props = {**self.pg_props,
                    **(ts if isinstance(pg, dict) else {})}

        # cmap = plt.get_cmap(cmap)
        self.q_levels = np.array(percentiles)
        self.figure, axes = self.setup_figure(bool(ts), bool(pg), bool(info))
        self.axes = AxesContainer(axes)
        self.info_text = self.get_info_text() if info else None

        art = self.plot(cmap, ts_props, pg_props)
        self.art = ArtistContainer(art)

        self.background = None
        self._need_save = False

    def setup_figure(self, show_ts=True, show_spec=True, show_info=True,
                     figsize=(11, 6.53), gridspec_kws=None):
        """Setup figure geometry"""

        # TODO limit axes to lower 0 Hz??  OR hatch everything below this
        # somehow
        # NOTE: You will need a custom transformation to implement this.
        # Something that does transAxes for the lower limit, but transData for
        # the upper MAYBE ask on SO??

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(
            100, 100,
            **dict(gridspec_kws or {},
                   hspace=0.02,
                   wspace=0.005,
                   top=0.925,
                   bottom=0.08,
                   left=0.05,
                   right=0.915)  # space for cbar ticks
        )

        # optionally display various axes
        axes = AttrDict()
        if show_spec:
            # map and ts, and spec
            w1 = 80
            h1 = 30 * show_ts
            h2 = 99
        else:
            #  no spec
            w1 = 98
            h1 = 20 * show_ts
            h2 = 100

        # add the subplots
        axes.map = fig.add_subplot(gs[h1:h2, :w1])
        axes.cbar = fig.add_subplot(gs[h2:, w1:])
        if show_ts:
            axes.ts = fig.add_subplot(gs[:h1, :w1], sharex=axes.map)
        if show_spec:
            axes.spec = fig.add_subplot(gs[h1:h2, w1:],
                                        sharey=axes.map)
        if show_info:
            axes.info = fig.add_subplot(gs[:h1, w1:])
            axes.info.set_axis_off()

        # options for displaying various parts
        axes.map.tick_params(which='both', labeltop=False, direction='out')
        axes.map.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        axes.map.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        axes.map.yaxis.set_tick_params('minor', labelsize=8)

        # Coordinate display format
        axes.map.format_coord = self.format_coord_map

        # setup_ticks
        if axes.ts:
            # set major/minor xticks invisible on light curve plot
            axes.ts.tick_params(axis='x', which='both',
                                top=True, labeltop=True,
                                labelbottom=False,
                                direction='inout', pad=0)
            axes.ts.tick_params(axis='y', which='both',
                                right=True, direction='inout')

            axes.ts.set_ylabel('Signal')   # TODO: units!!!!
            axes.ts.grid()

        # Get label for power values
        cbar_label = self.spec.get_ylabel()

        if axes.spec:
            # show Period as ticks on right spine of y
            # FIXME: ticks WRONG after zoom !
            axes.spec.parasite = axp = axes.spec.twinx()
            axp.yaxis.set_major_formatter(ReciprocalFormatter())
            axp.yaxis.set_tick_params(left=False, labelleft=False,
                                      right=False)

            # set yticks invisible on frequency spectum plot
            axes.spec.xaxis.offsetText.set_visible(False)

            # axes.spec.yaxis.set_tick_params()
            axes.spec.tick_params(which='both',
                                  left=False, labelleft=False,
                                  right=True,
                                  bottom=False, labelbottom=False,  # cbar gets these
                                  top=True,
                                  direction='inout')
            axes.spec.yaxis.set_label_position('right')
            axes.spec.set_ylabel('Period (s)', labelpad=50)

            # axes.spec.set_xlabel(cbar_label, labelpad=25)
            axes.spec.grid()
            axes.spec.format_coord = format_coord_spec
            axes.cbar.set_xlabel(cbar_label)

        else:
            axes.cbar.yaxis.set_label_position('right')
            axes.cbar.set_ylabel(cbar_label)

        axes.map.set_xlabel('Time (s)')
        axes.map.set_ylabel('Frequency (Hz)')

        return fig, axes

    def plot(self, cmap, ts_props=None, pg_props=None):
        
        spec = self.spec
        frq, pwr = spec.frq, spec.power
        valid = frq > spec.fRayleigh
        # NOTE: we intentionally do not mask power values below fRayleigh, even
        # though they are not physicaly meaningful because this often leads to
        # the appearance of a false "peak" at low frequencies.  Instead, we
        # hatch everything below the Rayleigh frequency.

        # guess reasonable colour limits
        plim = (0.25, 99.9)  # colour limits as percentile of power value
        clim = np.percentile(pwr[:, valid], plim)

        art = AttrDict()

        # Plot TFR image
        tlims = spec._ts.t[[0, -1]]
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
                                             **self.hatch_props)

        if self.axes.ts:
            # Plot time series
            tsp = spec._ts.plot(self.axes.ts,
                                # FIXME: use DEFAULT values
                                plims=[(0, 100), (-1, 101)],
                                errorbar=(ts_props or {}))
            self.axes.ts.xaxis.set_label_position('top')

        if self.axes.spec:
            self.plot_pgram(pg_props, clim=clim)

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

    def plot_pgram(self, pg_props=None, smoothing=5, clim=None):
        # Plot spectrum (median & inter-quartile range)
        pwr, frq = self.spec.power, self.spec.frq
        percentiles = np.percentile(pwr, self.q_levels, 0)
        ls = {50: '-'}
        art = {}
        for q, p in zip(self.q_levels, percentiles):
            art[f'pgram{q}'], = self.axes.spec.plot(
                smoother(p, smoothing), frq, ls.get(q, ':'),
                **(pg_props or {})
            )
        
        if clim is None:
            clim = (None, None)
        ylim = frq[[0, -1]]
        self.axes.spec.set(xlim=clim, ylim=ylim)
        self.axes.spec.parasite.set_ylim(ylim)  # FIXME
        # self.axes.spec.set_xscale('log')

        # hatch anything below self.fRayleigh
        rinv = Rectangle((0, 0), 1, self.spec.fRayleigh,
                         transform=btf(self.axes.spec.transAxes,
                                       self.axes.spec.transData),
                         **self.hatch_props)
        self.axes.spec.add_patch(rinv)

        # connect callbacks for limit change
        self.axes.spec.callbacks.connect('xlim_changed', self._set_clim)
        self.axes.spec.callbacks.connect(
            'ylim_changed', self._set_parasite_ylim)
        
        return art

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

    def get_info_text(self):

        spec = self.spec
        nwin, novr = spec.nwindow, spec.noverlap
        info = (
            rf'$\Delta t = {spec.dt:.3f}$ s ($f_s = {spec.fs:.3f}$ Hz)',
            f'window = {spec.window}',
            f'$n_w = {nwin:d}$ ({nwin * spec.dt:.1f} s)',
            f'$n_{{ovr}} = {novr:d}$ ({novr / nwin:.0f%%})'
        )
        if spec.pad:
            info += ('pad = %s' % str(spec.pad),)
        if spec.detrend:
            info += ('detrend = %s' % str(spec.detrend),)

        txt = '\n'.join(info)
        return self.axes.info.text(0.05, 1, txt,
                                   va='top',
                                   transform=self.axes.info.transAxes)

    def format_coord_map(self, x, y):
        pwr = self.spec.power
        frac = np.divide((x, y), (self.spec._ts.t[-1], self.spec.frq[-1]))
        col, row = np.round(frac * pwr.shape, 0).astype(int)
        nrows, ncols = pwr.shape
        if (0 < col < ncols) and (0 < row < nrows):
            z = pwr[row, col]
            return 't=%1.3f,\tf=%1.3f,\tPwr=%1.3g' % (x, y, z)
        return 'x=%1.3f, y=%1.3f' % (x, y)


class HoverSegment(ArtistContainer):

    props = {
        'ts':   dict(alpha=0.35),
        'map':  dict(lw=1,
                     ls='--')
    }
    spec_props = dict(alpha=0.65,
                      lw=1.5)

    def __init__(self, axes, color='r', **kws):
        self.figure = axes.ts.figure
        # instantaneous spectrum
        self.spectrum, = axes.spec.plot([], [], color, **self.spec_props)

        # window indicator on light curve axes
        # TODO:  show window shape
        for name in ('ts', 'map'):
            ax = axes[name]
            self[name] = rect = \
                Rectangle((0, 0), 0, 1,
                          color=color,
                          **self.props[name],
                          transform=btf(ax.transData,
                                        ax.transAxes))
            axes[name].add_patch(rect)

        # make map window transparent
        rect.set_facecolor('none')

    def update(self, x, width, spectrum):
        # update spectrum
        self.spectrum.set_data(spectrum)

        # update rectangle for highlighted span
        for art in self.values():
            art.set_x(x)
            art.set_width(width)

    def set_visible(self, b=True):
        self.ts.set_visible(b)
        self.map.set_visible(b)
        self.spectrum.set_visible(b)

    def draw(self):
        renderer = self.figure._cachedRenderer
        self.ts.draw(renderer)
        self.map.draw(renderer)
        self.spectrum.draw(renderer)

    def remove(self):
        self.ts.remove()
        self.map.remove()
        self.spectrum.remove()


class TimeFrequencyMap(TimeFrequencyBase, ConnectionMixin):
    """
    Time Frequency Representation (aka Power Spectral density map)
    Interactive plot elements live in this class
    """
    color_cycle = 'c', 'b', 'm', 'g', 'y', 'orange'

    def __init__(self, spectrogram, **kws):
        """ """
        # smoothing for displayed segment spectrum
        self.smoothing = kws.pop('smoothing', 0)

        self.hovering = False
        self.icolour = iter(self.color_cycle)
        # containers for highlighted segments
        self.hover = None  # placeholder
        self.windows = []

        TimeFrequencyBase.__init__(self, spectrogram, **kws)

        # initialize auto-connect
        ConnectionMixin.__init__(self, self.figure)

        # save background for blitting
        # self.canvas.draw()
        # self.save_background()

        # TODO: can you subclass widgets.cursor to emulate  desired behaviour??

    def plot(self, cmap, ts_props=None, pg_props=None):

        art = TimeFrequencyBase.plot(self, cmap, ts_props, pg_props)

        # Initiate elements for interactive display
        self.hover = art.hover = HoverSegment(self.axes)
        # TODO: only really need these upon connect

        return art

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
        spectrum = self.get_spectrum(ix, self.smoothing)
        x, _ = tspan = self.spec.t_seg[ix, (0, -1)]
        # NOTE padded values not included here
        # TODO: maybe some visual indicator for padding??
        self.hover.update(x, tspan.ptp(), spectrum)

    def highlight_section(self):
        # persistent highlight this window
        new = HoverSegment(self.axes,  next(self.icolour))
        new.update(self.hover.ts.xy[0], self.hover.ts.get_width(),
                   self.hover.spectrum.get_xydata().T)
        self.windows.append(new)

        self.hover.set_visible(False)

        self.draw_blit()
        new.draw()

        self.save_background()
        self.draw_blit()

    def draw_blit(self):
        self.canvas.restore_region(self.background)
        self.hover.draw()
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

        if self.ignore_hover(event):
            return

        # NOTE:  if the cursor "leaves" the axes directly onto another window,
        # the axis leave event is not triggered!!
        if not self.hovering:  # ie. legitimate axes enter
            # NOTE: Need to save the background here in case a zoom happened
            self.save_background()

        self.hovering = True
        self.hover.set_visible(True)
        self.draw_blit()

    @mpl_connect('axes_leave_event')
    def _leave_axes(self, event):
        # print('leave')
        if event.inaxes == self.axes.map:
            self.hover.set_visible(False)
            self.draw_blit()
            self.hovering = False

    @mpl_connect('button_press_event')
    def _on_button(self, event):
        if (event.button == 1) and (event.inaxes == self.axes.map):
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
        self.hover.set_visible(False)

        # reset colour cycle
        self.icolour = iter(self.color_cycle)

        for art in (self.hover, *self.windows):
            art.remove()

        self.windows = []

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


# class SpectralCoherenceMap(TimeFrequencyBase):

#     def __init__(self, t, signalA, signalB, **kws):
#         show_ts = kws.pop('show_ts', False)  # or ('ts_props' in kws)
#         ts_props = TimeFrequencyBase.ts_props.copy()
#         ts_props.update(kws.pop('ts_props', {}))
#         show_spec = kws.pop('show_spec', True)  # or ('pg_props' in kws)
#         pg_props = TimeFrequencyBase.pg_props.copy()
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
#         fig, axes = self.setup_figure(show_ts, show_spec)
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
