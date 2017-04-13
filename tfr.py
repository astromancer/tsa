
import numpy as np
import scipy

import matplotlib.pyplot as plt
import matplotlib.image as mimage
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory as btf
from matplotlib import gridspec
from matplotlib import ticker

# from misc.meta import
# from superplot.spectra import formatter_factory

from grafico.ts import axes_limit_from_data
from grafico.misc import ConnectionMixin, mpl_connect
from grafico.dualaxes import ReciprocalFormatter

from .spectral import Spectral
from .smoothing import smoother
from .spectral import resolve_nwindow, resolve_overlap



# from recipes.string import minlogfmt

# from IPython import embed

# TODO: limit frequency axes at 0 when zooming / panning etc
# TODO: redo ticklabel precision when zoomed ???

# TODO: plot inset of lc on hover
# FIXME: unintended highlight when zooming.

# ===============================================================================
# FIXME: repeat code!
def format_coord_spec(x, y):
    # x = ax.format_xdata(x)
    # y = ax.format_ydata(y)

    p = 1. / y
    # f = self.time_axis.get_major_formatter().format_data_short(x)

    # FIXME: not formatting correctly
    return 'f = {:.3f}; p = {:.3f};\tPwr = {:.3g}'.format(y, p, x)
    # return 'f = {}; p = {};\ty = {}'.format( x, p, y )

# def format_coord_map(x, y):
    # x = ax.format_xdata(x)
    # y = ax.format_ydata(y)

    # p = 1. / y
    # # f = self.time_axis.get_major_formatter().format_data_short(x)
    #
    # # FIXME: not formatting correctly
    # return 'f = {:.6f}; p = {:.3f};\tPwr = {:.3f}'.format(y, p, x)

# ===============================================================================
# def logformat(x, _=None):
# return minlogfmt(x, 2, '\cdot')

# ****************************************************************************************************
class TimeFrequencyMapBase(Spectral):
    '''base class'''
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    color_cycle = 'c', 'b', 'm', 'g', 'y', 'orange'
    lc_props = dict(color='g', marker='o', ms=1.5, mec='None')
    spec_props = dict(color='g')
    cb_props = {}  # dict(format=ticker.FuncFormatter(logformat))
    defaults = Spectral.defaults

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, t, signal, **kws):
        ''' '''
        cmap = kws.pop('cmap', 'viridis')

        show_lc = kws.pop('show_lc', True)  # or ('lc_props' in kws)
        lc_props = TimeFrequencyMapBase.lc_props.copy()
        lc_props.update(kws.pop('lc_props', {}))

        show_spec = kws.pop('show_spec', True)  # or ('spec_props' in kws)
        spec_props = TimeFrequencyMapBase.spec_props.copy()
        spec_props.update(kws.pop('spec_props', {}))
        self.spec_quant = kws.pop('spec_quant', (0.25, 0.50, 0.75))

        show_info = kws.pop('show_info', True)

        # Compute spectral estimates
        Spectral.__init__(self, t, signal, **kws)

        self.figure, axes = self.setup_figure(show_lc, show_spec, show_info)
        self.ax_map, self.ax_lc, self.ax_spec, self.ax_cb = axes
        self.plot(axes, t, signal, cmap, lc_props, spec_props)

        self._need_save = False

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def setup_figure(self, show_lc=True, show_spec=True, show_info=True):
        '''Setup figure geometry'''

        # TODO limit axes to lower 0 Hz??  OR hatch everything below this somehow
        # NOTE: You will need a custom transformation to implement this.
        # Something that does transAxes for the lower limit, but transData for the upper
        # MAYBE ask on SO??

        # TODO Axes container
        figsize = (16, 8)
        fig = plt.figure(figsize=figsize, )
        gskw = dict(hspace=0.02,
                    wspace=0.005,
                    top=0.96,
                    left=0.05,
                    right=0.94 if show_spec else 0.94,
                    bottom=0.07)
        rows_cols = (100, 100)
        gs = gridspec.GridSpec(*rows_cols, **gskw)

        # if show_spec:
        # w1, w2 = 80, 20
        # else:
        # w1, _ = 98, 2

        # optionally display various axes
        if show_spec & show_lc:
            w1, w2 = 80, 20
            h1, h2, h3 = 30, 69, 1
            ax_map = fig.add_subplot(gs[h1:-h3, :w1])
            ax_cb = fig.add_subplot(gs[-h3:, w1:])
            ax_lc = fig.add_subplot(gs[:h1, :w1], sharex=ax_map)
            ax_spec = fig.add_subplot(gs[h1:-h3, w1:], sharey=ax_map)

            if show_info:
                self.ax_info = ax_info = fig.add_subplot(gs[:h1, w1:])
                # ax_info.set_visible(False)
                ax_info.patch.set_visible(False)
                self.ax_info.tick_params('both', left='off', labelleft='off',
                                         labelbottom='off', bottom='off')
                for _, spine in self.ax_info.spines.items():
                    spine.set_visible(False)


        elif show_spec and not show_lc:
            w1, _ = 80, 20
            h1, h2, h3 = 0, 99, 1
            ax_map = fig.add_subplot(gs[h1:-h3, :w1])
            ax_cb = fig.add_subplot(gs[-h3:, w1:])
            ax_lc = fig.add_subplot(gs[:h1, :w1],
                                    sharex=ax_map) if show_lc    else None
            ax_spec = fig.add_subplot(gs[h1:h2, w1:],
                                      sharey=ax_map) if show_spec  else None
            fig.subplots_adjust(right=0.95)  # space for cbar ticks

        elif show_lc and not show_spec:
            w1, _ = 98, 2
            h1, h2, h3 = 20, 80, 0
            ax_map = fig.add_subplot(gs[h1:-h3, :w1])
            ax_cb = fig.add_subplot(gs[h1:, w1:])
            ax_lc = fig.add_subplot(gs[:h1, :w1],
                                    sharex=ax_map) if show_lc    else None
            ax_spec = fig.add_subplot(gs[h1:h2, w1:],
                                      sharey=ax_map) if show_spec  else None
        else:
            w1, _ = 98, 2
            h1, h2, h3 = 0, 100, 0
            ax_map = fig.add_subplot(gs[h1:-h3, :w1])
            ax_cb = fig.add_subplot(gs[h1:, w1:])
            ax_lc = fig.add_subplot(gs[:h1, :w1],
                                    sharex=ax_map) if show_lc    else None
            ax_spec = fig.add_subplot(gs[h1:h2, w1:],
                                      sharey=ax_map) if show_spec  else None

            # add the subplots
            # ax_map  = fig.add_subplot(gs[h1:-h3, :w1])
            # ax_cb   = fig.add_subplot(gs[h2:, w1:])
            # ax_lc   = fig.add_subplot(gs[:h1, :w1],
            # sharex=ax_map)    if show_lc    else None
            # ax_spec = fig.add_subplot(gs[h1:h2, w1:],
            # sharey=ax_map)    if show_spec  else None




        # options for displaying various parts
        minorTickSize = 8
        ax_map.tick_params(which='both', labeltop=False, direction='out')
        ax_map.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax_map.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        # MajForm = ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(x))
        # MajForm = formatter_factory( MajForm, tolerance=0.1 )
        # ax_map.yaxis.set_major_formatter(MajForm)
        # MinForm = ticker.FuncFormatter( lambda x, pos: '{:.1f}'.format(x).strip('0') )
        # ax_map.yaxis.set_minor_formatter( MajForm )            #ticker.ScalarFormatter()
        ax_map.yaxis.set_tick_params('minor', labelsize=minorTickSize)

        # Coordinate display format
        ax_map.format_coord = self.mapCoordDisplayFormatter

        # setup_ticks
        if ax_lc:
            # set major/minor xticks invisible on light curve plot
            ax_lc.tick_params(axis='x', which='both', labelbottom=False,
                              labeltop=True, top='on', direction='inout', pad=0)
            # ax_lc.tick_params(axis='y', which='both', top='on')

            ax_lc.set_ylabel('Flux (counts/s)')
            ax_lc.grid()

        # Get label for power values
        if self.opts.normalise == 'rms':
            cb_lbl = r'Power density (rms$^2$/Hz)'
        else:
            cb_lbl = 'Power'  # FIXME: unit??

        if ax_spec:
            # set yticks invisible on frequency spectum plot
            ax_spec.xaxis.set_tick_params(which='both', labelbottom=False)
            ax_spec.yaxis.set_tick_params(which='both', labelleft=False,
                                          direction='inout', right='on')
            ax_spec.xaxis.offsetText.set_visible(False)

            # show Period as ticks on right spine of y
            self._parasite = axp = ax_spec.twinx()

            axp.yaxis.set_major_formatter(ReciprocalFormatter())
            axp.yaxis.set_tick_params(left='off', labelleft='off')
            ax_spec.yaxis.set_tick_params(left='off', labelleft='off')
            #TODO: same tick positions

            ax_spec.yaxis.set_label_position('right')
            ax_spec.set_ylabel('Period (s)', labelpad=40)


            # ax_spec.set_xlabel(cb_lbl, labelpad=25)
            ax_spec.grid()
            ax_spec.format_coord = format_coord_spec
            ax_cb.set_xlabel(cb_lbl)

        else:
            ax_cb.yaxis.set_label_position('right')
            ax_cb.set_ylabel(cb_lbl)

        ax_map.set_xlabel('t (s)')
        ax_map.set_ylabel('Frequency (Hz)')

        return fig, (ax_map, ax_lc, ax_spec, ax_cb)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def plot(self, axes, t, signal, cmap, lc_props={}, spec_props={}):
        ''' '''
        ax_map, ax_lc, ax_spec, ax_cb = axes
        tms, frq, P = self.tms, self.frq, self.power
        #P /= P.mean(1)[:, None]
        valid = frq > self.fRayleigh
        # NOTE: we intentionally do not mask values below fRayleigh, even though
        # they are not physicaly meaningful because this often leads to the
        # appearance of a false "peak" at low frequencies.  Instead, we  hatch
        # everything below the Rayleigh frequency.

        # Plot TFR image
        tlims = self.t_seg[[0, -1], [0, -1]]
        flims = self.frq[[0, -1]]
        extent = np.r_[tlims, flims]
        self.im = im = mimage.NonUniformImage(ax_map,
                                              origin='lower',
                                              extent=extent,
                                              cmap=cmap)
        # data = im.cmap(P.T)
        # im.set_figure( fig_map )

        # detect_gaps
        # narrow_gap_inds = Spectral.detect_gaps(t, self.KCT*128/3600, 2)
        # narrow_gap_inds = narrow_gap_inds
        ##print( narrow_gap_inds )

        ##fade gaps
        # alpha = data[...,-1]
        # alpha[:,narrow_gap_inds] = 0.25
        # alpha[:,narrow_gap_inds+1] = 0.25

        # print( t.shape, frq.shape, data.shape )
        im.set_data(tms, frq, P.T)
        ax_map.images.append(im)
        # ax_map.set_xlim( t[0],t[-1] )
        # ax_map.set_ylim( frq[0],frq[-1] )

        # hatch anything below self.fRayleigh
        polycol = ax_map.fill_between(tlims, self.fRayleigh, facecolor='none',
                                      edgecolors='r', linewidths=1, hatch='\\')

        if ax_lc:
            # Plot light curve
            ax_lc.plot(t, signal, **lc_props)  # TODO: Uncertainties
            ax_lc.set_xlim(t[0], t[-1])
            ax_lc.set_ylim(*axes_limit_from_data(signal, 0.1))

        # guess reasonable colour limits
        climp = (0.25, 99.9)  # colour limits as percentile of power value
        Plim = np.percentile(P[:, valid], climp)
        # set map colour limits
        im.set_clim(Plim)

        if ax_spec:
            # Plot spectrum (median & inter-quartile range)
            quartiles = np.array(self.spec_quant) * 100 #(25, 50, 75)
            Pm_lci, Pm, Pm_uci = np.percentile(P, quartiles, 0)
            self.pwr_p25, self.pwr_p50, self.pwr_p75 = Pm_lci, Pm, Pm_uci

            sm = 5
            ax_spec.plot(smoother(Pm, sm), frq, **spec_props)

            ax_spec.plot(smoother(Pm_lci, sm), frq, ':', smoother(Pm_uci, sm), frq, ':', **spec_props)
            #HACK
            # ax_spec.plot(smoother(self.pwr_p75, 5), frq, '-', **spec_props)

            ax_spec.set_xlim(Plim)
            ax_spec.set_ylim(frq[0], frq[-1])
            self._parasite.set_ylim(frq[0], frq[-1])        # FIXME

            # ax_spec.set_xscale('log')

            # hatch anything below self.fRayleigh
            rinv = Rectangle((0, 0), 1, self.fRayleigh,
                             facecolor='none', edgecolor='r',
                             linewidth=1, hatch='\\',
                             transform=btf(ax_spec.transAxes, ax_spec.transData))
            ax_spec.add_patch(rinv)

            # show colourbar
            tmp = ax_cb.get_xlabel()
            self.colour_bar = self.figure.colorbar(im,
                                                   ticks=ax_spec.get_xticks(),
                                                   cax=ax_cb,
                                                   orientation='horizontal',
                                                   **self.cb_props)
            ax_cb.set_xlabel(tmp)  # redo the labels (which the previous line would have killed)

            # connect callbacks for limit change
            ax_spec.callbacks.connect('xlim_changed', self._set_clim)
            ax_spec.callbacks.connect('ylim_changed', self._set_parasite_ylim)

        else:
            #TODO: 'setup alt cbar'
            tmp = ax_cb.get_ylabel()
            self.colour_bar = self.figure.colorbar(im, cax=ax_cb, **self.cb_props) # ticks=ax_spec.get_xticks(),
            ax_cb.set_ylabel(tmp)          # set the labels (which the previous line killed)


        # TODO: MOVE TO SUBCLASS ?
        ax_map.callbacks.connect('xlim_changed', self.save_background)
        ax_map.callbacks.connect('ylim_changed', self.save_background)
        ax_map.callbacks.connect('ylim_changed', self._set_parasite_ylim)

        self.info_text()

    def save_background(self, _=None):
        # save_background
        print('SAVING BG')
        self.background = self.canvas.copy_from_bbox(self.figure.bbox)

    def _set_parasite_ylim(self, ax):      # FIXME: obviate by using TimeFreqDualAxes
        print('_set_parasite_ylim', ax.get_ylim())
        self._parasite.set_ylim(ax.get_ylim())
        print('YO')
        print(self._parasite.get_ylim())
        print('!!')

        self._need_save = True
        print('Done')


    def _set_clim(self, ax):
        print('_set_clim', ax.get_ylim())
        xlim = ax.get_xlim()
        self.im.set_clim(xlim)
        self.colour_bar.set_ticks(self.ax_spec.get_xticks())
        self._need_save = True
        # self.save_background()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def info_text(self):
        # TODO: include etc...

        info = ('$\Delta t = %.3f$ s ($f_s = %.3f$ Hz)' % (self.dt, self.fs),
                'window = %s' % (self.opts.window, ),
                '$n_w = %d$ (%.1f s)' % (self.opts.nwindow, self.opts.nwindow * self.dt),
                '$n_{ovr} = %d$ (%.0f%%)' % (self.noverlap, self.noverlap / self.nwindow * 100),
                )
        if self.opts.pad:
            info += ('pad = %s' % str(self.opts.pad),)
        if self.opts.detrend:
            info += ('detrend = %s' % str(self.opts.detrend),)

        txt = '\n'.join(info)
        self.infoText = self.ax_info.text(0.05, 1, txt, va='top',
                                          transform=self.ax_info.transAxes)


    def mapCoordDisplayFormatter(self, x, y):

        frac = np.divide((x, y), (self.tms[-1], self.frq[-1]))
        col, row = np.round(frac * self.power.shape, 0).astype(int)

        Nrows, Ncols = self.power.shape
        if col >= 0 and col < Ncols and row >= 0 and row < Nrows:
            z = self.power[row, col]
            return 't=%1.3f,\tf=%1.3f,\tPwr=%1.3g' % (x, y, z)
        else:
            return 'x=%1.3f, y=%1.3f' % (x, y)


# ****************************************************************************************************
class TimeFrequencyMap(TimeFrequencyMapBase, ConnectionMixin):
    '''
    Time Frequency Representation (aka Power Spectral density map)
    Interactive plot elements live in this class
    '''

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, t, signal, **kws):
        ''' '''
        self.smoothing = kws.pop('smoothing', 0)  # smoothing for displayed segment spectrum
        TimeFrequencyMapBase.__init__(self, t, signal, **kws)
        self.scaling = 1. / self.pwr_p50.sum()   # scale segment spectrum to sum of median (for display)

        # initialize auto-connect
        ConnectionMixin.__init__(self, self.figure)

        # save background for blitting
        #self.canvas.draw()
        #self.save_background()

        # TODO: can you subclass widgets.cursor to emulate the desired behaviour??
        self.hovering = False
        self.icolour = iter(self.color_cycle)
        self.spans = []  # container for highlighted segments
        self.spectra = []

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def plot(self, axes, t, signal, cmap, lc_props={}, spec_props={}):

        TimeFrequencyMapBase.plot(self, axes, t, signal, cmap, lc_props, spec_props)

        # Initiate elements for interactive display
        # TODO: only really need these upon connect
        self.ispec, = self.ax_spec.plot([], [], 'r', alpha=0.65)  # instantaneous spectum

        # TODO:  Alpha channel for window shape
        self.span_transform = btf(self.ax_lc.transData, self.ax_lc.transAxes)
        self.span = Rectangle((0, 0), 0, 1,
                              color='r', alpha=0.35,
                              transform=self.span_transform)
        self.ax_lc.add_patch(self.span)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_spectrum(self, ix, smoothing=0, scaling=1.):  # misnomer?
        ''' '''
        data = self.power[ix]
        if smoothing:
            data = smoother(data, smoothing)
        if scaling:
            data = data * scaling

        return np.array([data, self.frq])

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def update(self, ix):

        # update spectrum
        spectrum = self.get_spectrum(ix, self.smoothing, )
        self.ispec.set_data(spectrum)

        # update rectangle for highlighted span
        tspan = self.t_seg[ix, (0, -1)]             # NOTE padded values not included here #TODO: maybe some visual indicator for padding??
        width = tspan.ptp()
        self.span.set_x(tspan[0])
        self.span.set_width(width)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def highlight_section(self):
        # persistent spectrum for this window
        colour = next(self.icolour)
        spectrum = self.ispec.get_xydata().T  # instantaneous spectum
        line, = self.ax_spec.plot(*spectrum, color=colour, alpha=0.65, lw=1)
        self.spectra.append(line)

        # persistent highlight this window
        span = Rectangle(self.span.xy, self.span.get_width(), 1,
                         color=colour, alpha=0.35,
                         transform=self.span_transform)
        self.ax_lc.add_patch(span)
        self.spans.append(span)

        self.span.set_visible(False)
        self.ispec.set_visible(False)
        self.draw_blit()

        renderer = self.figure._cachedRenderer
        line.draw(renderer)
        span.draw(renderer)
        self.save_background()
        self.draw_blit()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def draw_blit(self):
        renderer = self.figure._cachedRenderer
        self.canvas.restore_region(self.background)
        self.ispec.draw(renderer)
        self.span.draw(renderer)

        self.canvas.blit(self.figure.bbox)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # @mpl_connect( 'draw_event' )
        # def _on_draw(self, event):
        # print( 'drawing:', event )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def ignore_hover(self, event):
        return not (event.inaxes == self.ax_map and
                    self.canvas.manager.toolbar._active is None)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @mpl_connect('motion_notify_event')
    def _on_motion(self, event):
        if event.inaxes != self.ax_map:
            return

        ix = abs(self.tms - event.xdata).argmin()
        self.update(ix)
        self.draw_blit()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

            self.span.set_visible(True)
            self.ispec.set_visible(True)
            self.draw_blit()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @mpl_connect('axes_leave_event')
    def _leave_axes(self, event):
        # print('leave')
        if event.inaxes == self.ax_map:
            self.span.set_visible(False)
            self.ispec.set_visible(False)
            self.draw_blit()

            self.hovering = False

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @mpl_connect('button_press_event')
    def _on_button(self, event):
        # print(event.button)

        if event.button == 1:
            if event.inaxes == self.ax_map:
                self.highlight_section()

        if event.button == 2:  # restart on middle mouse
            self.restart()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @mpl_connect('key_press_event')
    def _on_key(self, event):
        print(event.key)


    @mpl_connect('draw_event')
    def _on_draw(self, event):
        print('DRAWING', event)
        if self._need_save:
            self.save_background()      # DOESN'T work since it gets executed before the draw is complete
            self._need_save = False


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def restart(self):
        self.span.set_visible(False)
        self.ispec.set_visible(False)

        # reset colour cycle
        self.icolour = iter(self.color_cycle)
        for span, spec in zip(self.spans, self.spectra):
            span.remove()
            spec.remove()

        self.spans = []
        self.spectra = []

        self.canvas.draw()
        self.background = self.canvas.copy_from_bbox(self.figure.bbox)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # def connect(self):
        # self.canvas.mpl_connect( 'motion_notify_event', self._on_motion )
        # self.canvas.mpl_connect( 'axes_enter_event', self._enter_axes )
        # self.canvas.mpl_connect( 'axes_leave_event', self._leave_axes )


TimeFrequencyRepresentation = TimeFrequencyMap


# ****************************************************************************************************
class SpectralCoherenceMap(TimeFrequencyMapBase):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, t, signalA, signalB, **kws):
        show_lc = kws.pop('show_lc', False)  # or ('lc_props' in kws)
        lc_props = TimeFrequencyMapBase.lc_props.copy()
        lc_props.update(kws.pop('lc_props', {}))
        show_spec = kws.pop('show_spec', True)  # or ('spec_props' in kws)
        spec_props = TimeFrequencyMapBase.spec_props.copy()
        spec_props.update(kws.pop('spec_props', {}))

        cmap = kws.pop('cmap', 'viridis')

        self.t = t
        self.signalA = signalA
        self.signalB = signalB

        # TODO: base method that does this for TFR & CSD
        self.opts = SpectralCoherenceMap.defaults
        self.check(t, signalA, **kws)

        # timing stats
        dt = self.check_timing(t, self.opts.dt)
        self.dt = dt
        self.fs = 1. / dt

        # clean masked, fill gaps etc
        t, signalA = self.prepare_signal(t, signalA, self.dt)
        t, signalB = self.prepare_signal(t, signalB, self.dt)

        self.nwindow = resolve_nwindow(self.opts.nwindow, self.opts.split, t, self.dt)
        self.noverlap = resolve_overlap(self.nwindow, self.opts.noverlap)
        self.fRayleigh = 1. / (self.nwindow * dt)

        # fold
        self.t_seg, self.segAraw = self.get_segments(t, signalA, dt,
                                                     self.nwindow, self.noverlap)
        _, self.segBraw = self.get_segments(t, signalB, dt,
                                            self.nwindow, self.noverlap)

        # median time for each section
        self.tms = np.median(self.t_seg, 1)

        # FFT frequencies
        nw = self.opts.apodise or self.nwindow
        self.frq = np.fft.rfftfreq(nw, self.dt)
        self.ohm = 2. * np.pi * self.frq  # angular frequencies

        # pad, detrend, window
        self.segmentsA = self.prepare_segments(self.segAraw)
        self.segmentsB = self.prepare_segments(self.segBraw)

        # calculate spectra
        self.power = self.main(self.segmentsA, self.segmentsB)

        # plot stuff
        fig, axes = self.setup_figure(show_lc, show_spec)
        self.ax_map, self.ax_lc, self.ax_spec, self.ax_cb = axes
        self.plot(axes, t, signalA, cmap, lc_props, spec_props)  # FIXME: signalB

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def main(self, segA, segB):
        specA = scipy.fftpack.fft(segA)
        specA = specA[..., :len(self.frq)]  # since we are dealing with real signals
        specB = scipy.fftpack.fft(segB)
        specB = specB[..., :len(self.frq)]

        csd = np.conjugate(specA) * specB
        coh = np.abs(csd) ** 2 / (np.abs(specA) * np.abs(specB))

        T = self.t.ptp()
        power = T ** 2 * coh / (self.signalA.sum() * self.signalB.sum())

        return power
