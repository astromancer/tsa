import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mimage
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory as btf
from matplotlib import gridspec
from matplotlib import ticker

#from misc.meta import
#from superplot.spectra import formatter_factory

from .spectral import Spectral
from .tsa import smoother

from grafico.lc import get_axlim
from grafico.misc import ConnectionMixin, mpl_connect

from IPython import embed

#FIXME: repeat code!
def format_coord(x, y):
    
    
    #x = ax.format_xdata(x)
    #y = ax.format_ydata(y)
    
    p = 1./y
    #f = self.time_axis.get_major_formatter().format_data_short(x)
    
        
    return 'f = {:.6f}; p = {:.3f};\ty = {:.3f}'.format( x, p, y )
    #return 'f = {}; p = {};\ty = {}'.format( x, p, y )


#****************************************************************************************************
class TimeFrequencyRepresentation(Spectral, ConnectionMixin):
    '''Power Spectral density map'''
    
    color_cycle = 'c', 'b', 'm', 'g', 'y', 'orange'
    smoothing = 5
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, t, signal, **kw):
        Spectral.__init__(self, t, signal, **kw)
        self.t = t
        self.signal = signal
        
        fig, axes = self.setup_figure()
        ConnectionMixin.__init__(self, fig)
        self.plot(axes)
        
        #TODO: can you subclass widgets.cursor to emulate the desired behaviour??
        self.hovering = False
        self.icolour = iter(self.color_cycle)
        self.spans = []                 #container for highlighted segments
        self.spectra = []
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def setup_figure(self):
        '''Setup figure geometry'''
        
        #TODO Axes container
        
        figsize=(18,8)
        self.figure     = fig           = plt.figure( figsize=figsize, 
                                                      tight_layout=1 )
        
        gs = gridspec.GridSpec( 3, 2, 
                                height_ratios=(30,100,1), 
                                width_ratios=(4,1),
                                hspace=0.02, 
                                wspace=0.005,
                                top=0.96,
                                left=0.05,
                                right=0.98,
                                bottom=0.05 )
        self.ax_map     = ax_map        = fig.add_subplot( gs[1,0] )
        self.ax_lc      = ax_lc         = fig.add_subplot( gs[0], sharex=ax_map )
        self.ax_spec    = ax_spec       = fig.add_subplot( gs[1,1], sharey=ax_map )
        self.ax_cb      = ax_cb         = fig.add_subplot( gs[2,1] )
        
        #Coordinate display format
        #self.ax_map.format_coord = format_coord
        self.ax_spec.format_coord = format_coord
        
        #Ticks and labels
        #setup_ticks(ax_map)
        minorTickSize = 8
        ax_map.tick_params(which='both', labeltop=False, direction='out')
        ax_map.xaxis.set_minor_locator( ticker.AutoMinorLocator() )
        #ax_map.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
        MajForm = ticker.FuncFormatter(lambda x, pos: '{:.1f}'.format(x))
        #MajForm = formatter_factory( MajForm, tolerance=0.1 )
        ax_map.yaxis.set_major_formatter( MajForm )
        #MinForm = ticker.FuncFormatter( lambda x, pos: '{:.1f}'.format(x).strip('0') )
        #ax_map.yaxis.set_minor_formatter( MajForm )            #ticker.ScalarFormatter()
        ax_map.yaxis.set_tick_params( 'minor', labelsize=minorTickSize )
        

        ax_lc.xaxis.set_tick_params(which='both', labelbottom=False )   #set major/minor xticks invisible on light curve plot
        ax_spec.xaxis.set_tick_params(which='both', labelbottom=False )   #set yticks invisible on frequency spectum plot
        ax_spec.yaxis.set_tick_params(which='both', labelleft=False )
        
        
        ax_lc.set_ylabel( 'Flux (counts/s)' )
        ax_spec.set_xlabel( 'Power', labelpad=25 )
        ax_map.set_xlabel( 't (s)' )
        ax_map.set_ylabel( 'Frequency (Hz)' )

        ax_lc.grid()
        ax_spec.grid()
        
        return fig, (ax_map, ax_lc, ax_spec, ax_cb)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def plot(self, axes):
        
        t, signal = self.t, self.signal
        tms, frq, P = self.tms, self.frq, self.power
        
        ax_map, ax_lc, ax_spec, ax_cb = axes

        #Plot light curve
        ax_lc.plot(t, signal, 'go', ms=1.5, mec='None')
        ax_lc.set_xlim( t[0], t[-1] )
        ax_lc.set_ylim( *get_axlim(signal, 0.1) )

        #Plot spectrum
        Pm = P.mean(0)  #Mean power spectrum
        plim = (0.25, 99.5)
        Plim = np.percentile( Pm, plim )
        ax_spec.plot( Pm, frq, 'g')
        
        #
        self.hover, = ax_spec.plot( [], [], 'r', alpha=0.65)     #instantaneous spectum
        #TODO:  Alpha channel for window shape
        self.span_transform = btf(ax_lc.transData, ax_lc.transAxes)
        self.span   = Rectangle( (0,0), 0, 1,
                                 color='r',
                                 alpha=0.35,
                                 transform=self.span_transform )
        ax_lc.add_patch( self.span )
        
        
        
        ax_spec.set_xlim( Plim )
        ax_spec.set_ylim( frq[0], frq[-1] )
       
        #ax_spec.set_xscale('log')

        #Plot PSD map
        extent = (tms[0],tms[-1],frq[0],frq[-1])
        self.im  = im = mimage.NonUniformImage( self.ax_map, 
                                                origin='lower',
                                                extent=extent,
                                                cmap='viridis')
        #clim = np.percentile( P, (0, 100) )
        im.set_clim( Plim )
        data = im.cmap( P.T )

        #im.set_figure( fig_map )

        #detect_gaps
        #narrow_gap_inds = Spectral.detect_gaps(t, self.KCT*128/3600, 2)
        #narrow_gap_inds = narrow_gap_inds
        ##print( narrow_gap_inds )

        ##fade gaps
        #alpha = data[...,-1]
        #alpha[:,narrow_gap_inds] = 0.25
        #alpha[:,narrow_gap_inds+1] = 0.25

        #print( t.shape, frq.shape, data.shape )
        im.set_data( tms, frq, P.T )
        ax_map.images.append( im )
        #ax_map.set_xlim( t[0],t[-1] )
        #ax_map.set_ylim( frq[0],frq[-1] )
        
        self.colour_bar = self.figure.colorbar( im,
                                                ticks=ax_spec.get_xticks(),
                                                cax=ax_cb,
                                                orientation='horizontal' )

        #plt.setp(self.ax_spec.get_xmajorticklabels(), visible=False)
        
        self.canvas = self.figure.canvas
        self.canvas.draw()
        self.background = self.canvas.copy_from_bbox( self.figure.bbox )
        
        def set_clim(ax):
            xlim = ax.get_xlim()
            im.set_clim(xlim)
            self.colour_bar.set_ticks(ax_spec.get_xticks())
            
    
        ax_spec.callbacks.connect('xlim_changed', set_clim)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_spectrum(self, ix, smoothing=0): #misnomer?
        ''' '''
        data = self.power[ix]
        if smoothing:
            data = smoother(data, smoothing)
        
        return np.array([data, self.frq])
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def update(self, ix):
        
        #update spectrum
        spectrum = self.get_spectrum(ix, self.smoothing)
        self.hover.set_data(spectrum)
        
        #update rectangle for highlighted span
        tspan = self.t_seg[ix, (0,-1)]
        width = tspan.ptp()
        self.span.set_x( tspan[0] )
        self.span.set_width(width)
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def draw(self):
        self.canvas.restore_region( self.background )
        self.hover.draw(self.figure._cachedRenderer)
        self.span.draw(self.figure._cachedRenderer)
        
        self.canvas.blit( self.figure.bbox )
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #@mpl_connect( 'draw_event' )
    #def _on_draw(self, event):
        #print( 'drawing:', event )
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def ignore_hover(self, event):
        return not (event.inaxes == self.ax_map and
                    self.canvas.manager.toolbar._active is None)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @mpl_connect( 'motion_notify_event' )
    def _on_motion(self, event):
        if event.inaxes != self.ax_map:
            return
        
        ix = abs(self.tms - event.xdata).argmin()
        self.update(ix)
        self.draw()
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @mpl_connect( 'axes_enter_event' )
    def _enter_axes(self, event):
        #print('enter')
        #self.canvas.draw()
        
        if not self.ignore_hover(event):
            #NOTE:  if the cursor "leaves" the axes directly onto another window, 
            #the axis leave event is not triggered!!
            if not self.hovering: #i.e legitimate axes enter
                #NOTE: Need to save the background here in case a zoom happened
                self.background = self.canvas.copy_from_bbox(self.figure.bbox)
            
            self.hovering = True
            
            self.span.set_visible(True)
            self.hover.set_visible(True)
            self.draw()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @mpl_connect( 'axes_leave_event' )
    def _leave_axes(self, event):
        #print('leave')
        if event.inaxes == self.ax_map:
            self.span.set_visible(False)
            self.hover.set_visible(False)
            self.draw()
            
            self.hovering = False
            
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @mpl_connect( 'button_press_event' )
    def _on_button(self, event):
        #print(event.button)
        
        if event.button == 1:   #maintain this highlighted section
            if event.inaxes == self.ax_map:
                #print('maintain')
                colour = next(self.icolour)
                
                spectrum = self.hover.get_xydata().T
                line, = self.ax_spec.plot(*spectrum, color=colour, alpha=0.65, lw=1)     #instantaneous spectum
                self.spectra.append(line)
                
                
                span = Rectangle(self.span.xy, self.span.get_width(), 1,
                              color=colour,
                              alpha=0.35,
                              transform=self.span_transform )
                self.ax_lc.add_patch(span)
                self.spans.append(span)
                
                self.span.set_visible(False)
                self.hover.set_visible(False)
                self.draw()
                
                line.draw(self.figure._cachedRenderer)
                span.draw(self.figure._cachedRenderer)
                self.background = self.canvas.copy_from_bbox(self.figure.bbox)
                self.draw()
                
                
        if event.button == 2:   #restart
            self.restart()
            
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @mpl_connect( 'key_press_event' )
    def _on_key(self, event):
        print(event.key)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def restart(self):
        self.span.set_visible(False)
        self.hover.set_visible(False)
        
        
        #reset colour cycle
        self.icolour = iter(self.color_cycle)
        for span, spec in zip(self.spans, self.spectra):
            span.remove()
            spec.remove()
            
        self.spans = []
        self.spectra = []
        
        self.canvas.draw()
        self.background = self.canvas.copy_from_bbox(self.figure.bbox)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #def connect(self):
        #self.canvas.mpl_connect( 'motion_notify_event', self._on_motion )
        #self.canvas.mpl_connect( 'axes_enter_event', self._enter_axes )
        #self.canvas.mpl_connect( 'axes_leave_event', self._leave_axes )
        