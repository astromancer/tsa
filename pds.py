import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mimage
from matplotlib import gridspec
from matplotlib import ticker

from tsa.spectral import Spectral
from superplot.lc import get_axlim

from IPython import embed

class PowerDensitySpectrum( object ):
    '''Power Spectral density map'''

    def __init__(self, t, flux, **kw):
        '''
        Compute and plot the PSD map.
        Kw directly passed to Spectral.compute
        '''
        spec = Spectral( t, flux, **kw )
    
        self.t, self.frq, self.power = tmap, frq, P = spec.t, spec.frq, spec.power
    
        ax_map, ax_lc, ax_spec, ax_cb = self.setup_figure( )

        #Plot light curve
        ax_lc.plot(t, flux, 'go', ms=1.5, mec='None')
        ax_lc.set_xlim( t[0], t[-1] )
        ax_lc.set_ylim( *get_axlim(flux, 0.1) )

        #Plot spectrum
        Pm = P.mean(0)  #Mean power spectrum
        plim = (0.25, 99.5)
        Plim = np.percentile( Pm, plim )
        ax_spec.plot( Pm, frq, 'g')
        
        ax_spec.set_xlim( Plim )
        ax_spec.set_ylim( frq[0], frq[-1] )
       
        #ax_spec.set_xscale('log')

        #Plot PSD map
        extent = (tmap[0],tmap[-1],frq[0],frq[-1])
        self.im  = im = mimage.NonUniformImage( self.ax_map, 
                                                origin='lower',
                                                extent=extent )
        clim = np.percentile( P, (0, 99.95) )
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
        im.set_data( tmap, frq, P.T )
        ax_map.images.append( im )
        #ax_map.set_xlim( t[0],t[-1] )
        #ax_map.set_ylim( frq[0],frq[-1] )
        
        self.colour_bar = self.fig.colorbar( im,
                                                    ticks=self.ax_spec.get_xticks(),
                                                    #ax=(ax_map, ax_spec),
                                                    cax=ax_cb,
                                                    orientation='horizontal', )
                                                    #fraction=0.01, 
                                                    #pad=0.1, 
                                                    #anchor=(1,0), 
                                                    #aspect=50, 
                                                    #extend='max')
        plt.setp( self.ax_spec.get_xmajorticklabels(), visible=False )
        



    def setup_figure(self):
        '''Setup figure geometry'''
        #left, bottom, width, height = 0.055, 0.025, 0.65, 0.65
        #spacing = 0.01
        #bottom_h = left_h = left + width + spacing
        #self.width, self.spacing = width, spacing

        #rect_map = [left, bottom, width, height]
        #rect_lc = [left, bottom_h, width, 0.2]
        #rect_spec = [left_h, bottom, 0.2, height]

        #self.fig = fig = plt.figure( figsize=(18,8) )
        #self.ax_map = ax_map = fig.add_axes(rect_map)
        #self.ax_lc = ax_lc = fig.add_axes(rect_lc, sharex=ax_map)
        #self.ax_spec = ax_spec = fig.add_axes(rect_spec, sharey=ax_map)
        
        #fig.subplots_adjust( bottom=0.01 )
        
        #self.fig, self.ax_map = fig, ax_map = plt.subplots(figsize=(18,8))

        # create new axes on the right and on the top of the current axes.
        #divider = make_axes_locatable(ax_map)
        #ax_lc = divider.append_axes("top", size='30%', pad=0.1, sharex=ax_map)
        #ax_spec = divider.append_axes("right", size='25%', pad=0.1, sharey=ax_map)
        
        
        self.fig = fig = plt.figure( figsize=(18,8), tight_layout=1 )
        gs = gridspec.GridSpec(3, 2, 
                                height_ratios=(30,100,1), width_ratios=(4,1),
                                hspace=0.02, wspace=0.005)

        self.ax_map = ax_map = fig.add_subplot( gs[1,0] )
        self.ax_lc = ax_lc = fig.add_subplot( gs[0], sharex=ax_map )
        self.ax_spec = ax_spec = fig.add_subplot( gs[1,1], sharey=ax_map )
        self.ax_cb = ax_cb = fig.add_subplot( gs[2,1] )
        
        #Ticks and labels
        #setup_ticks(ax_map)
        minorTickSize = 8
        ax_map.tick_params(which='both', labeltop=False, direction='out')
        ax_map.xaxis.set_minor_locator( ticker.AutoMinorLocator() )
        ax_map.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
        MajForm = ticker.FuncFormatter(lambda x, pos: '{:.1f}'.format(x))
        ax_map.yaxis.set_major_formatter( MajForm )
        #MinForm = ticker.FuncFormatter( lambda x, pos: '{:.1f}'.format(x).strip('0') )
        ax_map.yaxis.set_minor_formatter( MajForm )            #ticker.ScalarFormatter()
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

        return ax_map, ax_lc, ax_spec, ax_cb