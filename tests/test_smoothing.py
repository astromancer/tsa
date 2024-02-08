

# std
import itertools as itt

# third-party
import pytest
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

# local
from obstools.lc import io
from tsa.ts import TimeSeries


logger.enable('recipes')

fn = '/home/hannes/work/phd/thesis/tex/chapters/ch4/data/shoc/phot/lightcurves/by_file/20130616.0030/20130616.0030.oflag.txt'
bjd, flux, err = io.read(fn)
ts = TimeSeries(bjd, flux.T, err.T)


@pytest.mark.mpl_image_compare(baseline_dir='images',
                               # filename='example0.png',
                               style='default')
def test_smoothing():

    tss = ts[2000:2500]
    fig, ax = tss.plot(errorbar={'ls': ''})

    n = 10
    colors = plt.colormaps['tab20'](range(n))
    for smoothing, c in zip(np.logspace(-3, 6, n), colors):
        tsv = tss.smooth.tv(smoothing)
        tsv.plot(ax=ax,
                 labels=[fR'$\lambda = 10^{{{int(np.log10(smoothing))}}}$', ''],
                 errorbar=dict(ls='-', ms=0, c=c))

    fig.set_size_inches(11, 5)
    ax.legend(loc='upper right')

    return fig


@pytest.mark.mpl_image_compare(baseline_dir='images',
                               # filename='example0.png',
                               style='default')
@pytest.mark.parametrize(
    'smoothing', [1e-4, 1e-2]
)
def test_smoothing_windowed(smoothing):

    n = 300
    nwindow = 100
    tss = ts[:n].normalize(loc=False, scale=False)
    # tss.t = np.linspace(0, 1, len(tss))  # so we can see indices
    fig, ax = tss.plot(errorbar={'ls': ''}, labels=['Data', ''])

    # Full solution reference
    tsv = tss.smooth.tv(smoothing)
    tsv.plot(ax=ax,
             labels=[fR'$ \lambda = {smoothing}$ (no windowing)', ''],
             ls='-', ms=0, color='k')

    noverlap = [ 0, 0.125, 0.25, 0.5 ] #
    colors = plt.colormaps['tab20'](range(4))
    for no, c in zip(noverlap, colors):

        tsv = tss.smooth.tv(smoothing, nwindow, no)
        #
        tsv.plot(ax=ax,
                 labels=[fR'$ \lambda = {smoothing}, n_o = {no}$', ''],
                 ls='-', ms=0, color=c)

    fig.set_size_inches(11, 5)
    fig.subplots_adjust(right=0.77)
    ax.legend(loc='upper left',
              bbox_to_anchor=(1.02, 1.02),
              title=f'$n = {n}, n_w = {nwindow}$')
    # ax.grid()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir='images',
                               # filename='example0.png',
                               style='default')
@pytest.mark.parametrize(
    'loc, scale, tscale',
    itt.product(*[[0, 1]] * 2, (1, ts.t.ptp()))
)
def test_optimal_smoothing(loc, scale, tscale):

    tss = ts[:100].normalize(loc, scale, tscale=tscale)
    tsv = tss.smooth.tv()
    fig, ax = tss.plot(errorbar={'ls': ''})

    tsv.plot(ax=ax, errorbar={'ls': '-', 'ms': 0})

    # z, opt = tv.smooth(np.array(tss.x.T[0]))
    # fig, ax = plt.subplots()
    # ax.plot(tss.x.T[0], '.', ms=2)
    # ax.plot(z, '-')

    # tss = ts.smooth.tv(nwindow=1000, noverlap='25%', strength=0.1)

    # # plot
    # ts.plot()
    # tss.plot(errorbar={'ls':'-'})

    plt.show()
