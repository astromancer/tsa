

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
from tsa.smoothing import tv

logger.enable('recipes')

fn = '/home/hannes/work/phd/thesis/tex/chapters/ch4/data/shoc/phot/lightcurves/by_file/20130616.0030/20130616.0030.oflag.txt'
bjd, flux, err = io.read(fn)
ts = TimeSeries(bjd, flux.T, err.T)


@pytest.mark.mpl_image_compare(baseline_dir='images/smoothing',
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
                 ls='-', ms=0, c=c, zorder=10)

    fig.set_size_inches(11, 5)
    ax.legend(loc='upper right')

    return fig


# NOTE: must give filename if cange format, but then no parametrization!
#    filename='example.pdf',
#    savefig_kwargs={'dpi': 450, 'format': 'pdf'},
#    style='default'
@pytest.mark.mpl_image_compare(baseline_dir='images/smoothing/windowed')
@pytest.mark.parametrize(
    'smoothing', [1e-4, 1e-2]
)
def test_smoothing_windowed_overlap(smoothing):

    n = 2000
    nwindow = 100
    tss = ts[:n].normalize(loc=False, scale=False)
    # tss.t = np.linspace(0, 1, len(tss))  # so we can see indices
    fig, ax = tss.plot(labels=['Data', ''], alpha=0.25)

    # Full solution reference
    tsv = tss.smooth.tv(smoothing)
    config = dict(zorder=10, lw=2)
    tsv.plot(ax=ax, labels=['no windowing', ''], color='k', **config)

    noverlap = [0, 0.125, 0.25, 0.5]
    colors = plt.colormaps['tab20'](range(4))
    for no, c in zip(noverlap, colors):

        tsv = tss.smooth.tv(smoothing, nwindow, no)
        tsv.plot(ax=ax, labels=[fR'$n_o = {no}$', ''], color=c, **config)

    fig.set_size_inches(11, 5)
    fig.subplots_adjust(right=0.75)
    ax.legend(loc='upper left',
              bbox_to_anchor=(1.02, 1.02),
              title=fR'$\lambda = {smoothing}, n = {n}, n_w = {nwindow}$')
    # ax.grid()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir='images/smoothing/optimal',
                               # filename='example0.png',
                               style='default')
@pytest.mark.parametrize(
    'loc, scale, tscale',
    itt.product(*[[0, 1]] * 2, (1, 86400, 'ptp'))
)
def test_smoothing_optimal_scaling(loc, scale, tscale):

    # norm
    tss = ts[2200:2400].normalize(loc, scale, tscale=tscale)
    fig, ax = tss.plot(errorbar={'ls': ''})

    # smooth
    tsv = tss.smooth.tv()
    tsv.plot(ax=ax, ls='-', ms=0)

    # compare (passing result back to smoother to produce model)
    # should overlap exactly if we did the parameter rescaling correctly internally
    for i, o in enumerate(tss.smooth.optima):
        tss[i].smooth.tv(o).plot(ax=ax, ms=0, lw=5, alpha=0.5)

    # ax.grid(False)
    return fig

    # z, opt = tv.smooth(np.array(tss.x.T[0]))
    # fig, ax = plt.subplots()
    # ax.plot(tss.x.T[0], '.', ms=2)
    # ax.plot(z, '-')

    # tss = ts.smooth.tv(nwindow=1000, noverlap='25%', strength=0.1)

    # # plot
    # ts.plot()
    # tss.plot(errorbar={'ls':'-'})

    # plt.show()


@pytest.mark.mpl_image_compare(baseline_dir='images/smoothing/windowed')
# @pytest.mark.parametrize(
#     'smoothing', [1e-2]
# )
def test_smoothing_optimal_long():

    section = np.s_[:3000]

    # normalize time
    tss = ts[section].normalize(loc=False, scale=False)
    fig, ax = tss.plot(labels=['Data', ''], alpha=0.25)

    # Full solution reference
    tsv = tss.smooth.tv()
    config = dict(zorder=100, lw=5, alpha=0.5)

    for i, o in enumerate(tss.smooth.optima):
        tsv[i].plot(ax=ax, label=fR'$\lambda = {o:.3f}$', **config, )

    fig.set_size_inches(11, 5)
    fig.subplots_adjust(right=0.75)
    ax.legend(loc='upper left',
              bbox_to_anchor=(1.02, 1.02),
              title=fR'Optimal TVR: $n = {len(tss)}$')
    return fig
