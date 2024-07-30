
# std
import tempfile as tmp
from pathlib import Path

# third-party
import pytest
import numpy as np

# local
from recipes.config import ConfigNode
from tsa import io
from tsa.ts import TimeSeries


# pylint: disable=missing-function-docstring


# ---------------------------------------------------------------------------- #
#
CASE_IDS = ['basic', 'timed', 'uncertain', 'masked']
TEST_FOLDER = Path(__file__).parent
DATA_FOLDER = TEST_FOLDER / 'data'
DATA_PARAMS = ConfigNode({
    'univariate':   {
        'size':        50,
        'period':      3,
        'amplitude':   2
    },
    'multivariate': {
        'size':        (100, 3),
        'period':      [1, 2, 3]
    }
})


# ---------------------------------------------------------------------------- #

def sinusoidal(size, interval=(0, 2 * np.pi), period=1, amplitude=1, phase=0,
               noise=1, mask=0.8):

    # generate some data
    size = np.atleast_1d(size)
    t = np.linspace(*interval, size[0])
    y = amplitude * np.sin(np.atleast_2d(period).T * t + phase)
    e = noise * np.random.rand(*size)

    m = np.random.randn(*size) > mask
    ym = np.ma.array(y.T, mask=m).squeeze()

    return t, ym, e


# ---------------------------------------------------------------------------- #
# data fixtures

def get_combination(case_data, id):

    t, ym, e = case_data
    y = ym.data

    if id == 'basic':
        # basic, implicit time index
        return y,

    if id == 'timed':
        # explicit time index
        return t, y

    if id == 'uncertain':
        # with uncertainties
        return t, y, e

    if id == 'masked':
        # masked data
        return t, ym, e

    raise ValueError


@pytest.fixture(ids=DATA_PARAMS.keys(), params=DATA_PARAMS.values())
def sample_data(request):
    return sinusoidal(**request.param)


@pytest.fixture(params=CASE_IDS, ids=CASE_IDS)
def case_data(sample_data, request):
    return get_combination(sample_data, request.param)


# ---------------------------------------------------------------------------- #
# Tests

def test_init(case_data):
    ts = TimeSeries(*case_data)
    if ts.m > 1:
        assert isinstance(ts, ts.multivariate)


def test_unequal_size():
    # unequal array sizes
    with pytest.raises(ValueError):
        TimeSeries([0, 1], [1, 1], [1])


def test_negative_uncertainty():
    # negative uncertainties not allowed
    with pytest.raises(ValueError):
        TimeSeries([0, 1], [1, 1], [1, -1])


@pytest.mark.parametrize('ext', io.SUPPORTED)
def test_io(case_data, ext):

    # init
    ts = TimeSeries(*case_data)

    # write
    # suf = 'i' * (ts.index is not None) + 'σ' * (ts.sigma is not None) + 'm' * np.ma.is_masked(ts.x)
    # name = DATA_FOLDER / f'{ts.m}-{ts.n}-{suf}.{ext}'
    fp, name = tmp.mkstemp(f'.{ext}', dir=DATA_FOLDER)
    ts.save(name)

    # test read
    clone = TimeSeries.read(name)

    # compare
    assert np.ma.allclose(ts.value, clone.value, atol=get_tol('values', ext))

    if ts.index is None:
        assert np.all(clone.index == np.arange(len(clone)))
    else:
        assert np.ma.allclose(ts.index, clone.index, atol=get_tol('index', ext))

    if ts.sigma is None:
        assert clone.sigma is None
    else:
        assert np.ma.allclose(ts.sigma, clone.sigma, atol=get_tol('sigma', ext))


def get_tol(field, ext):
    return 10 ** -get_precision(field, ext)


def get_precision(field, ext):
    cfg = io.txt.CONFIG
    if ext == 'txt':
        return cfg.columns[field].get('precision') or cfg.precision
    return cfg.precision


@pytest.mark.mpl_image_compare(baseline_dir='images/ts/',
                               style='default')
def test_plot(case_data):
    ts = TimeSeries(*case_data)
    tsp = ts.plot()  # mask = np.atleast_2d(mask)
    return tsp.figure

# ---------------------------------------------------------------------------- #

# test_init = Expected(TimeSeries)({
#     # basic, implicit time index
#     mock.TimeSeries(y):                             PASS,
#     # multivariate, implicit time index
#     mock.TimeSeries(y2):                            PASS,
#     # explicit time index
#     mock.TimeSeries(t, y):                          PASS,
#     # with uncertainties
#     mock.TimeSeries(t, y, e):                       PASS,
#     # masked data
#     mock.TimeSeries(t, ym, e):                      PASS,
#     # negative uncertainties not allowed
#     mock.TimeSeries(t, y, -np.ones_like(y)):        Throws(ValueError)
# })


# @pytest.mark.mpl_image_compare(baseline_dir = 'images',
# #                                 remove_text = True)
# def test_plot():

# def test_multivariate(self):
#     ts = TimeSeries(t2, y2)
#     assert isinstance(ts, MultiVariateTimeSeries)


# @pytest.mark.parametrize(
#         'args',
#         [  # basic
#             (y[0],),
#             # multiple series by index
#             (y,),
#             # multiple series, single time vector
#             (t, y),
#             # multiple series with uncertainties, single time vector
#             (t, y, e),
#             # masked data
#             (t, ym, e),  # show_masked='x',
#             #  multiple series non-uniform sizes
#             ([t, t2], [ym[0], y2], [e[1], None])
#         ]
# )
# def test_plot(args, **kws):
#     tsp = ts.plot(*args, **kws)


# kws = {}
# tsp = ts.plot(y[0], **kws)
# tsp = ts.plot(y, **kws)
# tsp = ts.plot(t, y, **kws)
# tsp = ts.plot(t, y, e, **kws)
# tsp = ts.plot(t, ym, e, show_masked='x', **kws)
# tsp = ts.plot([t, np.arange(n2)],
#               [ym[0], np.random.rand(n2)],
#               [e[1], None],
#               **kws)


# plt.show()
# raise err

# TODO: more tests
#  everything with histogram # NOTE: significantly slower
#  test raises
