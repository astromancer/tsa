
# third-party libs
import numpy as np

# local libs
from tsa.ts import TimeSeries
from recipes.testing import Expect, mock, ANY, Throws


# pylint: disable=missing-function-docstring

# generate some dat
np.random.seed(666)

n = 50
t = np.linspace(0, 2 * np.pi, n)
y = np.sin(3 * t)
e = np.random.rand(n)
m = np.random.rand(n) > 0.8
ym = np.ma.array(y, mask=m)
# np.cos(10*t),
#  np.cos(10 * np.sqrt(t))]

n2 = 100
t2 = np.linspace(0, np.pi, n2)
y2 = np.random.randn(3, n2)


# 

# class TestTimeSeries:

test_init = Expect(TimeSeries)({
     # basic, implicit time index
        mock.TimeSeries(y):                             ANY,
        # multivariate, implicit time index
        mock.TimeSeries(y2):                            ANY,
        # explicit time index
        mock.TimeSeries(t, y):                          ANY,
        # with uncertainties
        mock.TimeSeries(t, y, e):                       ANY,
        # masked data
        mock.TimeSeries(t, ym, e):                      ANY,
        # negative uncertainties not allowed
        mock.TimeSeries(t, y, -np.ones_like(y)):        Throws(ValueError)
})


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
