import numpy as np
import pytest
from tsa.ts import TimeSeries, MultiVariateTimeSeries


# generate some dat
np.random.seed(666)

n = 50
t = np.linspace(0, 2 * np.pi, n)
y = np.sin(3 * t)
e = np.random.randn(n)
# np.cos(10*t),
#  np.cos(10 * np.sqrt(t))]

n2 = 100
t2 = np.linspace(0, np.pi, n2)
y2 = np.random.randn(3, n2)

m = np.random.rand(n) > 0.8
ym = np.ma.array(y, mask=m)


# @pytest.mark.mpl_image_compare(baseline_dir = 'images',
#                                 remove_text = True)


class TestTimeSeries:
    @pytest.mark.parametrize(
        'args',
        [  # basic: implicit time index
            (y,),
            # explicit time index
            (t, y),
            # with uncertainties
            (t, y, e),
            # masked data
            (t, ym, e),  # show_masked='x',
        ]
    )
    def test_init(self, args):
        # intrinsically indexed TS
        ts = TimeSeries(*args)
    
    # def test_multivariate(self):
    #     ts = TimeSeries(t2, y2)
    #     assert isinstance(ts, MultiVariateTimeSeries)