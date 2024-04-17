# std
import numbers
import contextlib
import functools as ftl

# third-party
import numpy as np
import scipy as sp
import scipy.signal

# local
from recipes.array import fold
from recipes.string import Percentage
from recipes.concurrency import Executor


# ---------------------------------------------------------------------------- #

def get_window(window, n=None):
    """
    Return window values of window described by str `window' and length `n'.
    ...
    """
    if isinstance(window, (str, tuple)):
        if n is None:
            raise ValueError('Please specify window size `n`.')

        if window == 'hanning':
            window = 'hann'

        return sp.signal.get_window(window, n)

    # if window values are passed explicitly as a sequence of values
    if np.iterable(window):
        # if N given, assert that it matches the window length
        if n and len(window) != n:
            raise ValueError(
                f'Length {len(window)} of given window does not match '
                f'array length {n}.'
            )
        return window

    raise ValueError(f'Cannot make window from object: {window!r}.')


def windowed(a, window=None):
    """get window values for array, and multiply."""
    return a if (window is None) else a * get_window(window, a.shape[-1])


# ---------------------------------------------------------------------------- #
#
def resolve_size(size, n=None, dt=None):

    # overlap specified by percentage string eg: 99% or timescale eg: 60s
    if isinstance(size, str):
        assert n, 'Array size `n` required if `size` given as percentage (str).'

        # percentage
        if size.endswith('%'):
            return round(Percentage(size).of(n))

        return _size_from_unit_string(size, dt)

    if isinstance(size, float):
        if size < 1:
            assert n, 'Array size `n` required if `size` given as percentage (float).'
            return round(size * n)

        raise ValueError('Providing a float value for `size` is only valid if '
                         'that value is smaller than 1, in which case it is '
                         'interpreted as a fraction of the array size.')

    if isinstance(size, numbers.Integral):
        return size

    raise ValueError(
        f'Invalid size: {size!r}. This should be an integer, or a percentage '
        'of the array size as a string eg: "12.4%", or equivalently a float < 1'
        ' eg: 0.124, in which case the array size should be supplied. '
        'Finally, you may also provide the size in units of seconds eg: '
        '"30s", in whic case, the timestep `dt`, should also be provided.'
    )


def _size_from_unit_string(size, dt):
    if size.endswith('s'):
        return round(float(size.strip('s')) / dt)

    raise NotImplementedError


# ---------------------------------------------------------------------------- #
# plot

def show_all_windows(cmap='gist_rainbow', size=1024):
    """
    plot all the spectral windows defined in scipy.signal (at least those that
    don't want a parameter argument.)
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    cm = plt.get_cmap(cmap)
    windows = scipy.signal.windows.__all__
    ax.set_color_cycle(cm(np.linspace(0, 1, len(windows))))

    winge = ftl.partial(scipy.signal.get_window, Nx=size)
    for w in windows:
        with contextlib.suppress(Exception):
            plt.plot(winge(w), label=w)

    plt.legend()
    plt.show()


# ---------------------------------------------------------------------------- #

class MovingWindowAnalysis(Executor):
    """Base class for sliding windows"""

    __slots__ = ('n', 'nwindow', 'noverlap', 'n_repeats', 'weight_kernel', 'weights')

    def __init__(self, nwindow, noverlap='25%', weight_kernel=None,
                 jobname=None, backend='multiprocessing', xfail=10, **kws):

        # init Executor
        super().__init__(jobname, backend, xfail, **kws)

        self.n = self.n_repeats = None  # set in call
        self.nwindow = nwindow
        self.noverlap = noverlap
        self.weight_kernel = weight_kernel
        self.weights = None

    def __repr__(self, ignore=('n_repeats', 'jobname', 'backend',  'nfail'), **kws):
        return super().__repr__(ignore=ignore, **kws)

    def __call__(self, t, x, njobs=-1, **kws):

        x = np.asanyarray(x).squeeze()
        # assert x.ndim == 1, '1D input required'  # Should be dropped eventually

        self.n = n = len(x)
        assert n > 1, f'Too few data points: {n}'

        if t is None:
            t = np.arange(n)
        else:
            assert len(t) == n

        # get window / overlap size
        self.nwindow = nwindow = fold.resolve_size(self.nwindow, n)
        self.noverlap = noverlap = fold.resolve_size(self.noverlap, nwindow)
        self.n_repeats = fold.get_n_repeats(n, nwindow, noverlap)
        self.check()  # NOTE: changes nwindow!
        # nwindow = self.nwindow
        # noverlap = self.noverlap

        if self.weight_kernel:
            self.weights = get_window(self.weight_kernel, nwindow)

        # Fold arrays
        tf = fold.fold(t, nwindow, noverlap)
        data = fold.fold(x, nwindow, noverlap)
        data = np.moveaxis(np.atleast_3d(data), 2, 1)

        # Compute
        masked = np.ma.is_masked(x) | np.ma.is_masked(t)
        self.init_memory(data.shape, masked)
        return self.run(zip(tf, data), njobs=njobs, **kws)

    def check(self):
        if self.n < self.nwindow:
            self.logger.warning(
                'Data length {0.n} is smaller than window size {0.nwindow}! '
                'Setting the window size to data size.', self
            )
            # self.nwindow = self.n
            # self.noverlap = 0

    def _compute(self, data, **kws):
        raise NotImplementedError()

    def finalize(self, **kws):
        # collect results
        results = np.ma.MaskedArray(self.results, self.mask)

        if self.n <= self.nwindow:
            # start, end = 0, None
            return results[0].T

        if self.noverlap:
            # concatenate
            start, odd = divmod(self.noverlap, 2)
            end = -(start + odd)
            return np.ma.hstack([
                results[0, ..., :end],
                *results[1:, ..., start:end]
            ]).T[:self.n]

        return results.reshape(-1)
