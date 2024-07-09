
# third-party
import numpy as np

# local
from recipes.array import fold
from recipes.concurrency import Executor
from recipes.oo.represent import Represent

# relative
from . import resolve


# ---------------------------------------------------------------------------- #

class MovingWindowAnalysis(Executor):
    """Base class for sliding/rolling/moving window analysis tasks."""

    __slots__ = ('n', 'nwindow', 'noverlap', 'n_repeats', 'weight_kernel', 'weights')
    __repr__ = Represent(ignore=('n_repeats', 'jobname', 'backend', 'nfail'))

    # @api.synoonyms({'(window|taper)': 'weight_kernel'})

    def __init__(self, nwindow, noverlap='25%', weight_kernel=None,
                 jobname=None, backend='multiprocessing', xfail=10, **kws):

        # init Executor
        super().__init__(jobname, backend, xfail, **kws)

        self.n = self.n_repeats = None  # set in call
        self.nwindow = nwindow
        self.noverlap = noverlap
        self.weight_kernel = weight_kernel
        self.weights = None

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
            self.weights = resolve.array(self.weight_kernel, nwindow)

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
