"""
Generalized Cross Validation for Total Variation regularization.

Source: 
    Data smoothing and numerical differentiation by a regularization method.
    Stickel (2011)
See also:
    Appendix A; Lubansky+ (2006)
    
Functions in this module are named after the variables in the Stickel (2011)
paper.
"""

# std
import numbers

# third-party
import numpy as np
from loguru import logger
from scipy.optimize import minimize
from scipy.sparse.linalg import inv
from scipy.sparse import csc_matrix, eye, lil_matrix

# local
from recipes.array import fold
from recipes.concurrent import Executor


# ---------------------------------------------------------------------------- #
# Even the spare matrix implimentation has limits
MAX_ARRAY_SIZE = 5000

# ---------------------------------------------------------------------------- #


# def _sanitize(x, y):
#     # remove masked points
#     ok = ~(np.ma.getmaskarray(x) | np.ma.getmaskarray(y))
#     return np.ma.getdata(x[ok]), np.ma.getdata(y[ok])


class WindowSmoother(Executor):

    __slots__ = ('nwindow', 'noverlap')

    def __init__(self, nwindow, noverlap='25%',
                 jobname=None, backend='multiprocessing', **kws):

        self.nwindow = nwindow
        self.noverlap = noverlap
        super().__init__(jobname, backend, **kws)

    # def __repr__(self):
    #     return f'{type(self).__name__}()'

    def __call__(self, t, x, smoothing=None, njobs=-1):

        x = np.asanyarray(x).squeeze()
        n = len(x)
        assert x.ndim == 1
        if t is None:
            t = np.arange(n)
        else:
            assert len(t) == n

        nwindow = fold.resolve_size(self.nwindow, n)
        noverlap = fold.resolve_size(self.noverlap, nwindow)
        if noverlap > nwindow // 2:
            raise ValueError(
                'Window overlap should be less than half window size for TVR.'
            )
        if noverlap == 0:
            logger.warning('Overlap is recommended to avoid edge effects.')

        # Fold arrays
        tf = fold.fold(t, nwindow, noverlap)
        data = fold.fold(x, nwindow, noverlap)
        nsegs = tf.shape[0]

        if smoothing:
            smoothing *= (n / nwindow) ** 3

        masked = np.ma.is_masked(x) | np.ma.is_masked(t)
        self.init_memory((nsegs, nwindow), masked)
        self.run(zip(tf, data), λs=smoothing, njobs=njobs)

        results = np.ma.MaskedArray(self.results, self.mask)

        if noverlap:
            # concatenate
            start, odd = divmod(noverlap, 2)
            end = -(start + odd)
            return np.ma.hstack([
                results[0, :end],
                results[1:, start:end].ravel(),
            ])[:n]
        else:
            return self.results.reshape(-1)

    def _compute(self, data, **kws):
        return smooth(*data, **kws)


def _check_arrays(x, y, size_limit=MAX_ARRAY_SIZE):
    y = np.asanyarray(y).squeeze()
    if y.ndim != 1:
        raise ValueError(f'Input data should be 1D not {y.ndim}D.')

    if (n := y.size) > size_limit:
        raise ValueError(
            f'Array too large: {n} > {size_limit}. This is here to prevent '
            'memory overflow during the optimization. For large arrays you may '
            'wish to usethe `tv.WindowSmoother` class to smooth the time '
            'sereis segment by segment.'
        )

    if x is None:
        x = np.arange(n)
    else:
        x = np.asanyarray(x).squeeze()
        if x.ndim != 1:
            raise ValueError(f'Input data should be 1D not {y.ndim}D.')

        if x.size != n:
            raise ValueError('Input vectors should be the same size: '
                             f'({len(x) = }) != ({len(y) = })')

    return x, y


def smooth(x, y=None, λs=None, d=2):
    """
    Total Variation Regularization (TVR) smoothing

    Parameters
    ----------
    x: time_or_signal
    y: signal
    λs: float or None
        If None (the default) search for optimal value of smoothing
        parameter by minimizing cross-validation variance
        If float, this value will be used as the smoothing value
    d: int, float
        Exponent in distance metric, 2 by default.

    Returns
    -------

    """
    if (y is None) or isinstance(y, numbers.Real):
        # single vector input mode
        x, y, λs = None, x, y

    # sanitize
    x, y = _check_arrays(x, y)

    if λs is None:
        # optimal λ search
        # yhat, λopt = smooth_optimal(x, y, d)
        return smooth_optimal(x, y, d=d)

    if isinstance(λs, numbers.Real):
        if np.ma.is_masked(x) | np.ma.is_masked(y):
            # handle masked data
            out = np.ma.empty(y.shape)
            ok = ~(np.ma.getmaskarray(x) | np.ma.getmaskarray(y))
            out[ok] = _smooth(x[ok], y[ok], λs, d)
            out[~ok] = np.ma.masked
            return out

        return _smooth(x, y, λs, d)

    raise ValueError(f'Invalid smoothing parameter: {λs = }. Should be a real'
                     ' number, or `None` for optimal smoothing.')


# alias
smoother = smooth


def _smooth(x, y, λs, d=2):
    # todo: mapping matrix (interpolation);  weights
    """
    Base smoother employing total variational regularization

    Parameters
    ----------
    x
    y
    λs
    d

    Returns
    -------

    """

    n = len(y)
    D_ = D(x, d)
    # scale smoothing parameter
    λ = λs / ((D_.T @ D_).trace() / n ** (d + 2))

    # integral estimate
    i0, i1 = int(np.ceil(d / 2)), int(np.floor(d / 2))
    U = B(x)[i0:-i1, i0:-i1]

    # solve
    return inv(csc_matrix(eye(n) + λ * (D_.T @ U @ D_))) @ y


def smooth_optimal(x, y, d=2):
    """
    Optimal Total Variational smoothing. Optimal smoothing value is found by
    minimizing cross-validation variance.


    Parameters
    ----------
    x
    y
    λ0=1
        seems to be a good initial choice
    d

    Returns
    -------

    """
    
    x, y = _check_arrays(x, y)
    
    if not (np.ma.is_masked(x) | np.ma.is_masked(y)):
        return _smooth_optimal(x, y, d)

    # handle masked
    ok = ~(np.ma.getmaskarray(x) | np.ma.getmaskarray(y))
    yhat = np.ma.empty(y.shape)
    yhat[ok], λopt = _smooth_optimal(x[ok], y.data[ok], d)
    yhat[~ok] = np.ma.masked
    return yhat, λopt


def _smooth_optimal(x, y, d=2):
    # Solve for optimal smoothing parameter λs. That is minimize
    # cross-validation variance

    # rescale x (for numerical stability)
    xscale = (x[-1] - x[0])
    x = (x - x[0]) / xscale

    n = len(y)
    I = eye(n)
    D_ = D(x, d)
    R = D_.T @ D_

    # scale factor for smoothing parameter
    δ = R.trace() / n ** (d + 2) 
    
    # integral estimate
    i0, i1 = int(np.ceil(d / 2)), int(np.floor(d / 2))
    U = B(x)[i0:-i1, i0:-i1]

    # assert λ0 > 0
    λs0 = 1 / δ
    logger.opt(lazy=True).info(
        '{}', lambda: f'Minimize starting at {λs0 = :.3g} ({δ = :.3g}) {xscale = :.3g}'
    )
    #
    result = minimize(_objective, λs0, (y, I, D_, R, U), 'Nelder-Mead',
                      bounds=[(0, None)])

    if result.success:
        λopt = result.x.item()
        logger.success('Converged: λ = {}', λopt)
        yhat = inv(csc_matrix(I + λopt * (D_.T @ U @ D_))) @ y
        
        # rescale result back to input coordinate scale
        λ = λopt * δ / xscale

        return yhat, λ

    raise ValueError(f'Optimization unsuccessful: {result.message!r}.')


def _objective(λ, y, I, D_, R, U):
    # objective function for minimization. returns the cross validation
    # variance associated with the smoothness λ
    n = len(y)
    λ = λ.item()  # minimize turns this into an array
    yhat = inv(csc_matrix(I + λ * R)) @ y

    H_ = inv(csc_matrix(I + λ * (D_.T @ U @ D_)))

    #       rss
    # returns the cross validation variance associated with the smoothness λ
    return (np.square(yhat - y).sum() / n) / (1 - H_.trace() / n) ** 2


# def D(x, d=1):
#     # order d finite difference derivative estimator for unequally spaced data:
#     # f' = D f
#     # Defines differential operator via recurrence relation.
#     # TODO: various methods.

#     n = len(x)
#     # first order derivative estimator (matrix operator)
#     dx = np.roll(x, -d)[:-d] - x[:-d]
#     Vd = np.eye(n - d) / dx  # FIXME: MemoryError for large arrays
#     Dhat1 = np.eye(n - d, n - d + 1, 1) - np.eye(n - d, n - d + 1)

#     dr = d * Vd @ Dhat1
#     return dr if d == 1 else dr @ D(x, d - 1)


def D(x, d=1):
    # order d finite difference derivative estimator:
    # f' = D f
    # Defines differential operator via recurrence relation.

    n = len(x)
    # first order derivative estimator (matrix operator)
    δx = np.roll(x, -d)[:-d] - x[:-d]
    δx[δx == 0] = 1e-10                 # numerical stability
    Vd = lil_matrix((n - d, n - d))
    Vd.setdiag(1 / δx)
    Dhat1 = eye(n - d, n - d + 1, 1) - eye(n - d, n - d + 1)
    dr = d * Vd @ Dhat1
    return dr if d == 1 else dr @ D(x, d - 1)


def B(x):
    # midpoint rule integration matrix
    # TODO: various other methods of integration?
    n = len(x)
    B_ = np.empty(n)
    B_[0] = np.diff(x[:2])
    B_[1:-1] = np.roll(x, -2)[:-2] - x[:-2]
    B_[-1] = np.diff(x[-2:])

    B = lil_matrix((n, n))
    B.setdiag(B_)
    return B


def H(x, λs, d=2):
    # M = W = I for now

    # trim down integral estimator matrix
    i0, i1 = int(np.ceil(d / 2)), int(np.floor(d / 2))
    U = B(x)[i0:-i1, i0:-i1]

    # derivative estimator
    D_ = D(x, d)

    # rescale smoothness parameter
    δ = np.trace(D_.T @ D_) / len(x) ** (d + 2)
    return inv(np.eye(len(x)) + (λs / δ) * (D_.T @ U @ D_))


def Vgcv(x, y, yhat, λs):
    # cross validation variance
    n = len(y)
    rss = np.square(yhat - y).sum()
    return (rss / n) / (1 - np.trace(H(x, y, λs)) / n) ** 2
