# Generalized Cross Validation for TV regularization
# See: Stickel (2011), Appendix A; Lubansky+ (2006)
# functions below are named after the variables in the paper

import numbers

# third-party libs
import numpy as np
from scipy.optimize import minimize


def _sanitize(x, y):
    # remove masked points
    good = ~(np.ma.getmaskarray(x) | np.ma.getmaskarray(y))
    return np.ma.getdata(x[good]), np.ma.getdata(y[good])


def smooth(x, y=None, λs=None, d=2):
    """
    Total Variation Regularization based smoothing

    Parameters
    ----------
    x:
    y:
    λs: float or None
        If None (the default) search for optimal value of smoothing
        parameter by minimizing cross-validation variance
        If float, this value will be used as the smoothing value
    d: int, float
        Exponent in distance metric. Default is 2

    Returns
    -------

    """
    if (y is None) or isinstance(y, numbers.Real):
        # single vector input mode
        λs = y
        y = x
        x = np.arange(len(y))
    else:
        x = np.asanyarray(x)
        y = np.asanyarray(y)

    #
    if λs is None:
        # optimal λ search
        yhat, λopt = smooth_optimal(x, y, d)
        return yhat
    elif isinstance(λs, numbers.Real):
        if np.ma.is_masked(x) | np.ma.is_masked(y):
            good = ~(np.ma.getmaskarray(x) | np.ma.getmaskarray(y))
            out = np.ma.empty(y.shape)
            out[good] = _smooth(x[good], y[good], λs, d)
            out[~good] = np.ma.masked
            return out

        return _smooth(x, y, λs, d)

    raise ValueError('Invalid smoothing parameter %s' % λs)


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
    δ = np.trace(D_.T @ D_) / n ** (d + 2)
    λ = λs / δ

    # integral estimate
    i0, i1 = int(np.ceil(d / 2)), int(np.floor(d / 2))
    U = B(x)[i0:-i1, i0:-i1]

    # solve
    # I = np.eye(n)
    return np.linalg.inv((np.eye(n) + λ * (D_.T @ U @ D_))) @ y


def smooth_optimal(x, y, d=2):
    """
    Optimal Total Variational smoothing. Optimal smoothing value is found by
    minimizing cross-validation variance.


    Parameters
    ----------
    x
    y
    d

    Returns
    -------

    """

    if np.ma.is_masked(x) | np.ma.is_masked(y):
        good = ~(np.ma.getmaskarray(x) | np.ma.getmaskarray(y))
        yhat = np.ma.empty(y.shape)
        yhat[good], λopt = _smooth_optimal(x[good], y.data[good], d)
        yhat[~good] = np.ma.masked
        return yhat, λopt
    else:
        return _smooth_optimal(x, y.data, d)


def _smooth_optimal(x, y, d=2):
    # Solve for optimal smoothing parameter λs. That is minimize
    # cross-validation variance
    n = len(y)
    I = np.eye(n)
    D_ = D(x, d)
    R = D_.T @ D_
    # scale factor for smoothing parameter
    δ = np.trace(R) / n ** (d + 2)

    # integral estimate
    i0, i1 = int(np.ceil(d / 2)), int(np.floor(d / 2))
    U = B(x)[i0:-i1, i0:-i1]

    λs0 = 0.001  # seems to be a good initial choice
    result = minimize(_objective, λs0 / δ, (y, I, D_, R, U), 'Nelder-Mead')
    if result.success:
        λopt = result.x
        yhat = np.linalg.inv((I + λopt * (D_.T @ U @ D_))) @ y
        return yhat, (λopt * δ)

    raise ValueError('Optimization unsuccessful: %r' % result.message)


def _objective(λ, y, I, D_, R, U):
    # objective function for minimization.  returns the cross validation
    # variance associated with the smoothness λ
    n = len(y)
    yhat = np.linalg.inv((I + λ * R)) @ y

    H_ = np.linalg.inv(I + λ * (D_.T @ U @ D_))

    rss = np.square(yhat - y).sum()
    return (rss / n) / (1 - np.trace(H_) / n) ** 2


def D(x, d=1):
    # order d finite difference derivative estimator for unequally spaced data:
    # f' = D f
    # Defines differential operator via recurrence relation.
    # TODO: various methods.

    n = len(x)
    # first order derivative estimator (matrix operator)
    dx = np.roll(x, -d)[:-d] - x[:-d]
    Vd = np.eye(n - d) / dx  # FIXME: MemoryError for large arrays
    Dhat1 = np.eye(n - d, n - d + 1, 1) - np.eye(n - d, n - d + 1)

    if d == 1:
        return d * Vd @ Dhat1
    return d * Vd @ Dhat1 @ D(x, d - 1)


def B(x):
    # midpoint rule integration matrix
    # TODO: various other methods of integration?
    B = np.empty(len(x))
    B[0] = np.diff(x[:2])
    B[1:-1] = np.roll(x, -2)[:-2] - x[:-2]
    B[-1] = np.diff(x[-2:])
    return np.diag(B)


def H(x, λs, d=2):
    # M = W = I for now

    # trim down integral estimator matrix
    i0, i1 = int(np.ceil(d / 2)), int(np.floor(d / 2))
    U = B(x)[i0:-i1, i0:-i1]

    # derivative estimator
    D_ = D(x, d)

    # rescale smoothness parameter
    δ = np.trace(D_.T @ D_) / len(x) ** (d + 2)
    I = np.eye(len(x))
    return np.linalg.inv(I + (λs / δ) * (D_.T @ U @ D_))


def Vgcv(x, y, yhat, λs):
    # cross validation variance
    n = len(y)
    rss = np.square(yhat - y).sum()
    return (rss / n) / (1 - np.trace(H(x, y, λs)) / n) ** 2
