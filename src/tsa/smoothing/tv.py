"""
Generalized Cross Validation for Total Variation regularization.

Source: 
    Data smoothing and numerical differentiation by a regularization method.
    Stickel (2011)
See also:
    Appendix A; Lubansky+ (2006)
    
Functions in this module are named after the variables in the Stickel (2011) paper.
"""

# std
import numbers

# third-party
import numpy as np
from scipy import sparse
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
        λs, y = y, x
        x = np.arange(len(y))
    else:
        x, y = map(np.asanyarray, (x, y))

    #
    if λs is None:
        # optimal λ search
        # yhat, λopt = smooth_optimal(x, y, d)
        return smooth_optimal(x, y, d)

    if isinstance(λs, numbers.Real):
        if np.ma.is_masked(x) | np.ma.is_masked(y):
            good = ~(np.ma.getmaskarray(x) | np.ma.getmaskarray(y))
            out = np.ma.empty(y.shape)
            out[good] = _smooth(x[good], y[good], λs, d)
            out[~good] = np.ma.masked
            return out

        return _smooth(x, y, λs, d)

    raise ValueError(f'Invalid smoothing parameter: {λs}. Should be a real'
                     ' number, or None for optimal smoothing.')


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
    return sparse.linalg.inv(sparse.csc_matrix(sparse.eye(n) + λ * (D_.T @ U @ D_))) @ y


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

    if not np.ma.is_masked(x) | np.ma.is_masked(y):
        return _smooth_optimal(x, y.data, d)

    good = ~(np.ma.getmaskarray(x) | np.ma.getmaskarray(y))
    yhat = np.ma.empty(y.shape)
    yhat[good], λopt = _smooth_optimal(x[good], y.data[good], d)
    yhat[~good] = np.ma.masked
    return yhat, λopt


def _smooth_optimal(x, y, d=2, ):
    # Solve for optimal smoothing parameter λs. That is minimize
    # cross-validation variance
    n = len(y)
    I = sparse.eye(n)
    D_ = D(x, d)
    R = D_.T @ D_
    # scale factor for smoothing parameter
    δ = R.trace() / n ** (d + 2)

    # integral estimate
    i0, i1 = int(np.ceil(d / 2)), int(np.floor(d / 2))
    U = B(x)[i0:-i1, i0:-i1]

    λs0 = 0.001  # seems to be a good initial choice
    result = minimize(_objective, λs0 / δ, (y, I, D_, R, U), 'Nelder-Mead')

    if result.success:
        λopt = result.x.item()
        yhat = sparse.linalg.inv(sparse.csc_matrix(I + λopt * (D_.T @ U @ D_))) @ y
        return yhat, (λopt * δ)

    raise ValueError(f'Optimization unsuccessful: {result.message!r}.')


def _objective(λ, y, I, D_, R, U):
    # objective function for minimization. returns the cross validation
    # variance associated with the smoothness λ
    n = len(y)
    λ = λ.item()  # minimize turns this into an array
    yhat = sparse.linalg.inv(sparse.csc_matrix(I + λ * R)) @ y

    H_ = sparse.linalg.inv(sparse.csc_matrix(I + λ * (D_.T @ U @ D_)))

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
    δx[δx == 0] = 1e-9
    Vd = sparse.lil_matrix((n - d, n - d))
    Vd.setdiag(1 / δx)
    Dhat1 = sparse.eye(n - d, n - d + 1, 1) - sparse.eye(n - d, n - d + 1)
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

    B = sparse.lil_matrix((n, n))
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
    I = np.eye(len(x))
    return sparse.linalg.inv(I + (λs / δ) * (D_.T @ U @ D_))


def Vgcv(x, y, yhat, λs):
    # cross validation variance
    n = len(y)
    rss = np.square(yhat - y).sum()
    return (rss / n) / (1 - np.trace(H(x, y, λs)) / n) ** 2
