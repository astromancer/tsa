
# std
import warnings
from collections import defaultdict

# third-party
import numpy as np
from scipy.signal import get_window
from astropy.stats import sigma_clipped_stats
from loguru import logger

# local
from recipes.array import fold

# relative
from .spectral import resolve_overlap
from .windowing import resolve_size


# TODO: OCSVM, Tietjen-Moore, Topological Anomaly detection


def generalizedESD(x, kmax='0.02%', alpha=0.05, full_output=False):
    """
    Carry out a Generalized ESD Test for Outliers.

    The Generalized Extreme Studentized Deviate (ESD) test for outliers can be
    used to search for outliers in a univariate data set, which approximately
    follows a normal distribution. A description of the algorithm is, e.g.,
    given at `Nist
    <http://www.itl.nist.gov/div898/handbook/eda/section3/eda35h3.htm>`_ or
    [Rosner1983]_.

    Parameters
    ----------
    x:  array-like
        Data to test.
    kmax : int
        Maximum number of outliers in the data set.
    alpha : float, optional
        Significance (default is 0.05).
    full_output : boolean, optional
        Determines whether additional return values are provided. Default is
        False.

    Returns
    -------
    Number of outliers : int
        The number of data points characterized as outliers by the test.
    Indices : list of ints
        The indices of the data points found to be outliers.
    R : list of floats, optional
        The values of the "R statistics". Only provided if `full_output` is set
        to True.
    L : list of floats, optional
        The lambda values needed to test whether a point should be regarded an
        outlier. Only provided if `full_output` is set to True.
    """

    from scipy.stats import t

    kmax = resolve_size(kmax, len(x))

    xm = np.ma.array(x.copy())
    n = len(xm)
    kmax = min(kmax, n)  # can't have more outliers than points!

    # Compute R-values
    R, L = [], []
    idx = []
    for i in range(kmax + 1):
        # Compute mean and std of x
        xmean, xstd = xm.mean(), xm.std()
        # Find maximum deviation (Malhanobis distance)
        rr = np.abs((xm - xmean) / xstd)
        # index of data point with maximal Malhanobis distance
        wrr = np.argmax(rr)
        idx.append(wrr)
        R.append(rr[wrr])
        if i >= 1:
            p = 1.0 - alpha / (2.0 * (n - i + 1))
            per_point = t.ppf(p, n - i - 1)
            # the 100p percentage point from the t-distribution
            L.append(
                (n - i) * per_point / np.sqrt(
                    (n - i - 1 + per_point ** 2) * (n - i + 1))
            )
        # Mask that value and proceed
        xm[idx[-1]] = np.ma.masked

    # Remove the last entry from R, which is of no meaning for the test
    R.pop(-1)
    # Get number of outlier points
    k = next((i + 1 for i in range(kmax - 1, -1, -1) if R[i] > L[i]), 0)
    out = idx[:k]
    return (out, k, R, L, idx) if full_output else out


# alias
gESD = generalizedESD

# def CovEstOD(data, classifier=None, threshold=0):
#     if classifier is None:
#     from sklearn.covariance import EllipticEnvelope
#     classifier = EllipticEnvelope(support_fraction=1., contamination=0.1)
#
#     classifier.fit(data)
#     idx, = np.where( classifier.decision_function(data) < threshold )
#
#     return idx


def CovEstOD(data, classifier=None, n=1, **kw):
    # multivariate outlier detection

    if classifier is None:
        from sklearn.covariance import EllipticEnvelope

        contamination = n / data.shape[0]
        classifier = EllipticEnvelope(support_fraction=1.,
                                      contamination=contamination)

    classifier.fit(data)
    outliers, = np.where(classifier.predict(data) == -1)
    return outliers


def get_ellipse(classifier, **kws):
    from matplotlib.patches import Ellipse

    # todo: probably check that data is 2d

    w, T = np.linalg.eigh(classifier.precision_)
    # T (eigenvectors of precision matrix) is the transformation matrix
    # between principle axes and data coordinates
    Ti = np.linalg.inv(T)
    M = np.dot(Ti, classifier.precision_) * T
    # Diagonalizing the precision matrix ==> quadratic representation of
    # decision boundary (ellipse): z^T M z = threshold. where x-<x> = Tz
    # transforms to principle axes
    a, b = np.sqrt(classifier.threshold / np.diag(M))
    # a, b are semi-major & semi-minor axes

    # T is (im)proper rotation matrix
    theta = np.degrees(np.arccos(T[0, 0]))
    theta = np.linalg.det(T) * theta
    # If det(T)=-1 ==> improper rotation matrix (roto-inversion -
    # one of the axes is inverted)
    return Ellipse(classifier.location_, 2 * a, 2 * b, theta, **kws)


def detect_window(data, nwindow, noverlap,
                  method=gESD, weight_kernel='boxcar', threshold=0.5,
                  *args, **kws):
    """
    Outlier detection using moving window

    Parameters
    ----------
    data: array-like
        The data set to be tested for outliers
    nwindow: int
        window size
    noverlap: int
        overlap from one window to next
    method: callable
        function to be used for outlier detection on each window
    weight_kernel: str | np.array, optional
        window function to weight the outlier probabilities for each window.
        Default is uniform weighting.
    return_index: bool
    return_mask: bool
    return_masked_data: bool
    args
    kws

    Returns
    -------

    """
    # TODO: paralellize!

    noverlap = resolve_overlap(nwindow, noverlap)

    assert data.ndim == 1

    n = len(data)
    assert n > 1

    if n < nwindow:
        warnings.warn(f'Data length {n} is smaller than window size {nwindow}! '
                      'Setting the window size to data size.')
        return method(data, *args, **kws)

    step = nwindow - noverlap
    noc = fold.get_n_repeats(n, nwindow, noverlap)

    weight_kernel = weight_kernel or 'boxcar'
    weights = get_window(weight_kernel, nwindow)

    prob = defaultdict(int)
    for i, section in enumerate(folded := fold.fold(data, nwindow, noverlap)):
        logger.debug('Section {}/{}.', i, len(folded))
        if np.ma.is_masked(section):
            section = section[~section.mask]

        # indices of outliers relative to this window
        widx = method(section.T, *args, **kws) if len(section) else []
        # can be that the entire sectionment is masked

        if len(widx):
            didx = i * step + np.array(widx)  # indices relative to data
            didx = didx[didx < n]  # remove indices that exceed array dimensions
            for ii, jj in zip(widx, didx):
                prob[jj] += weights[ii] / noc[jj]
                # mean probability that points where flagged as outliers

    return np.sort([idx for idx, p in prob.items() if p > threshold]).astype(int)

    # if return_index:
    #     return indices

    # if return_mask:
    #     mask = np.zeros(data.shape, bool)
    #     if len(indices):
    #         mask[indices] = True
    #     return mask

    # if return_masked_data:
    #     if np.ma.isMA(data):
    #         data.mask[indices] = True
    #     else:
    #         mask = np.zeros_like(data, bool)
    #         mask[indices] = True
    #         data = np.ma.masked_where(mask, data)

    #     return data


WindowOutlierDetection = detect_window


def sigma_clip_masked(x, siglow=3, sighi=3):
    xmean, xmed, xstd = sigma_clipped_stats(x)
    return np.ma.masked_outside(x, xmed - siglow * xstd, xmed + sighi * xstd)


def running_sigma_clip(x, sig=3., nwindow=100, noverlap=0, iters=None,
                       cenfunc=np.ma.median, varfunc=np.ma.var):
    # TODO:  Incorporate in WindowOutlierDetection

    # NOTE: SLOWWWWWWWWW...................

    if noverlap:
        print('Overlap not implemented yet!  Setting noverlap=0')
        noverlap = 0

    sections = fold.fold(x, nwindow, 0)

    filtered_data = []
    for sec in sections:
        #
        filtered_sec = np.ma.masked_where(np.isnan(sec), sec)

        if iters is None:
            i = -1
            lastrej = filtered_sec.count() + 1
            while (filtered_sec.count() != lastrej):
                i += 1
                lastrej = filtered_sec.count()
                secdiv = filtered_sec - cenfunc(filtered_sec)
                filtered_sec.mask |= np.ma.greater(secdiv * secdiv,
                                                   varfunc(secdiv) * sig ** 2)
                # print( filtered_sec.mask )
                # iters = i + 1
        else:
            for _ in range(iters):
                secdiv = filtered_sec - cenfunc(filtered_sec)
                filtered_sec.mask |= np.ma.greater(secdiv * secdiv,
                                                   varfunc(secdiv) * sig ** 2)

        filtered_data.append(filtered_sec)

    return np.ma.concatenate(filtered_data)[:len(x)]


def plot_clippings(ax, t, x, tclp, xclp, med, std, threshold, nwindow=0,
                   label='data', **kw):
    # med, v = running_stats(x, nwindow, center=False)
    # std = np.sqrt( v )

    ax.plot(t, x, 'go', ms=3, label=label)
    ax.plot(tclp, xclp, 'x', mec='r', mew=1, label='clipped')

    # print( 'top', len(top), 'bottom', len(bottom), 't', len(t[st:end]) )
    sigma_label = f'{threshold}$\sigma$ ($N_w={nwindow}$)'
    median_label = f'median ($N_w={nwindow}$)'
    ax.plot(t, med + threshold * std, '0.6')
    ax.plot(t, med - threshold * std, '0.6', label=sigma_label)
    ax.plot(t, med, 'm-', label=median_label)

    sw = kw['show_window'] = kw.pop('sw', 0)
    for i in range(sw):  # itt.islice( fold.gen(x, nwindow), 0, sw ):
        t0, tnw = t[i * nwindow], t[(i + 1) * nwindow]  # t[nwindow]
        xw = x[i * nwindow:(i + 1) * nwindow]
        xmed = np.median(xw)
        xad = np.abs(xw - xmed).max()
        xmin = xmed - xad  # xmax = xmed+xad

        rect = Rectangle((t0, xmin), tnw - t0, 2 * xad, fc='g', alpha=.5)
        ax.add_artist(rect)

    # clp = sigma_clip( lcr, 3 )
    # ax.plot( t[clp.mask], lcr[clp.mask], 'ko', mfc='None', mec='k', mew=1.75, ms=10 )
    # m, ptp = np.mean(med), np.ptp(med)
    # ax.set_ylim( m-3*ptp, m+3*ptp )

    # white_frac = 0.025
    # xl, xu, xd = t.min(), t.max(), t.ptp()
    # ax.set_xlim( xl-white_frac*xd, xu+white_frac*xd )

    # ax.set_title( self.name )
    # ax.invert_yaxis()
    ax.grid()
    ax.legend(loc='best')
