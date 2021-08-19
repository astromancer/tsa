from scipy.stats import binned_statistic
import numpy as np
import multiprocessing as mp
import functools as ftl
import itertools as itt


# from recipes.decorators.base import DecoratorBase

class map_to_array(object):
    def __init__(self, fun, array):
        self.fun = fun
        self.array = array
        self.every = self.array.size // 20

    def __call__(self, i, *args):
        if (i % self.every) == 0:
            print('{:.0%}'.format(i / self.array.size))
        self.array[i] = self.fun(*args)


#  TODO: make this a method of PhotRun
def fold_stat(t0, p, offsets, data, n_bins=100, statistic=np.mean):
    # night to night magnitude offset
    #     s = 0
    #     for t, d, e in data:
    #         s += y.mask.sum()

    # stack light curves
    t, y, e = map(np.ma.hstack, zip(*add_offsets(data, offsets)))
    # assert s = d.mask.sum()
    use = ...
    if np.ma.is_masked(y):
        use = ~y.mask

    phase = (t - t0) / p
    phaseMod1 = (phase + 0.5) % 1 - 0.5
    result = binned_statistic(phaseMod1[use], y[use], statistic, n_bins,
                              (-0.5, 0.5))
    return phaseMod1, y, result


def add_offsets(data, offsets):
    for (t, d, u), o in zip(data, offsets):
        yield t, d + o, u


def objective_sum_of_variance(data, offsets, n_bins, t0, p):
    ph, dat, stat = fold_stat(t0, p, offsets, data, n_bins, np.var)
    return np.nansum(stat.statistic)


def et_tu_brute(p, t0, offsets, data, n_bins=100):
    # brute force period search
    r = np.empty_like(p)
    sov = ftl.partial(objective_sum_of_variance, data, offsets, n_bins, t0)
    fun = map_to_array(sov, r)
    list(itt.starmap(fun, enumerate(p)))

    # with mp.Pool() as pool:
    #     pool.starmap(fun, enumerate(p))
    # pool.join()
    return r


def guess_offsets(data):
    # use mean deviation (anomaly) as magnitude offsets
    med = [np.median(d[1]) for d in data]
    return np.mean(med) - med
