
import numpy as np


def stack_arrays(time, values, sigma, mask=None):
    """
    Stack time series data into table for writing to file. Measurements for
    each series (value, sigma, [flag]) columns are horizontally stacked.

    Parameters
    ----------
    time : array
        Time stamps.
    values : array
        Data values.
    sigma : array
        Standard deviation uncertainty of data values.
    mask: array, optional
        Boolean mask for data values.

    Returns
    -------
    array
        Array containing stacked data columns.
    """
    # nseries = values.shape
    # assert len(sigma) == nseries

    components = [values.T, sigma.T]
    if mask is not None:
        # mask = np.atleast_2d(mask)
        assert mask.shape == values.shape
        mask = mask.astype(int)
        components.append(mask.T)

    tbl = [time]
    for columns in zip(*components):
        tbl.extend(columns)

    # convert to array
    return np.array(tbl).T


def unstack_arrays(data, oflag):

    oflag = int(oflag is not None)
    step = 2 + oflag
    values, sigma, *oflag = (data[:, i::step] for i in range(1, 3 + oflag))

    if oflag:
        values = np.ma.MaskedArray(values, oflag[0])

    return data[:, 0], values, sigma


def split_mask(values, sigma):

    values = np.asanyarray(values).squeeze()
    sigma = np.asanyarray(sigma).squeeze()

    if values.ndim == 1:
        values = values[:, None]
        sigma = sigma[:, None]

    if np.ma.isMA(values) or np.ma.isMA(sigma):
        mask = np.ma.getmaskarray(values) | np.ma.getmaskarray(sigma)

    return values, sigma, mask
