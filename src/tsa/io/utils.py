
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

    components = [values.T]
    if sigma is not None:
        components.append(sigma.T)

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


def unstack_arrays(data, has_index, has_sigma, has_mask):

    index = data[:, 0] if has_index else None

    step = sum((has_sigma, has_mask, 1))
    values = data[:, has_index::step]
    if has_mask:
        mask = data[:, sum((has_index, has_sigma, 1))::step]
        values = np.ma.MaskedArray(values, mask)

    sigma = data[:, has_index + 1::step] if has_sigma else None

    return index, values, sigma


# def apply_mask(values, mask):
    
    

def split_mask(values, sigma):

    values = np.asanyarray(values).squeeze()
    if have_sigma := sigma is not None:
        sigma = np.asanyarray(sigma).squeeze()

    if values.ndim == 1:
        values = values[:, None]
        if have_sigma:
            sigma = sigma[:, None]

    mask = None
    if np.ma.is_masked(values) or np.ma.is_masked(sigma):
        mask = np.ma.getmaskarray(values) | np.ma.getmaskarray(sigma)

    return values, sigma, mask
