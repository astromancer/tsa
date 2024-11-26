
# std
from enum import Enum
from pathlib import Path

# third-party
import numpy as np
from loguru import logger

# local
from recipes import op
from recipes.iter import cofilter
from recipes.oo.slots import sanitize
from recipes.functionals import not_none

# relative
from . import txt
from .utils import split_mask, stack_arrays, unstack_arrays


# ---------------------------------------------------------------------------- #

class _SupportedFormats(Enum):

    @classmethod
    def _missing_(cls, ext):
        if isinstance(ext, Path):
            ext = ext.suffix.strip('.')
            if member := getattr(cls, ext.upper(), ()):
                return member

        raise ValueError(f'Unsupported format: ".{ext}"')

    @classmethod
    def supported(cls):
        return tuple(x.value for x in cls)

    @classmethod
    def check(cls, file):
        if isinstance(file, str):
            return file.endswith(cls.supported())

        if isinstance(file, Path):
            return file.suffix.strip('.') in cls.supported()

        raise TypeError(f'{type(file)}')


class SupportedFormats(_SupportedFormats):

    TXT = 'txt'
    NPY = 'npy'
    NPZ = 'npz'

    # TODO
    # HDF5 = 'hd5'

# ---------------------------------------------------------------------------- #


def get_metadata(data, keys):
    # meta
    meta = {key: val.item() if val.size == 1 else ''.join(val)
            for key, val in op.ItemMap(*keys)(data).items()}
    logger.debug('The following meta data was read: {}.', meta)

    return meta


class Reader:

    supported = SupportedFormats

    def __call__(self, filename, **kws):
        logger.info('Loading data from {}.', filename)
        filename = Path(filename)
        reader = getattr(self, SupportedFormats(filename).value)
        return reader(filename, **kws)

    txt = staticmethod(txt.read)

    def npz(self, filename, fields=('index', 'values', 'sigma'), order=...):

        data = np.load(filename, allow_pickle=True)
        out = (index, value, *sigma) = tuple(data.get(field) for field in fields)

        if (mask := data.get('mask')) is not None:
            assert len(mask) == len(value)
            mask = mask.astype(bool)
            value = np.ma.MaskedArray(value, mask)

        # resolve metadata
        meta = get_metadata(data, set(data.keys()) - set(fields))

        # if order is not ...:
        return out, meta

    def npy(self, filename):

        data = np.load(filename)

        namespace = {name: data[name] for name in data.dtype.fields}
        data = map(namespace.get, ('index', 'values', 'sigma'), [None] * 3)

        return data, {}


# Singleton
read = Reader()


# --------------------------------------------------------------------------- #

class Writer:

    txt = staticmethod(txt.write)

    def __call__(self, filename, index, values, sigma, **metadata):
        """
        Write measurement sequence data to file. Various formats are supported.

        Parameters
        ----------
        filename : Path-like
            Path to the destination file.
        index : array or None
            Independent variable. Ignored if None.
        values : array
            Data values.
        sigma : array or None
            Standard deviation uncertainty of data value. Ignored if None.

        """
        filename = Path(filename)
        method = getattr(self, SupportedFormats(filename).value)

        # extract mask
        values, sigma, mask = split_mask(values, sigma)

        # log info
        nrows, nseries = values.shape
        logger.info('Saving sequence data ({} rows, {} series, containing {} '
                    'masked points{}) to file: {}',
                    nrows, nseries, (0 if mask is None else mask.sum()),
                    # ', including meta data' if metadata else '',
                    '', filename)

        return method(filename, index, values, sigma, mask, **metadata)

    def npz(self, filename, index, values, sigma=None, mask=None, **metadata):

        # Get namespace, filtering `None` values
        namespace = sanitize(locals(), 'filename', 'metadata')
        namespace = cofilter(not_none, namespace.values(), namespace.keys())[::-1]
        namespace = dict(zip(*namespace))

        # save
        np.savez_compressed(filename, **namespace, **metadata)

    def npy(self, filename, index, values, sigma=None, mask=None, **metadata):

        if metadata:
            logger.info("Ignoring metadata since not supported by format: 'npy'")
        #
        namespace = sanitize(locals(), 'filename', 'metadata')
        namespace = cofilter(not_none, namespace.values(), namespace.keys())[::-1]
        namespace = dict(zip(*namespace))

        # merge data into single array for saving as npy

        # create dtype
        n_points, n_series = values.shape

        dtypes = [(name, data.dtype, ((1 if name == 'index' else n_series), ))
                  for name, data in namespace.items()]

        # merge
        data = np.empty(n_points, dtypes)
        for name, array in namespace.items():
            if array.ndim == 1:
                array = array.reshape((-1, 1))
            data[name] = array

        # write
        np.save(filename, data)


# Singleton
write = Writer()
