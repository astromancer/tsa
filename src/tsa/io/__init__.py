
# std
from enum import Enum
from pathlib import Path

# third-party
import numpy as np
from loguru import logger

# local
from recipes.iter import cofilter
from recipes.oo.slots import sanitize
from recipes.functionals import not_none

# relative
from . import txt
from .utils import split_mask, stack_arrays, unstack_arrays


# ---------------------------------------------------------------------------- #

class SupportedFileType(Enum):

    TXT = 'txt'
    NPZ = 'npz'
    # DAT = 'dat'
    # FITS = 'fits'
    # hd5

    @classmethod
    def _missing_(cls, ext):
        if isinstance(ext, Path):
            if member := getattr(cls, ext.suffix.strip('.').upper(), ()):
                return member

        raise ValueError(f'Unsupported format: {ext!r}')

    @classmethod
    def check(cls, file):
        if isinstance(file, str):
            return file.endswith(SUPPORTED)

        if isinstance(file, Path):
            return file.suffix.strip('.') in SUPPORTED

        raise TypeError(f'{type(file)}')


#
SUPPORTED = tuple(x.value for x in SupportedFileType)


# ---------------------------------------------------------------------------- #
class Reader:

    txt = staticmethod(txt.read)

    def __call__(self, filename, **kws):
        logger.info('Loading data from {}.', filename)
        filename = Path(filename)
        reader = getattr(self, SupportedFileType(filename).value)
        return reader(filename, **kws)

    # def memmap(self, filename, hdu, order=..., names=None):

        # CONFIG.pre_subtract
        # since the (gain) calibrated frames are being used below,
        # CCDNoiseModel(hdu.readout.noise)

        # FIXME:
        # flux = io.load_memmap(filename)['flux']
        # return hdu.t.bjd, flux['value'][:, order], flux['sigma'][:, order]

    def npz(self, filename, fields=('index', 'values', 'sigma')):

        data = np.load(filename)
        index, value, *sigma = tuple(data.get(field, None) for field in fields)

        if (mask := data.get('mask', None)) is not None:
            assert len(mask) == len(value)
            mask = mask.astype(bool)
            value = np.ma.MaskedArray(value, mask)

        # return dict(zip(fields, filter(None, (index, value, sigma))))
        return index, value, *sigma


# Singleton
read = Reader()

# --------------------------------------------------------------------------- #


class Writer:

    txt = staticmethod(txt.write)

    def __call__(self, filename, index, values, sigma, **kws):
        """
        Write measurement sequence data to file. Various formats are supported.

        Parameters
        ----------
        filename : Path-like
            Path to the destination file.
        index : array
            Independent variable.
        values : array
            Data values.
        sigma : array
            Standard deviation uncertainty of data value.

        """
        filename = Path(filename)
        method = getattr(self, SupportedFileType(filename).value)

        # extract mask
        values, sigma, mask = split_mask(values, sigma)

        # log info
        nrows, nseries = values.shape
        logger.info('Saving sequence data ({} rows, {} series, containing {} '
                    'masked points{}) to file: {}',
                    nrows, nseries, (0 if mask is None else mask.sum()),
                    #  ', including meta data' if meta else ''
                    '', filename)

        return method(filename, index, values, sigma, mask=mask, **kws)


    def npz(self, filename, index, values, sigma=None, **kws):

        # Get namespace, filtering `None` values
        kws.update(sanitize(locals(), 'filename'))
        kws = dict(zip(*cofilter(not_none, kws.values(), kws.keys())[::-1]))

        # save
        np.savez_compressed(filename, **kws)


# Singleton
write = Writer()
