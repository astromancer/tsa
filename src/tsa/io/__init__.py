
# std
from enum import Enum
from pathlib import Path

# third-party
import numpy as np
from loguru import logger

# local
from recipes import io
from recipes.iter import cofilter

# relative
from . import txt


# ---------------------------------------------------------------------------- #

class SupportedFileType(Enum):

    TXT = 'txt'
    NPY = 'npy'
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

    def __call__(self, filename, hdu=None, **kws):
        logger.info('Loading data from {}.', filename)
        filename = Path(filename)
        reader = getattr(self, SupportedFileType(filename).value)
        return reader(filename, hdu, **kws)

    def npy(self, filename, hdu, order=...):

        # CONFIG.pre_subtract
        # since the (gain) calibrated frames are being used below,
        # CCDNoiseModel(hdu.readout.noise)

        flux = io.load_memmap(filename)['flux']
        return hdu.t.bjd, flux['value'][:, order], flux['sigma'][:, order]

    def npz(self, filename, hdu=None, fields=('index', 'value', 'sigma')):

        data = np.load(filename)
        index, value, *sigma = tuple(data.get(field, None) for field in fields)

        if (flag := data.get('flag', None)) is not None:
            assert len(flag) == len(value)
            flag = flag.astype(bool)
            value = np.ma.MaskedArray(value, flag)

        return index, value, *sigma


# Singleton
read = Reader()

# --------------------------------------------------------------------------- #


class Writer:

    txt = staticmethod(txt.write)

    def __call__(self, filename, index, value, sigma, **kws):
        filename = Path(filename)
        method = getattr(self, SupportedFileType(filename).value)
        return method(filename, index, value, sigma, **kws)

    def npy(self, filename, index, value, sigma, mask=None, **kws):

        if np.ma.isMA(value) or np.ma.isMA(sigma):
            mask = np.ma.getmaskarray(value) | np.ma.getmaskarray(sigma)

        logger.info('Saving light curve data ({} rows, {} sources, {} masked '
                    'points{}) to file: {}',
                    len(index), len(value), (0 if mask is None else mask.sum()),
                    '', filename)

        # stack data
        data = stack_arrays(index, value, sigma, mask)

        return np.save(filename, data)

    def npz(self, filename, index, value, **kws):

        # filter `None` values
        kws = dict(zip(*cofilter(None, kws.values(), kws.keys())[::-1]))
        np.savez_compressed(filename, index=index, value=value, **kws)


# Singleton
write = Writer()


# ---------------------------------------------------------------------------- #
# Utility function

def stack_arrays(index, flx, sigma, flag=None):
    """
    Stack light curve data into table for writing to file. Measurements for
    each star (Flux, σFlux, ...) columns are horizontally stacked.

    Parameters
    ----------
    index
    flx
    sigma
    flag

    Returns
    -------

    """
    nstars = len(flx)
    assert len(sigma) == nstars

    components = [flx, sigma]
    if flag is not None:
        assert len(flag) == nstars
        flag = flag.astype(int)
        components.append(flag)

    tbl = [index]
    for columns in zip(*components):
        tbl.extend(columns)

    # convert to array
    return np.array(tbl).T
