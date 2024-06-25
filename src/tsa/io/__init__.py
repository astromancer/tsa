
# std
from enum import Enum
from pathlib import Path

# third-party
import numpy as np
from loguru import logger

# local
from recipes import io

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

    def npz(self, filename):
        data = np.load(filename)
        t, y, σ = tuple(data[field] for field in ('t', 'counts', 'sigma', 'flag'))

        if (flag := data.get('flag', None)) is not None:
            assert len(flag) == len(y)
            flag = flag.astype(bool)
            y = np.ma.MaskedArray(y, flag)
            
        return t, y, σ


# Singleton
read = Reader()

# --------------------------------------------------------------------------- #

class Writer:

    def __call__(self, filename, t, counts, std, **kws):
        filename = Path(filename)
        method = getattr(self, SupportedFileType(filename).value)
        return method(filename, t, counts, std, **kws)

    def npy(self, filename, t, counts, std, mask=None, **kws):

        if np.ma.isMA(counts) or np.ma.isMA(std):
            mask = np.ma.getmaskarray(counts) | np.ma.getmaskarray(std)

        logger.info('Saving light curve data ({} rows, {} sources, {} masked '
                    'points{}) to file: {}',
                    len(t), len(counts), (0 if mask is None else mask.sum()),
                    '', filename)

        # stack data
        data = stack_arrays(t, counts, std, mask)

        return np.save(filename, data)

    def npz(self, filename, t, counts, std, flag=None, **kws):
        np.savez_compressed(filename,
                            t=t, counts=counts, std=std,
                            **({'flag': flag} if flag is not None else {}))


# Singleton
write = Writer()


# ---------------------------------------------------------------------------- #
# Utility function

def stack_arrays(t, flx, std, flag=None):
    """
    Stack light curve data into table for writing to file. Measurements for
    each star (Flux, σFlux, ...) columns are horizontally stacked.

    Parameters
    ----------
    t
    flx
    std
    flag

    Returns
    -------

    """
    nstars = len(flx)
    assert len(std) == nstars

    components = [flx, std]
    if flag is not None:
        assert len(flag) == nstars
        flag = flag.astype(int)
        components.append(flag)

    tbl = [t]
    for columns in zip(*components):
        tbl.extend(columns)

    # convert to array
    return np.array(tbl).T
