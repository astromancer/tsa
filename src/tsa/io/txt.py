"""
Write time series / frequency spectra to plain text in utf-8, emphasis on human
readable forms.
"""

# std
import re
import itertools as itt
from pathlib import Path

# third-party
import numpy as np
from loguru import logger

# local
from recipes import api, op
from recipes.io import read_lines
from recipes.config import ConfigNode
from recipes.functionals import not_none
from recipes.containers import split_where
from recipes.containers.utils import split
from recipes.pprint.mapping import pformat
from recipes.string import hstack, remove_prefix
from recipes.string.unicode import vertical_brace as vbrace

# relative
from .utils import split_mask, stack_arrays, unstack_arrays


# ---------------------------------------------------------------------------- #
CONFIG = ConfigNode.load_module(__file__, dot_split=True)

# write oflag data to file
REGEX_FORMAT_SPEC = re.compile(r'%[+\- ]?(\d{0,2})\.?(\d{0,2})?([if])')

COLUMN_SPEC = CONFIG.columns.info
COLUMN_INFO_NAME = 'Column Info'
SHAPE_INFO_NAME = 'Table Info'
UNIT_FORMAT = '[{}]'
HEADER_SCAN_LIMIT = 50

# ---------------------------------------------------------------------------- #


def read(filename, *_, order=..., **kws):
    """
    Read data from a file

    Parameters
    ----------
    filename : Path-like
        Path to the file.

    Returns
    -------
    tuple
        Time stamps, data values, standard deviation uncertainty of data.
    """

    header = read_lines(filename, HEADER_SCAN_LIMIT)
    meta_data = read_meta(header)

    ncols = int(meta_data[SHAPE_INFO_NAME]['n_cols'])

    flags = [op.index(header, f'# {CONFIG.columns.info[name].title}',
                      test=str.startswith, default=None)
             for name in ('index', 'sigma', 'mask')]

    # read
    data = np.genfromtxt(filename, delimiter=CONFIG.columns.sep,
                         usecols=range(ncols))

    # if filename.name == '20150228.020._X2.raw.txt':
    #     from IPython import embed
    #     embed(header="Embedded interpreter at 'src/tsa/io/txt.py':73")

    index, values, sigma = unstack_arrays(data, *map(not_none, flags))

    if sigma is not None:
        sigma = sigma[:, order]

    data = (index, values[:, order], sigma)

    return data, meta_data


# alias
read_text = read


def read_meta(lines):
    data = {}
    lines = [remove_prefix(line, '# ') for line in lines]

    sections = list(split_where(lines, '', offset=1))
    sections.remove([''])

    for name, _, *info, _ in sections:
        data[name] = read_block(info)

    return data


def convert_numeric(string):
    if string.isdigit():
        return int(string)
    try:
        return float(string)
    except ValueError:
        return string


def read_block(text):
    buffer = ''
    data = {}
    for line in text:
        if ':' in line:
            lhs, rhs = line.split(':', 1)
            data[lhs] = convert_numeric(rhs.strip())
        else:
            buffer = '\n'.join((buffer, line))

    if data:
        return data

    return buffer.lstrip('\n')


@api.synonyms(values='value')
def write(filename, index, values, sigma, mask=None, precision=6,
          sep=CONFIG.columns.sep, title=CONFIG.title, col_info=COLUMN_SPEC,
          target='', series_type='series', **metadata):
    """
    Write measurement sequence data to text file.

    Parameters
    ----------
    filename : str, Path
        Path to save data to.
    index : array
        Time stamps.
    values : array
        Data values to write.
    sigma : array
        Standard deviation uncertainty of data values.
    precision : int
        Numeric precision for float columns.
    sep : str
        Character used to separate columns.
    title : str, optional
        Title for header, by default CONFIG.title
    col_info : 
        Column information.
    target : str, optional
        Name of the target, by default None
    series_type : str
        The type of series that the data represents. This information is written
        to the document header.
    **metadata
        Meta data for header, by default None
    """

    # get the masked values as separate array for saving as column values
    values, sigma, mask = split_mask(values, sigma)
    _, n_series = values.shape

    # logger.info('Saving sequence data ({} rows, {} series, containing {} '
    #             'masked points{}) to file: {}',
    #             len(time), n_rows, n_series, (0 if mask is None else mask.sum()),
    #             ', including meta data' if meta else '', filename)

    # stack data
    data = stack_arrays(index, values, sigma, mask)

    n_rows, n_cols = data.shape
    shape_info = dict(n_rows=n_rows, n_cols=n_cols, n_series=n_series)

    # prepare
    first = True
    col_details = []
    col_info = ConfigNode(col_info)
    for name in ('index', 'values', 'sigma', 'mask'):
        if (col_data := eval(name)) is None:
            col_info.pop(name, None)
            continue

        # get format
        spec = dict(col_info[name])
        unit = spec.get('unit', '')
        unit = spec['unit'] = UNIT_FORMAT.format(unit) if unit else ''
        spec.pop('description')

        # auto format
        spec.setdefault('precision', precision)
        df, hf, col_width = _auto_format(col_data, **spec, comment_size=2 * first)
        first = False
        col_details.append((name, spec['title'], unit, df, hf, col_width))

    # duplicate per series
    base, per_series = split(col_details, ['index' in col_info])
    names, titles, units, data_fmt, head_fmt, widths = \
        zip(*(*base, *(per_series * n_series)))

    # adjust first column for comment
    if widths[0] > int(parse_format_spec(data_fmt[0])[0]):
        data_fmt = (data_fmt[0] + '  ', *data_fmt[1:])
    data_fmt = sep.join((*data_fmt, ''))

    # file header
    col_details = (names, titles, units, widths, head_fmt)
    description = metadata.pop('description', '')
    header_info = collect_metadata(title, target, description,
                                   shape_info, col_info,
                                   series_type, **metadata)

    header = format_header(header_info, col_details, target, n_series, sep)

    # write to file
    with Path(filename).open('w') as fp:
        fp.write(header)
        np.savetxt(fp, data, data_fmt)


# alias
write_text = write


def _auto_format(data, precision, title, unit, comment_size=0):

    dtype = data.dtype.kind
    if np.ma.is_masked(data):
        data = data.compressed()

    data = data[~np.isnan(data)]
    assert data.size

    mx, mn = data.max(), data.min()
    neg = mn < 0

    if dtype == 'b':
        dw = 1
        df = f'%{dw}i'
    elif dtype == 'i':
        dw = len(str(int(mx - mn)))
        df = f'%{" " * int(neg)}{dw}i'
    else:
        # data format
        mx = np.abs([mx, mn]).max()
        dw = max(int(np.ceil(np.log10(mx))), 1) + precision + 1
        df = f'%{" " * int(neg)}{dw}.{precision}f'

    # column format
    cw = max(len(title), len(unit))  # + 1
    dw += neg
    adjust = dw >= cw
    cw = max(dw, cw)
    if adjust:
        cw -= comment_size

    if (space := (cw - dw)) > 0:
        df += ' ' * space

    cf = f'{{: {CONFIG.columns.title_align}{cw}s}}'

    logger.debug('Column {!r}: data_fmt: {!r}, col_fmt: {!r}', title, df, cf)

    return df, cf, cw


def collect_metadata(title, target_name, description, shape_info, col_info,
                     series_type='series', **metadata):
    """
    Make a header for the file.

    Parameters
    ----------
    title : str
        Title for the header.
    target_name : str
        Name of the target object.
    shape_info : dict
        Information about the shape of the data.
    col_info : dict[dict]
        Info for the columns. The dict is keyed on the column variate
        type (index, value, sigma, mask). Fields are dicts with 
        'title' and 'description' for each column.
    metadata : dict, optional
        Meta data for the header, by default None.
    series_type : str, optional
        Type of series, by default 'series'.

    Returns
    -------
    dict
        Meta data to save in file header.
    """

    # get column info
    n_series = shape_info['n_series']

    # format column name: description
    titles = col_info.find('title', collapse=True)
    descriptions = col_info.find('description', collapse=True)

    offset = int('index' in descriptions)
    postscript = vbrace(len(descriptions) - offset, f'x{n_series} {series_type}')
    info_block = hstack(['\n'.join(descriptions.values()), postscript],
                        spacing=3, offsets=offset, rstrip=True)
    descriptions = dict(zip(titles.values(), info_block.splitlines()))

    return {
        # title, table shape info
        f'# {title.format(target_name)}': description,  # .format(target_name)
        SHAPE_INFO_NAME:  shape_info,
        # column descriptions
        COLUMN_INFO_NAME: descriptions,
        **metadata
    }


def format_header(header_info, col_details, target_name, n_series, sep):
    lines = _gen_header_lines(header_info, col_details, target_name, n_series, sep)
    return '\n'.join(lines).replace('\n', '\n# ')[:-2]


def _gen_header_lines(header_info, col_details, target_name, n_series, sep):

    names, titles, units, widths, head_fmt = col_details

    # header blocks for additional meta data
    yield from map(header_info_block, *zip(*header_info.items()))

    # section divider
    yield (hline := '-' * (sum(widths) + len(names) * len(sep)))

    # object names
    n_cols = len(names)
    n_col_per_series = len(set(header_info[COLUMN_INFO_NAME].keys())) - 1

    group_widths = np.reshape(widths[1:], (-1, n_col_per_series)).sum(1) + len(sep)
    target_names = [target_name or 'C0', *map('C{}'.format, range(1, n_series))]
    group_head_line = ' ' * widths[0] + sep
    for name, width in zip(target_names, group_widths):
        group_head_line += f'{name: <{width}}{sep}'

    yield group_head_line

    # column titles
    col_head_fmt = sep.join((*head_fmt, ''))
    yield from itt.starmap(col_head_fmt.format, (titles, units))

    # section divider
    yield hline
    yield ''  # advance to new line


def parse_format_spec(fmt):
    """
    Parse a format specifier into width, precision, and data type.

    Parameters
    ----------
    fmt : str
        Format specifier.

    Returns
    -------
    tuple
        A tuple containing width, precision, and data type.

    Raises
    ------
    ValueError
        If the format specifier is invalid.
    """
    if mo := REGEX_FORMAT_SPEC.match(fmt):
        return mo.groups()  # width, precision, dtype
    else:
        raise ValueError('Invalid format specifier!')


def header_info_block(name, info):
    """
    Create a header information block.

    Parameters
    ----------
    name : str
        Name of the block.
    info : dict
        Information to include in the block.

    Returns
    -------
    str
        Header information block.
    """
    return '\n'.join(_header_info_block(name, info))


def _header_info_block(name, info):
    if name:
        yield underline_ascii(name)

    if isinstance(info, str):
        if info:
            yield info
    else:
        yield pformat(info, name='', lhs=str,  rhs=get_name, brackets='', sep='')

    yield ''


def underline_ascii(text):
    """
    Underline a given text with ASCII dashes "-".

    Parameters
    ----------
    text : str
        Text to underline.

    Returns
    -------
    str
        Underlined text.
    """
    return '\n'.join([text, '-' * len(text)])


def get_name(o):
    """
    Get the name of an object.

    Parameters
    ----------
    o : object
        Object to retriev the name of.

    Returns
    -------
    str
        Name of the object.
    """
    return o.__name__ if callable(o) else str(o)
