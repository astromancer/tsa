"""
Write time series / frequency spectra to plain text in utf-8, emphasis on human
readable forms.
"""

# std
import re
import textwrap
from pathlib import Path

# third-party
import numpy as np
import more_itertools as mit
from loguru import logger

# local
from recipes import op
from recipes.io import read_lines
from recipes.string import hstack
from recipes.config import ConfigNode
from recipes.pprint.mapping import pformat

# relative
from .utils import stack_arrays, unstack_arrays, split_mask


# ---------------------------------------------------------------------------- #
CONFIG = ConfigNode.load_module(__file__, dot_split=True)

# write oflag data to file
REGEX_FORMAT_SPEC = re.compile(r'%(\d{0,2})\.?(\d{0,2})([if])')

COLUMN_INFO = 'Column Info'
UNIT_FORMAT = '[{}]'

MULTILINE_CURLY_BRACKET = textwrap.dedent(
    '''
    ⎫
    ⎬ x%i %s
    ⎭
    '''
)
# U+23aa Sm CURLY BRACKET EXTENSION ⎪
# U+23ab Sm RIGHT CURLY BRACKET UPPER HOOK ⎫
# U+23ac Sm RIGHT CURLY BRACKET MIDDLE PIECE ⎬
# U+23ad Sm RIGHT CURLY BRACKET LOWER HOOK ⎭
# ---------------------------------------------------------------------------- #


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
        return mo.groups()  # width, precision, dtype =
    else:
        raise ValueError('Invalid format specifier!')


def format_list(data, fmt='%g', width=8, sep=','):
    """
    Format a list of data into a string.

    Parameters
    ----------
    data : list
        List of data to format.
    fmt : str, optional
        Format string, by default '%g'.
    width : int, optional
        Width of each formatted element, by default 8.
    sep : str, optional
        Separator between elements, by default ','.

    Returns
    -------
    str
        Formatted string.
    """
    lfmt = f'%-{width}s' * len(data)
    s = lfmt % tuple(np.char.mod(fmt + sep, data))
    return s[::-1].replace(',', ' ', 1)[::-1].join('[]')


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

    yield pformat(info, name='', lhs=str,  rhs=get_name, brackets='', sep='')
    yield ''


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


def check_column_widths(names, formats):
    """
    Check and adjust column widths based on names and format specifiers.

    Parameters
    ----------
    names : list
        List of column names.
    formats : list
        List of format specifiers.

    Returns
    -------
    tuple
        Three lists containing widths, precisions, and data types.
    """
    widths, precisions, dtypes = [], [], []
    for name, fmt in zip(names, formats):
        width, precision, dtype = parse_format_spec(fmt)
        width = int(width or 1)
        width = max(width, len(name) + 1)

        widths.append(width)
        precisions.append(precision)
        dtypes.append(dtype)

    return widths, precisions, dtypes


# def make_format_spec(names, precisions, dtype='s'):
#     """
#     Adjust the column format specifiers to accommodate width of the column names
#     """
#
#     widths = list(map(len, names))
#     col_fmt_data = ''.join(map('%%-%i.%s%s'.__mod__,
#                                zip(widths, precisions, dtype)))
#     return widths, col_fmt_data


def write_header_aligned(a, out):
    """
    Write the header aligned with the data.

    Parameters
    ----------
    a : array
        Array containing the data.
    out : str, Path
        Output file path.
    """

    # fn = '/home/hannes/work/pyshoc/pyshoc/data/SHOC1.txt'
    # a = np.genfromtxt(fn, dtype=None, names=True, encoding=None)

    a = a.astype([(_, t.replace('S', 'U')) for _, t in a.dtype.descr])

    widths = np.array(list(map(len, a.dtype.names)))
    fmt_head = ' '.join(map('%%-%is'.__mod__, widths))
    widths[0] += 2
    fmt = tuple(map('%%-%i%s'.__mod__, zip(widths, 's' * len(widths))))

    # out = '/home/hannes/work/pyshoc/pyshoc/data/SHOC1.txt'
    header = fmt_head % a.dtype.names
    np.savetxt(out, a, fmt, header=header)


def make_column_format(names, formats):
    """
    Adjust the column format specifiers to accommodate width of the column names.

    Parameters
    ----------
    names : list
        List of column names.
    formats : list
        List of format specifiers.

    Returns
    -------
    tuple
        Column widths, format string for the header, and format string for the data.
    """
    widths, precisions, dtypes = check_column_widths(names, formats)
    col_fmt_head = ''.join(map('%%-%is'.__mod__, [widths[0] - 2, *widths[1:]]))
    col_fmt_data = ''.join(map('%%-%i.%s%s'.__mod__,
                               zip(widths, precisions, dtypes)))
    return widths, col_fmt_head, col_fmt_data


def _make_name_format(col_widths, has_oflag):
    w0, *ww = col_widths
    w2 = map(sum, mit.grouper(ww, 2 + has_oflag, fillvalue=ww))  # 2-column widths
    return ''.join('%%-%is' % w for w in [w0 - 2, *w2])


def get_column_info(nseries, col_info, has_oflag, precision=CONFIG.precision,
                    series_type='series'):
    """
    Get column information.

    Parameters
    ----------
    nseries : int
        Number of series.
    col_info : dict
        Column information.
    has_oflag : bool
        Flag indicating if outlier flag is present.
    precision : int, optional
        Precision of the data, by default CONFIG.precision.
    series_type : str, optional
        Type of series, by default 'series'.

    Returns
    -------
    tuple
        Names, units, formats, and info for the columns.
    """

    _names, _units, descript = zip(*col_info.values())
    info = dict(zip(_names, descript))

    names = [col_info['index'][0]]
    units = [col_info['index'][1]]
    formats = ['%18.9f']

    col_names_per_series, col_units_per_series, _ = \
        map(list, zip(col_info['value'], col_info['sigma']))
    col_fmt_per_series = [f'%12.{precision}f'] * 2

    # outlier detection parameters block
    if has_oflag:
        col_names_per_series.append(col_info['outlier'][0])
        col_units_per_series.append('')
        col_fmt_per_series.append('%i')
    else:
        # oflag_desc = ''
        info.pop(col_info['outlier'][0])
        info[''] = ' '  # place holder

    # column descriptions
    info_text = hstack(('\n'.join(info.values()),
                        MULTILINE_CURLY_BRACKET % (nseries, series_type)), 3)

    info.update(zip(info, map(str.rstrip, info_text.splitlines())))

    # build column headers
    for _ in range(nseries):
        names.extend(col_names_per_series)
        units.extend(col_units_per_series)
        formats.extend(col_fmt_per_series)

    # prepend comment str in such a way as to not screw up alignment with data
    units = list(map(UNIT_FORMAT.format, units))

    return names, units, formats, info


def make_header(title, target_name, shape_info, col_info, has_oflag, meta=None,
                precision=CONFIG.precision, series_type='series'):
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
    col_info : dict
        Information about the columns.
    has_oflag : bool
        Flag indicating if outlier flag is present.
    meta : dict, optional
        Meta data for the header, by default None.
    precision : int, optional
        Precision of the data, by default CONFIG.precision.
    series_type : str, optional
        Type of series, by default 'series'.

    Returns
    -------
    tuple
        Header string and column format string for the data.
    """

    if meta is None:
        meta = {}
    # todo: delimiter ??

    # get column info
    nseries = shape_info['nseries']
    names, units, formats, col_info = get_column_info(
        nseries, col_info, has_oflag, precision, series_type)

    # adjust the formatters
    col_widths, col_fmt_head, col_fmt_data = make_column_format(names, formats)

    info = {
        #  title, table shape info
        f'# {title}': shape_info,
        # column descriptions
        COLUMN_INFO: col_info,
        **meta
    }

    # column headers block
    lines = _make_header(info, target_name, nseries, has_oflag,
                         (names, units, col_widths, col_fmt_head))
    return '\n'.join(lines).replace('\n', '\n# ')[:-2], col_fmt_data


def _make_header(header_info, target_name, nseries, has_oflag, col_spec):

    *col_names_units, col_widths, col_fmt_head = col_spec

    # header blocks for additional meta data
    yield from map(header_info_block, *zip(*header_info.items()))

    # header as commented string
    # prepend comment character
    # header = '\n'.join(lines).replace('\n', '\n# ')

    yield (hline := '-' * (sum(col_widths) - 2))

    # object names
    target_names = ('', target_name, *(f'C{i}' for i in range(nseries - 1)))
    yield _make_name_format(col_widths, has_oflag) % target_names

    # column titles
    for o in col_names_units:
        yield col_fmt_head % tuple(o)

    yield hline
    yield ''  # advance to new line


def write(filename, time, values, sigma, mask=None,
          title=CONFIG.title, col_info=CONFIG.columns, meta=None,
          target='<unknown>', precision=CONFIG.precision, series_type='series'):
    """
    Write to text file.

    Parameters
    ----------
    filename : str, Path
        Path to save data to.
    time : array
        Time stamps.
    values : array
        Data values to write.
    sigma : array
        Standard deviation uncertainty of data values.
    mask : array, optional
        Masked values boolean array, by default None.
    title : str, optional
        Title for header, by default CONFIG.title
    col_info : 
        Column information.
    meta : dict, optional
        Meta data for header, by default None
    target : str, optional
        Name of the target, by default '<unknown>'
    precision : int
        Numberical precision to use when formatting the data.
    series_type : str
        The type of series that the data represents. This information is written
        to the document header.

    """

    if meta is None:
        meta = {}

    # # get the masked values as separate array for saving as column values
    # values, sigma, mask = split_mask(values, sigma, mask)
    _, nseries = values.shape

    # logger.info('Saving sequence data ({} rows, {} series, containing {} '
    #             'masked points{}) to file: {}',
    #             len(time), nrows, nseries, (0 if mask is None else mask.sum()),
    #             ', including meta data' if meta else '', filename)

    # stack data
    data = stack_arrays(time, values, sigma, mask)

    nrows, ncols = data.shape
    shape_info = dict(nrows=nrows, ncols=ncols, nseries=nseries)
    has_oflag = mask is not None
    header, col_fmt_data = make_header(title.format(target), target,
                                       shape_info, col_info, has_oflag, meta,
                                       precision, series_type)

    # write to file
    with Path(filename).open('w') as fp:
        fp.write(header)
        np.savetxt(fp, data, col_fmt_data)


# alias
write_text = write


def read(filename, *_, **__):
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

    header = read_lines(filename, 25)
    data = np.loadtxt(filename)

    oflag = op.index(header, f'# {CONFIG.columns.outlier[0]}',
                     test=str.startswith, default=None)

    return unstack_arrays(data, oflag)


# alias
read_text = read
