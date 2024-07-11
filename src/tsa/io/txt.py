"""
Write time series / frequency spectra to plain text in utf-8, emphasis on human
readable forms.
"""

# std
import re
from pathlib import Path

# third-party
import numpy as np
import more_itertools as mit

# local
from recipes import api, op
from recipes.io import read_lines
from recipes.iter import cofilter
from recipes.config import ConfigNode
from recipes.functionals import not_none
from recipes.containers.utils import split
from recipes.pprint.mapping import pformat
from recipes.string import hstack, remove_prefix
from recipes.string.unicode import vertical_brace

# relative
from .utils import split_mask, stack_arrays, unstack_arrays


# ---------------------------------------------------------------------------- #
CONFIG = ConfigNode.load_module(__file__, dot_split=True)

# write oflag data to file
REGEX_FORMAT_SPEC = re.compile(r'%?(\d{0,2})\.?(\d{0,2})([if])')

COLUMN_INFO = 'Column Info'
UNIT_FORMAT = '[{}]'


# U+23aa Sm CURLY BRACKET EXTENSION ⎪
# U+23ab Sm RIGHT CURLY BRACKET UPPER HOOK ⎫
# U+23ac Sm RIGHT CURLY BRACKET MIDDLE PIECE ⎬
# U+23ad Sm RIGHT CURLY BRACKET LOWER HOOK ⎭
# ---------------------------------------------------------------------------- #


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

    flags = [bool(op.index(header, f'# {name}', test=str.startswith, default=None))
             for name in ('index', 'sigma', 'mask')]
    
    return unstack_arrays(data, *flags)


# alias
read_text = read


@api.synonyms(values='value')
def write(filename, index, values, sigma,
          title=CONFIG.title, col_info=CONFIG.columns,
          target='', series_type='series', **meta):
    """
    Write to text file.

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
    title : str, optional
        Title for header, by default CONFIG.title
    col_info : 
        Column information.
    target : str, optional
        Name of the target, by default None
    series_type : str
        The type of series that the data represents. This information is written
        to the document header.
    **meta
        Meta data for header, by default None
    """

    if meta is None:
        meta = {}

    # # get the masked values as separate array for saving as column values
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
    # has_oflag = mask is not None

    # avail = ('index', 'value', 'sigma', 'mask')
    _, col_info = cofilter(not_none, list(map(eval, col_info)), col_info.items())
    col_info = dict(col_info)

    header, col_fmt_data = make_header(title.format(target), target,
                                       shape_info, col_info, meta, series_type)

    # write to file
    with Path(filename).open('w') as fp:
        fp.write(header)
        np.savetxt(fp, data, col_fmt_data)


# alias
write_text = write


def make_header(title, target_name, shape_info, col_info, meta=None,
                series_type='series'):
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
        Information about the data contained available for each sequence. This
        is a dict with up to 4 items, keyed on the column data type (index,
        value, sigma, mask). The dict values are dicts with 'title',
        'description', 'fmt' and 'unit' entries.
    meta : dict, optional
        Meta data for the header, by default None.
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
    n_series = shape_info['n_series']
    col_headers, col_info = get_column_info(n_series, col_info, series_type)

    # adjust the formatters
    names, units, formats = zip(*col_headers)
    widths, fmt_head, fmt_data = make_column_format(names, units, formats)

    info = {
        #  title, table shape info
        f'# {title}': shape_info,
        # column descriptions
        COLUMN_INFO: col_info,
        **meta
    }

    # column headers block
    lines = _gen_header_lines(info, target_name, n_series,
                              (names, units, widths, fmt_head))
    return '\n'.join(lines).replace('\n', '\n# ')[:-2], fmt_data


def get_column_info(n_series, col_info, series_type='series'):
    """
    Get column information.

    Parameters
    ----------
    n_series : int
        Number of series.
    col_info : dict
        Column information.
    series_type : str, optional
        Type of series, by default 'series'.

    Returns
    -------
    tuple
        Names, units, formats, and info for the columns.
    """

    col_info, descriptions = ConfigNode(col_info).split('description')

    # column descriptions
    descriptions = descriptions.find('description', collapse=True)
    offset = int('index' in descriptions)
    info_text = hstack(('\n'.join(descriptions.values()),
                       vertical_brace(len(descriptions) - offset,
                                      f'x{n_series} {series_type}')),
                       spacing=3, offsets=offset, rstrip=True)
    descriptions.update(zip(descriptions.keys(), info_text.splitlines()))

    # build column headers
    headers = [
        (info['title'],
         UNIT_FORMAT.format(u) if (u := info['unit']) else '',
         '%{}'.format(remove_prefix(info['fmt'], '%')))
        for info in col_info.values()
    ]

    base, per_series = split(headers, ['index' in col_info])
    headers = [*base, *(per_series * n_series)]

    return headers, descriptions


def make_column_format(names, units, formats):
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
    widths, precisions, dtypes = check_column_widths(names, units, formats)
    col_fmt_head = ''.join(map('%%-%is'.__mod__, [widths[0] - 2, *widths[1:]]))
    col_fmt_data = ''.join(map('%%-%i.%s%s'.__mod__,
                               zip(widths, precisions, dtypes)))
    return widths, col_fmt_head, col_fmt_data


def check_column_widths(names, units, formats):
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
    return tuple(zip(*_check_column_widths(names, units, formats)))


def _check_column_widths(names, units, formats):
    for name, unit, fmt in zip(names, units, formats):
        width, precision, dtype = parse_format_spec(fmt)
        width = max(int(width or 1), len(name) + 1, len(unit) + 1)
        yield width, precision, dtype


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


def _gen_header_lines(header_info, target_name, n_series, col_spec):

    *col_names_units, col_widths, col_fmt_head = col_spec

    # header blocks for additional meta data
    yield from map(header_info_block, *zip(*header_info.items()))
    # section divider
    yield (hline := '-' * (sum(col_widths) - 2))

    # object names
    n_col_per_series = len(set(header_info[COLUMN_INFO].keys()) - {'index'})
    target_names = ('', target_name, *(f'C{i}' for i in range(n_series - 1)))
    yield _make_name_format(col_widths, n_col_per_series) % target_names

    # column titles
    for o in col_names_units:
        yield col_fmt_head % tuple(o)

    # section divider
    yield hline
    yield ''  # advance to new line


def _make_name_format(col_widths, n_col_per_series):
    w0, *ww = col_widths
    w2 = map(sum, mit.grouper(ww, n_col_per_series, fillvalue=ww))  # 2-column widths
    return ''.join('%%-%is' % w for w in [w0 - 2, *w2])


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
