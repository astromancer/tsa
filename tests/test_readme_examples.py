# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=invalid-name
# pylint: disable=unused-import


# std
import re
import itertools as itt
from pathlib import Path


# third-party
import pytest
import numpy as np
import matplotlib.pyplot as plt

# local
from tsa.ts import TimeSeries
from tsa.spectral import Periodogram
from tsa.ts.generate import Harmonic


# ---------------------------------------------------------------------------- #
# project folder
SRC = Path(__file__).parent.parent

RGX_EXAMPLE = re.compile(r'(?s)\n\s*```python\s+(.+?)```')

# ---------------------------------------------------------------------------- #


def get_readme_examples(filename=SRC / 'README.md'):
    # sourcery skip: hoist-statement-from-loop
    readme = Path(filename).read_text()
    for match in RGX_EXAMPLE.finditer(readme):
        yield match[1]


class get_name:
    count = itt.count()

    def __call__(self, val):
        return next(self.count)

# def get_figure(namespace):
#     for obj in reversed(namespace.values()):
#         if isinstance(obj, (TimeSeries, Periodogram)):
#             return obj.fig


@pytest.fixture(params=itt.islice(get_readme_examples(), 3),
                ids=get_name())
def readme_code(request):
    return request.param


@pytest.mark.mpl_image_compare(baseline_dir='images', style='default')
def test_readme_example(readme_code):
    locals_ = {}
    code = (readme_code + '\n'
            'fig = plt.gcf()')
    exec(code, None, locals_)
    globals().update(locals_)
    return locals_['fig']
