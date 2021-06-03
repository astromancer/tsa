# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name


# std libs
import re
import itertools as itt
from pathlib import Path
import matplotlib.pylab as plt

# third-party libs
import pytest

import numpy as np

from tsa import TimeSeries
from tsa.spectral import Periodogram
from tsa.ts.generate import Harmonic

RGX_EXAMPLE = re.compile(r'(?s)\n\s*```python\s+(.+?)```')


def get_readme_examples(filename='../README.md'):
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


@pytest.mark.mpl_image_compare(baseline_dir='images')
def test_readme_example(readme_code):
    locals_ = {}
    code = f'{readme_code}\nfig = plt.gcf()'
    exec(code, None, locals_)
    globals().update(locals_)
    return locals_['fig']
