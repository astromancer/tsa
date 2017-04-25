#!/usr/bin/env python3

import pickle
import argparse
import itertools as itt
from sys import argv
from pathlib import Path

from setuptools import setup, find_packages


def count_bad(ignored_files):
    def count(itr):
        i = 0
        for _ in itr:
            i += 1
        return i

    def splitter(fn):
        parts = fn.replace('.mat', '').split('_')
        if len(parts) == 3:
            return parts[0], parts[2]
        else:
            return parts[0]

    totals = []
    for grpId, members in itt.groupby(ignored_files, splitter):
        totals.append(count(members))

    return totals


def write_bad_count(counts, filename):
    with open(str(filename), 'wb') as fp:
        pickle.dump(counts, fp)

def check_data_dir(d):
    path = Path(d).resolve()
    assert path.exists(), 'Not a valid directory'
    return path

# ensure data dir supplied
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('action')
parser.add_argument('--data-dir', required=True,
                    type=check_data_dir,
                    help='Specifies the location of the data')
args = parser.parse_args()
argv.pop(-1)        #avaid setuptools error: option --data-dir not recognized


# make data file of bad counts
here = Path(__file__).resolve().parent
ignore_file = here / 'dat/ignore.txt'
bad_count_file =  here / 'dat/bad_count.dat'
with ignore_file.open('r') as fp:
    ignored_files = fp.readlines()
bc = count_bad(ignored_files)
write_bad_count(bc, bad_count_file)



# Get the long description from the README file
# with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
#     long_description = f.read()


# NAME = 'eeg'
setup(
    name='eeg',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='1.0.0.dev1',

    description='Code for (attempting) epileptic seizure prediction from multi-channel EEG',
    # long_description=long_description,

    # The project's main homepage.
    url='https://gitlab.com/seizures_2016/eeg',

    # Author details
    author='Hannes Breytenbach',
    author_email='hannes@saao.ac.za',

    # Choose your license
    # license='None yet',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Kagglers',
        # 'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish (should match "license" above)
        # 'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        # 'Programming Language :: Python :: 2',
        # 'Programming Language :: Python :: 2.6',
        # 'Programming Language :: Python :: 2.7',
        # 'Programming Language :: Python :: 3',
        # 'Programming Language :: Python :: 3.3',
        # 'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

    # What does your project relate to?
    keywords=['EEG', 'electroencephalography', 'seizure prediction', 'machine learning'],

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['numpy', 'scipy', 'scikit-learn'],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    # extras_require={
    #     'dev': ['check-manifest'],
    #     'test': ['coverage'],
    # },

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    package_data={
        'eeg': ['dat/ignore.dat', 'dat/bad_count.dat'],
    },

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    # data_files=[('data_files', ['data/data_file']),],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    # entry_points={
    #     'console_scripts': [
    #         'sample=sample:main',
    #     ],
    # },
    # entry_points={
    #     "distutils.commands": [
    #         "foo = mypackage.some_module:foo",
    #     ],
    #     "distutils.setup_keywords": [
    #         "data_dir = mypackage.some_module:validate_bar_baz",
    #     ],
    # },
)
