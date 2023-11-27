# !/usr/bin/env python

"""
Universal build script for python project git repos.
"""

# std
import os
import re
import sys
import glob
import site
import math
import fnmatch
import subprocess as sub
from pathlib import Path
from distutils import debug

# third-party
from setuptools.command.build_py import build_py
from setuptools import Command, find_packages, setup


# ---------------------------------------------------------------------------- #
debug.DEBUG = True

# allow editable user installs
# see: https://github.com/pypa/pip/issues/7953
site.ENABLE_USER_SITE = ('--user' in sys.argv[1:])


# Git ignore
# ---------------------------------------------------------------------------- #
# Source: https://github.com/astromancer/recipes/blob/main/src/recipes/io/gitignore.py


def _git_status(raises=False):
    # check if we are in a repo
    status = sub.getoutput('git status --porcelain')
    if raises and status.startswith('fatal: not a git repository'):
        raise RuntimeError(status)

    return status


UNTRACKED = re.findall(r'\?\? (.+)', _git_status())
IGNORE_IMPLICIT = ('.git', )


class GitIgnore:
    """
    Class to read `.gitignore` patterns and filter source trees.
    """

    __slots__ = ('root', 'names', 'patterns')

    def __init__(self, path='.gitignore'):
        self.names = self.patterns = ()
        path = Path(path)
        self.root = path.parent

        if not path.exists():
            return

        # read .gitignore patterns
        lines = (line.strip(' /')
                 for line in path.read_text().splitlines()
                 if not line.startswith('#'))

        items = names, patterns = [], []
        for line in filter(None, lines):
            items[glob.has_magic(line)].append(line)

        self.names = (*IGNORE_IMPLICIT, *names)
        self.patterns = tuple(patterns)

    def match(self, filename):
        path = Path(filename).relative_to(self.root)
        filename = str(path)
        for pattern in self.patterns:
            if fnmatch.fnmatchcase(filename, pattern):
                return True

        return filename.endswith(self.names)

    def iter(self, folder=None, depth=any, _level=0):
        depth = math.inf if depth is any else depth
        folder = folder or self.root

        _level += 1
        if _level > depth:
            return

        for path in folder.iterdir():
            if self.match(path):
                continue

            if path.is_dir():
                yield from self.iter(path, depth, _level)
                continue

            yield path

    def match(self, filename):
        path = Path(filename)
        rpath = path.relative_to(self.root)
        filename = str(rpath)
        for pattern in self.patterns:
            if fnmatch.fnmatchcase(filename, pattern):
                return True

        return filename.endswith(self.names)


# Setuptools
# ---------------------------------------------------------------------------- #

class Builder(build_py):
    # need this to exclude ignored files from the build archive

    def find_package_modules(self, package, package_dir):
        # filter folders
        if gitignore.match(package_dir) or gitignore.match(Path(package_dir).name):
            self.debug_print(f'(git)ignoring {package_dir}')
            return

        # package, module, files
        info = super().find_package_modules(package, package_dir)

        for package, module, path in info:
            # filter files
            if path in UNTRACKED:
                self.debug_print(f'Ignoring untracked: {path}.')
                continue

            if gitignore.match(path):
                self.debug_print(f'(git)ignoring: {path}.')
                continue

            self.debug_print(f'Found: {package = }: {module = } {path = }')
            yield package, module, path


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./src/*.egg-info')


# Main
# ---------------------------------------------------------------------------- #
gitignore = GitIgnore()

setup(
    packages=find_packages(exclude=['tests', "tests.*"]),
    use_scm_version=True,
    include_package_data=True,
    exclude_package_data={'': [*gitignore.patterns, *gitignore.names]},
    cmdclass={'build_py': Builder,
              'clean': CleanCommand}
    # extras_require = dict(reST = ["docutils> = 0.3", "reSTedit"])
    # test_suite = 'pytest',
)
