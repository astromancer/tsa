"""
Universal build script for python project git repos.
"""

# std
import os
import re
import sys
import glob
import site
import fnmatch
import subprocess as sub
from pathlib import Path
from distutils import debug

# third-party
from setuptools.command.build_py import build_py
from setuptools import Command, setup, find_packages


debug.DEBUG = True

# allow editable user installs
# see: https://github.com/pypa/pip/issues/7953
site.ENABLE_USER_SITE = ('--user' in sys.argv[1:])

# check if we are in a repo
status = sub.getoutput('git status --porcelain')
untracked = re.findall(r'\?\? (.+)', status)


class GitIgnore:
    """exclude gitignored files from build archive"""

    def __init__(self, path='.gitignore'):
        self.names = self.patterns = ()
        path = Path(path)
        if not path.exists():
            return

        # read .gitignore patterns
        lines = path.read_text().splitlines()
        lines = (_.strip().rstrip('/') for _ in lines if not _.startswith('#'))
        items = names, patterns = [], []
        for line in filter(None, lines):
            items[glob.has_magic(line)].append(line)

        self.names = tuple(names)
        self.patterns = tuple(patterns)

    def match(self, filename):
        for pattern in self.patterns:
            if fnmatch.fnmatchcase(filename, pattern):
                return True
        return filename.endswith(self.names)


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
            if path in untracked:
                self.debug_print(f'ignoring untracked: {path}')
                continue

            if gitignore.match(path):
                self.debug_print(f'(git)ignoring: {path}')
                continue
            
            self.debug_print(f'FOUND: {package=}: {module=} {path=}')
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


gitignore = GitIgnore()

setup(
    packages=find_packages(exclude=['tests', "tests.*"]),
    use_scm_version=True,
    include_package_data=True,
    exclude_package_data={'': [*gitignore.patterns, *gitignore.names]},
    cmdclass={'build_py': Builder,
              'clean': CleanCommand}
    # extras_require = dict(reST = ["docutils>=0.3", "reSTedit"])
    # test_suite='pytest',
)
