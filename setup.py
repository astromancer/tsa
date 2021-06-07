

# std libs
import os
import sys
import site
from pathlib import Path
from distutils import debug
from collections import defaultdict

# third-party libs
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist


from setuptools import setup, Command, find_packages


debug.DEBUG = True

# allow editable user installs
# see: https://github.com/pypa/pip/issues/7953
site.ENABLE_USER_SITE = ('--user' in sys.argv[1:])

# exclude ignored files from build archive
IGNORE = ['.gitignore',
          '.pylintrc',
          '.travis.yml']
GITIGNORE = defaultdict(list, {'': IGNORE})


gitignore = Path('.gitignore')
if gitignore.exists():
    # read .gitignore patterns
    lines = gitignore.read_text().splitlines()
    for line in filter(str.strip, lines):
        if not line.startswith('#'):
            *base, pattern = line.rsplit('/', 1)
            base = (base or [''])[0]
            # print(base, pattern)
            GITIGNORE[base].append(pattern)


# pprint(GITIGNORE)
# raise SystemExit


class builder(build_py):
    # need this to exclude ignored files from the build archive
    def find_package_modules(self, package, package_dir):
        # package, module, files
        *data, files = zip(*super().find_package_modules(package, package_dir))
        data = dict(zip(files, zip(*data)))

        ex = self.exclude_package_data
        if package_dir in ex:
            ex[package] = ex.pop(package_dir)

        keep = self.exclude_data_files(package, package_dir, files)
        return [(*data[file], file) for file in keep]


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./src/*.egg-info')


setup(
    packages=find_packages(exclude=['tests']),
    use_scm_version=True,
    include_package_data=True,
    exclude_package_data=GITIGNORE,
    cmdclass={'build_py': builder,
              'sdist': sdist,
              'clean': CleanCommand}
)
