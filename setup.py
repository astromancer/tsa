
import site
import sys
import setuptools

# see: https://github.com/pypa/pip/issues/7953
site.ENABLE_USER_SITE = ('--user' in sys.argv[1:])
setuptools.setup()