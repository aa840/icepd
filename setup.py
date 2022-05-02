from distutils.core import setup
from setuptools import setup, find_packages
from io import open  # pylint:disable=redefined-builtin

setup(name='icepd',
      author='Alice E. A. Allen',
      version='0.0',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      )
