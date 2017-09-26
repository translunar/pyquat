#!/usr/bin/env python

from setuptools import setup, find_packages
from setuptools.extension import Extension
import os
import numpy

version='0.1.2'
c_ext = Extension('pyquat/_pyquat',
                  ['pyquat/pyquat.c'],
                  include_dirs  = [numpy.get_include()],
                  define_macros = [('MAJOR_VERSION', '0'),
                                   ('MINOR_VERSION', '1'),
                                   ('TINY_VERSION', '2')])
setup(name='pyquat',
      version=version,
      description='Python C quaternion type with attitude utility functions',
      author='John O. Woods, Ph.D.',
      author_email='john.woods@intuitivemachines.com',
      url='http://www.intuitivemachines.com',
      include_package_data=True,
      packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
      ext_modules=[c_ext]
    )
