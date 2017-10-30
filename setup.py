#!/usr/bin/env python

from setuptools import setup, find_packages
from setuptools.extension import Extension
import os
import numpy

MAJOR = 0
MINOR = 2
TINY  = 3
version='%d.%d.%d' % (MAJOR, MINOR, TINY)

c_quat = Extension('pyquat/_pyquat',
                   ['pyquat/pyquat.c'],
                   include_dirs  = [numpy.get_include()],
                   define_macros = [('MAJOR_VERSION', MAJOR),
                                    ('MINOR_VERSION', MINOR),
                                    ('TINY_VERSION',  TINY)])
setup(name='pyquat',
      version=version,
      description='Python C quaternion type with attitude utility functions',
      author='John O. Woods, Ph.D.',
      author_email='john.woods@intuitivemachines.com',
      url='http://www.intuitivemachines.com',
      include_package_data=True,
      packages=['pyquat', 'test'],
      ext_modules=[c_quat],
      test_suite='test.pyquat_test_suite'
    )
