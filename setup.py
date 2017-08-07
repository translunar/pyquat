#!/usr/bin/env python

from setuptools import setup, find_packages
from setuptools.extension import Extension
import os
import numpy

version='0.0.5'
c_ext = Extension('pyquat/_pyquat', ['pyquat/pyquat.c'], include_dirs=[numpy.get_include()])

setup(name='pyquat',
      version=version,
      description='Python C quaternion type',
      author='John O. Woods, Ph.D.',
      author_email='john.woods@intuitivemachines.com',
      url='http://www.intuitivemachines.com',
      include_package_data=True,
      packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
      ext_modules=[c_ext]
    )
