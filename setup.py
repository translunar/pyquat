#!/usr/bin/env python

from distutils.core import setup, Extension
import numpy

setup(name='pyquat',
      version='0.0.1',
      description='Python C quaternion type',
      author='John O. Woods, Ph.D.',
      author_email='john.woods@intuitivemachines.com',
      url='http://www.intuitivemachines.com',
      ext_modules=[
          Extension('pyquat', ['pyquat.c'], include_dirs=[numpy.get_include()])
        ]
    )
