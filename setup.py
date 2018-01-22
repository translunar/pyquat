#!/usr/bin/env python
""" Python C quaternion type for fast attitude and rotation computations.
See:
https://github.com/mohawkjohn/pyquat
"""

from setuptools import setup, find_packages
from setuptools.extension import Extension
import os
import numpy

MAJOR = 0
MINOR = 3
TINY  = 2
version='%d.%d.%d' % (MAJOR, MINOR, TINY)

c_quat = Extension('_pyquat',
                   ['pyquat/pyquat.c'],
                   extra_compile_args = ["-std=c99"],
                   include_dirs  = [numpy.get_include()],
                   define_macros = [('MAJOR_VERSION', MAJOR),
                                    ('MINOR_VERSION', MINOR),
                                    ('TINY_VERSION',  TINY)])
setup(name='pyquat',
      version=version,
      description='Python C quaternion type for fast attitude and rotation computations',
      author='John O. Woods, Ph.D.',
      author_email='john.woods@intuitivemachines.com',
      url='http://github.com/mohawkjohn/pyquat/',
      download_url='https://github.com/mohawkjohn/pyquat/archive/v' + version + '.tar.gz',
      include_package_data=True,
      packages=['pyquat', 'pyquat.wahba'],
      ext_modules=[c_quat],
      test_suite='test.pyquat_test_suite',
      license='BSD 3-clause',
      classifiers=[ # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
          # How mature is this project? Common values are
          #   3 - Alpha
          #   4 - Beta
          #   5 - Production/Stable
          'Development Status :: 3 - Alpha',

          # Indicate who your project is intended for
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Topic :: Software Development :: Embedded Systems',
          'Topic :: Software Development :: Libraries :: Python Modules',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Scientific/Engineering :: Physics',
          'Topic :: Scientific/Engineering :: Visualization',


          # Pick your license as you wish (should match "license" above)
          'License :: OSI Approved :: BSD License',

          # Specify the Python versions you support here. In particular, ensure
          # that you indicate whether you support Python 2, Python 3 or both.

          'Programming Language :: C',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: Implementation :: CPython'
      ],
      keywords=['quaternion', 'math', 'maths', 'graphics', 'physics', 'orientation', 'attitude', 'pose', 'geometry', 'visualization', 'visualisation', 'animation', 'game development', 'simulation'],
      install_requires=['numpy']
    )
