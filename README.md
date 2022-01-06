# pyquat

pyquat is a Python C extension providing quaternions and functions relating to simulation of attitudes and rotations.

[![Build Status](https://travis-ci.org/mohawkjohn/pyquat.svg?branch=master)](https://travis-ci.org/mohawkjohn/pyquat)

## Installation

    python setup.py build
    python setup.py install

## Usage

    from pyquat import Quat
    import pyquat

    q = pyquat.identity()

    q4 = Quat(0.73029674334022143, 0.54772255750516607, 0.36514837167011072, 0.18257418583505536)
    q2 = q * q4.conjugated()
    q2.normalize()
    q2.conjugate()

    transform = q2.to_matrix()

    vec = q2.to_rotation_vector()

## License

Copyright 2016--2022 Juno Woods, Ph.D., and Translunar LLC.

See LICENSE.txt.