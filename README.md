# pyquat

pyquat is an extremely simple Python C extension for unit quaternions.

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

    vec = q2.to_angle_vector()

## Copyright

Copyright 2016 John O. Woods, Ph.D., and Intuitive Machines

BSD 3-clause license