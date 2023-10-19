# pyquat

pyquat is a Python C extension providing quaternions and functions relating to simulation of attitudes and rotations.
This might be useful for video games, spacecraft simulators, and sensors such as LIDARs and star trackers which provide
attitude information. You can use them in place of rotation matrices. Unit quaternions represent attitude as half of the
surface of a unit sphere in four dimensions (SO4).

[![Build Status](https://travis-ci.org/translunar/pyquat.svg?branch=master)](https://travis-ci.org/translunar/pyquat)

[![image](http://img.shields.io/pypi/v/pyquat.svg)](https://pypi.python.org/pypi/pyquat/)

## Developers

Note that since Travis discontinued free CI for open source projects, our CI pipeline
has stopped working. The correct sequence to run before initiating a pull request is:

    pip3 install -r requirements.txt
    python3 setup.py build_ext --inplace
    python3 setup.py test
    cd pyquat/
    mypy .  --config-file=mypy.ini --exclude=build --explicit-package-bases --strict

## Installation

    python3 setup.py build
    python3 setup.py install

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

Copyright 2016--2023 Juno Woods, Ph.D., and Translunar LLC.

See LICENSE.txt.
