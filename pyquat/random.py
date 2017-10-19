"""
This file contains functions for the generation of random
quaternions using various statistical distributions (namely
uniform and gaussian).

It is also useful simply for generating random axes.
"""

import pyquat as pq

from math import sqrt, cos, sin, pi
import numpy as np

def randu(symmetric_range = 1.0):
    """
    Generate a uniform random number with mean 0 and range
    (-symmetric_range, +symmetric_range).  Default argument (1.0)
    gives mean 0 and range (-1, 1).
    """
    
    return (np.random.rand() * 2.0 - 1.0) * symmetric_range

def uniform_random_axis(max_theta = 2.0 * pi, z_range = 1.0):
    """
    Generate a unit random axis from a uniform distribution.
    """
    if max_theta == 0.0:
        theta = 0.0
    else:
        theta = np.random.rand() * max_theta

    axis = np.zeros((3,1))
    axis[2] = randu(z_range)
    axis[0] = sqrt(1.0 - axis[2]**2) * cos(theta)
    axis[1] = sqrt(1.0 - axis[2]**2) * sin(theta)
    return axis


def rand(
        axis = None,
        angle = None,
        axis_generator = uniform_random_axis,
        angle_generator = randu,
        **axis_generator_kwargs):
    """
    Generate a random quaternion. With all defaults, the quaternion
    will be both random angle and random axis.

    Allows each of axis and angle to be generated or provided; and a
    generator can be provided as well. The default is to use a uniform
    distribution for both axis and angle.

    Any extra arguments to this function are passed on to the axis
    generator.

    """
    if axis is not None and angle is not None:
        raise StandardError("expected non-fixed angle or non-fixed axis or both")
    if axis is None:
        axis  = axis_generator(**axis_generator_kwargs)
    if angle is None:
        angle = angle_generator()
    return pq.Quat.from_angle_axis(angle, *axis)


