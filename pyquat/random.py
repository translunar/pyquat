"""
This file contains functions for the generation of random
quaternions using various statistical distributions (namely
uniform and gaussian).

It is also useful simply for generating random axes.
"""

from typing import Any, Optional, Callable, TypedDict, TYPE_CHECKING

if TYPE_CHECKING:
    from pyquat._pyquat import Quat
else:
    from _pyquat import Quat

from math import sqrt, cos, sin, pi
import numpy as np

class AxisGeneratorKwargs(TypedDict, total=False):
    max_theta: float
    z_range: float

class AngleGeneratorKwargs(TypedDict, total=False):
    symmetric_range: float

AxisGeneratorType = Callable[..., np.ndarray[Any, np.dtype[np.float64]]]
AngleGeneratorType = Callable[..., float]

def randu(symmetric_range: float = 1.0) -> float:
    """
    Generate a uniform random number with mean 0 and range
    (-symmetric_range, +symmetric_range).  Default argument (1.0)
    gives mean 0 and range (-1, 1).
    """
    
    return float(np.random.rand() * 2.0 - 1.0) * symmetric_range

def uniform_random_axis(
        max_theta: float = 2.0 * pi,
        z_range: float = 1.0
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
    """
    Generate a unit random axis from a uniform distribution.
    """
    if max_theta == 0.0:
        theta = 0.0
    else:
        theta = float(np.random.rand()) * max_theta

    axis = np.zeros((3,1))
    axis[2] = randu(z_range)
    axis[0] = sqrt(1.0 - axis[2]**2) * cos(theta)
    axis[1] = sqrt(1.0 - axis[2]**2) * sin(theta)
    return axis


def rand(
        axis: Optional[np.ndarray[Any, np.dtype[np.float64]]] = None,
        angle: Optional[float] = None,
        axis_generator: AxisGeneratorType = uniform_random_axis,
        angle_generator: AngleGeneratorType = randu,
        axis_generator_kwargs: AxisGeneratorKwargs = {},
        angle_generator_kwargs: AngleGeneratorKwargs = {},
    ) -> Quat:
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
        raise ValueError("expected non-fixed angle or non-fixed axis or both")
    if axis is None:
        axis  = axis_generator(**axis_generator_kwargs)
    if angle is None:
        angle = angle_generator(**angle_generator_kwargs)
    return Quat.from_angle_axis(angle, *axis)


