import numpy
import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D


def prepare_quaternion_array_for_plotting(q_ary, rotate_axis='x'):
    """ Converting all attitudes to points on a sphere """
    ary = None
    if len(q_ary.shape) == 1:
        ary = numpy.empty((3, q_ary.shape[0]))
        for i, q in enumerate(q_ary):
            ary[:,i] = q.to_unit_vector(rotate_axis)[:,0]
    elif len(q_ary.shape) == 2:
        ary = numpy.empty((3, q_ary.shape[1]))
        for i, q in enumerate(q_ary):
            ary[:,i] = q[0].to_unit_vector(rotate_axis)[:,0]
    else:
        raise InputError("expected 1- or 2-D array")
    return ary


def scatter(q_ary, rotate_axis='x', fig = None, axes = None, **kwargs):
    """ Plot an array of quaternions as scatter points on a sphere """

    if fig is None:
        fig = matplotlib.pyplot.figure()
    if axes is None:
        axes = fig.add_subplot(111, projection='3d')

    ary = prepare_quaternion_array_for_plotting(q_ary, rotate_axis = rotate_axis)

    return axes.scatter(ary[0,:], ary[1,:], ary[2,:], **kwargs)

def plot(q_ary, t = None, rotate_axis='x', fig = None, axes = None, **kwargs):
    """
    Plot an array of quaternions using lines on the surface of a sphere.
    
    Optionally, you can provide a t argument to plot as a timeseries,
    where the lowest t is displayed red, and the highest t is
    displayed blue.

    """

    if fig is None:
        fig = matplotlib.pyplot.figure()
    if axes is None:
        axes = fig.add_subplot(111, projection='3d')

    ary = prepare_quaternion_array_for_plotting(q_ary, rotate_axis = rotate_axis)

    if t is not None:
        tc = (t - t.min()) / (t.max() - t.min())
        c = numpy.vstack((tc, numpy.zeros_like(tc), -tc + 1.0))

        for i in range(1,ary.shape[1]):
            axes.plot(ary[0,i-1:i+1], ary[1,i-1:i+1], ary[2,i-1:i+1], c = tuple(c[:,i]), **kwargs)
    else:
        return axes.plot(ary[0,:], ary[1,:], ary[2,:], **kwargs)
