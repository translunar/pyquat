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

def plot_frame(q, r = numpy.zeros((3,1)), fig = None, axes = None, axis_size = 1.0):
    """
    Plot a quaternion as a coordinate frame, with red, green, and blue 
    referring to x, y, and z.

    Also accepts an optional position to move the frame to (post-rotation).
    """
    if fig is None:
        fig = matplotlib.pyplot.figure()
    if axes is None:
        axes = fig.add_subplot(111, projection='3d')

    xh = numpy.zeros((3,2))
    yh = numpy.zeros_like(xh)
    zh = numpy.zeros_like(xh)
    xh[0,1] = axis_size
    yh[1,1] = axis_size
    zh[2,1] = axis_size

    # Transform the frame (rotate it and then move it)
    T = q.to_matrix()
    txh = np.dot(T, xh) + r
    tyh = np.dot(T, yh) + r
    tzh = np.dot(T, zh) + r
    
    axes.plot(txh[0,:], txh[1,:], txh[2,:], c='r')
    axes.plot(tyh[0,:], tyh[1,:], tyh[2,:], c='g')
    axes.plot(tzh[0,:], tzh[1,:], tzh[2,:], c='b')
    
    return axes
