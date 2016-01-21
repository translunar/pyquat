import numpy
import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D

def scatter(q_ary, **kwargs):
    """ Plot an array of quaternions as scatter points on a sphere """

    fig = kwargs.pop('fig',   matplotlib.pyplot.figure())
    axes = kwargs.pop('axes', fig.add_subplot(111, projection='3d'))

    # Create a vectorized function for converting all attitudes to points on
    # a sphere
    ary = None
    if len(q_ary.shape) == 1:
        ary = numpy.empty((3, q_ary.shape[0]))
        for i, q in enumerate(q_ary):
            ary[:,i] = q.to_unit_vector()[:,0]
    elif len(q_ary.shape) == 2:
        ary = numpy.empty((3, q_ary.shape[1]))
        for i, q in enumerate(q_ary):
            ary[:,i] = q[0].to_unit_vector()[:,0]
    else:
        raise InputError("expected 1- or 2-D array");

    return axes.scatter(ary[0,:], ary[1,:], ary[2,:], **kwargs)
