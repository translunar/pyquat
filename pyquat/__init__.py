from pyquat._pyquat import *

import numpy
from scipy import linalg

QUAT_SMALL = 1e-8

def change(q, w, dt):
    """
    Change a quaternion q by some angular velocity w over some small timestep dt.
    """

    # Find magnitude of angular velocity (in r/s)
    w_norm = linalg.norm(w)
    if w_norm < QUAT_SMALL:
        return q
    e = w / w_norm
    dq = Quat.from_angle_axis(w_norm * dt, *e)
    return dq * q

def cov(ary):
    """
    Compute the covariance of an array of quaternions, where each column represents a quaternion.
    """
    # If the user supplies an array of N quaternions, convert it to a 4xN array,
    # since we need it in this form to get its covariance.
    if ary.dtype == numpy.dtype(Quat):
        a = numpy.empty((4, max(ary.shape)), dtype=numpy.double)
        q_ary = ary.T
        for i, q in enumerate(q_ary.flatten()):
            a[:,i] = q.to_vector()[:,0]
        ary = a
            
    # Compute the covariance of the supplied quaternions.
    return numpy.cov(ary)

def mean(ary, covariance=None):
    """
    Compute the average quaternion using Markey, Cheng, Craissidis, and Oshman (2007)
    
    This method takes a 4xN array and computes the average using eigenvalue decomposition.
    """
    if covariance == None:
        covariance = cov(ary)

    # Compute their eigenvalues and eigenvectors
    eigenvalues, eigenvectors = linalg.eig(covariance)
    max_index = numpy.argmax(eigenvalues)
    q = eigenvectors[max_index]
    mean = Quat(q[0], q[1], q[2], q[3])
    mean.normalize()
    return mean

def mean_and_cov(ary):
    c = cov(ary)
    m = mean(ary, covariance=c)
    return (m,c)

def angle_vector_cov(ary):
    """
    Compute the covariance of an array of quaternions, like cov(), except use the attitude vector
    representation of each.
    """

    if ary.dtype == numpy.dtype(Quat):
        a = numpy.empty((3, max(ary.shape)), dtype=numpy.double)
        q_ary = ary.T
        for i, q in enumerate(q_ary.flatten()):
            a[:,i] = q.to_angle_vector()[:,0]
        ary = a
    elif ary.dtype == numpy.double:
        a = numpy.empty((3, ary.shape[1]), dtype=numpy.double)
        q_ary = ary.T
        for i, q in enumerate(q_ary):
            a[:,i] = Quat(q[0], q[1], q[2], q[3]).to_angle_vector()[:,0]
        ary = a

    return numpy.cov(ary)

def from_rotation_vector(v):
    """
    Shortcut for Quat.from_rotation_vector(v).
    """
    return Quat.from_rotation_vector(v)

def from_matrix(m):
    """
    Shortcut for Quat.from_matrix(v).
    """
    return Quat.from_matrix(m)

def skew(v):
    """
    Generate a skew-symmetric matrix from a vector.

    Code borrowed from: https://pythonpath.wordpress.com/2012/09/04/skew-with-numpy-operations/
    """
    #if len(v) == 4: v = v[:3]/v[3]
    skv = numpy.roll(numpy.roll(numpy.diag(v.flatten()), 1, 1), -1, 0)
    return skv - skv.T
