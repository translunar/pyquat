from pyquat._pyquat import *

import numpy
from scipy import linalg

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

