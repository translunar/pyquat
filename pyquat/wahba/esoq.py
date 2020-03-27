"""
This file contains quaternion and attitude utility functions
relating to the ESOQ2 solution to what is known as Wahba's problem
[0], namely the problem of finding a rotation matrix which brings one
set of vectors into the best possible alignment with a second set of
vectors.

Reference:
[0] Wahba, Grace. 1965. Problems and Solutions: Problem 65--1.
    SIAM Review 7(3): 409.

"""

import numpy as np
from scipy import linalg
from math import sqrt, pi, cos, acos

def trace_adj(a):
    """Returns the trace of the adjunct of matrix a, which must be 3x3 or 4x4."""
    if max(a.shape) == 3:
        c1 = a[1,1] * a[2,2] - a[1,2] * a[2,1]
        c2 = a[0,0] * a[2,2] - a[0,2] * a[2,0]
        c3 = a[0,0] * a[1,1] - a[0,1] * a[1,0]
        return c1 + c2 + c3
    else:
        return minor_det_4x4(a, 0, 1, 2) + minor_det_4x4(a, 0, 2, 3) + minor_det_4x4(a, 1, 2, 3) + minor_det_4x4(a, 0, 1, 3)

def minor_det_4x4(a, i = 0, j = 1, k = 2):
    """Returns the determinant of a minor of a 4x4 matrix.

    Args:
        a      matrix whose minor we're interested in
        i,j,k  rows/columns to include in the determinant calculation (e.g.,
               if 0, 1, and 2 are supplied, the determinant will be of the
               upper left-hand submatrix)

    Returns:
        A Float giving the determinant of the indicated 3x3 submatrix.
        
    """
    return a[i,i] * (a[j,j]*a[k,k] - a[j,k]*a[k,j]) - a[i,j] * (a[j,i]*a[k,k] - a[j,k]*a[k,i]) + a[i,k] * (a[j,i]*a[k,j] - a[j,j]*a[k,i])

def trace_adj_symm(a):
    """Returns the trace of the adjunct of a symmetric matrix a, which
    must be 3x3 or 4x4."""
    if max(a.shape) == 3:
        c1 = a[1,1] * a[2,2] - a[1,2]**2
        c2 = a[0,0] * a[2,2] - a[0,2]**2
        c3 = a[0,0] * a[1,1] - a[0,1]**2
        return c1 + c2 + c3
    else:
        return trace_adj(a)


def lambda_nr_iterate(lambda_c, b, c, d):
    l2 = lambda_c**2
    l3 = l2*lambda_c
    l4 = l2**2
    return lambda_c - (l4 + b*l2 + c*lambda_c + d) / (4.0*l3 + 2.0*b*lambda_c + c)


def sequential_rotation(B = None, q = None, irot = None):
    """"It is noted without proof that a rotation through an angle
    greater than \frac{\pi}{2} can be expressed as a rotation
    through \pi about one of the coordinate axes followed by a
    rotation about a new axis through an angle less than 
    \frac{\pi}{2}. An initial rotation through \pi about one of
    the coordinate axes is equivalent to changing the signs of
    two components of each of the reference vectors. The 
    quaternion p=(p_1,p_2,p_3,p_4)^\T of the optimal rotation
    transforming the new reference vectors \hat{V}_i', i=1,...,n
    into the observation vectors \hat{W}_i, i=1,...,n as
    calculated...is related very simply to the desired optimal
    quaternion. The results for the three possible cases are

    1. Initial rotation through \pi about the x axis:
        \hat{V}_i' = (\hat{V}_{ix}, -\hat{V}_{iy}, -hat{V}_iz)
        q          = (p_4, -p_3, p_2, -p_1)^\T
    2. Initial rotation through \pi about the y axis:
        \hat{V}_i' = (-\hat{V}_{ix}, \hat{V}_{iy}, -hat{V}_iz)
        q          = (p_3, p_4, -p_1, -p_2)^\T
    3. Initial rotation through \pi about the z axis:
        \hat{V}_i' = (-\hat{V}_{ix}, -\hat{V}_{iy}, hat{V}_iz)
        q          = (-p_2, p_1, p_4, -p_3)^\T
   
    Clearly, that initial rotation (including no initial rotation)
    will yield the most accurate estimate of q_\textrm{opt} for
    which |\gamma| achieves the largest value. In any practical
    application, however, |\gamma| can be allowed to become quite
    small before the method of sequential rotations need [sic.]
    to be invoked." [0]

    This method has three different calling modes:

    1. Supply irot and attitude profile matrix B. This performs the
       specified rotation on B and returns B.
    2. Supply irot and quaternion q. This performs the specified
       rotation on q and returns q.
    3. Supply only an attitude profile matrix B. Determines the
       rotation to perform, performs the rotation, and returns
       irot.

    This method produces an *in-place* sequential rotation of the
    attitude profile matrix B and returns the rotation performed. It
    is indifferent to the angle of rotation.

    Note: pyquat uses a different quaternion ordering than Shuster 
    & Oh.

    Reference:
    [0] Shuster, Oh. 1981. Three-axis attitude determination from 
        vector observations. Journal of Guidance and Control 4(1): 
        70-77.

    """
    import pyquat as pq
    
    if irot is None:
        diag_B  = np.diag(B)
        irot    = np.argmin(diag_B)
        sequential_rotation(B, irot = irot)
        return irot
    else:
        if B is None:
            if irot == 0:
                return pq.Quat(-q.x, q.w, -q.z, q.y)
            elif irot == 1:
                return pq.Quat(-q.y, q.z, q.w, -q.x)
            elif irot == 2:
                return pq.Quat(-q.x, q.w, -q.z, q.y)
        else:
            if irot == 0:
                B[:,1:3] *= -1.0
            elif irot == 1:
                B[:,0] *= -1.0
                B[:,2] *= -1.0
            elif irot == 2:
                B[:,0:2] *= -1.0
            return B
        

def davenport_eigenvalues(K, B, n_obs = None):
    """
    Compute the eigenvalues of the Davenport matrix K.

    Assumes sequential rotation has already been applied.

    Take careful note that you should pass n_obs = 2 if you only have
    two observations making up your attitude profile matrix B.

    Reference:

    [0] Mortari (1997). ESOQ-2 single-point algorithm for fast
        optimal spacecraft attitude determination. Advances in the
        Astronautical Sciences 95: 817-826.

    Args:
        K      Davenport matrix (4x4)
        B      attitude profile matrix (3x3)
        n_obs  number of observations involved in computing B (optional
               unless you only have 2, in which case this needs to be
               provided)

    Returns:
        The sorted eigenvalues of the Davenport K matrix.

    """
    tr_adj_K = trace_adj(K)
    tr_B     = np.trace(B)
    S        = B+B.T
    tr_adj_S = trace_adj_symm(S)
    z        = K[1:4,0].reshape((3,1))

    # a never actually gets used, so we won't compute it, but note
    # here that this is what it ought to be.

    # a = np.trace(K)
    b = -2.0 * tr_B**2 + tr_adj_S - np.dot(z.T, z)[0,0]
    d = linalg.det(K)

    if n_obs == 2:
        g3 = sqrt( 2*sqrt(d) - b)
        g4 = sqrt(-2*sqrt(d) - b)        
        lambda_4 =  (g3 + g4) / 2.0
        lambda_1 = -lambda_4
        lambda_3 =  (g3 - g4) / 2.0
        lambda_2 = -lambda_3
    else:
        c = -tr_adj_K

        p  = (b/3.0)**2 + 4*d / 3.0
        q  = (b/3.0)**3 - 4*d*b / 3.0 + c**2 / 2.0

        u1 = 2 * sqrt(p) * cos(acos(q / p**1.5) / 3.0) + b / 3.0

        g1 = sqrt(u1 - b)
        g2 = 2.0 * sqrt(u1**2 - 4.0*d)

        lambda_1 = (-g1 - sqrt(-u1 - b + g2)) / 2.0
        lambda_2 = (-g1 + sqrt(-u1 - b + g2)) / 2.0
        lambda_3 =  (g1 - sqrt(-u1 - b - g2)) / 2.0
        lambda_4 =  (g1 + sqrt(-u1 - b - g2)) / 2.0

    return np.array([lambda_4, lambda_3, lambda_2, lambda_1])


def esoq2(K,
          B,
          lambda_0 = None,
          n_obs    = None):
    """
    Compute the optimal quaternion using Mortari's second estimator.
    
    References: 
    [0] Mortari (2000). Second estimator of the optimal quaternion. 
        Journal of Guidance, Control, and Dynamics 23(5): 885-8.
    [1] Mortari (1997). ESOQ: A closed-form solution to the Wahba
        problem. Journal of the Astronautical Sceinces 45(2): 195--204.
    [2] Mortari (1997). ESOQ-2 single-point algorithm for fast optimal
        spacecraft attitude determination. In Advances in the 
        Astronautical Sciences, 4--7 Aug, Sun Valley, Idaho, 
        pp. 817--826.

    Args:
        K          Davenport matrix (4x4)
        B          attitude profile matrix (3x3)
        lambda_0   optional initial guess about maximum eigenvalue (default
                   is to set this equal to n_obs)
        n_obs      optional number of observations involved in computing B

    Returns:
        A tuple of the optimal quaternion (corresponding to the maximum
    eigenvalue), which has been normalized; and the value of the loss
    function from Wahba's problem.

    """
    import pyquat as pq

    if lambda_0 is None: # Initial guess for maximum eigenvalue
        lambda_0 = float(n_obs)

    tr_B = np.trace(B)
    z    = K[1:4,0].reshape((3,1))
    S    = B+B.T

    lambdas = davenport_eigenvalues(K, B, n_obs = n_obs)
    lambda_max = lambdas[0]
    loss = lambda_0 - lambda_max

    trace_plus_lambda  = tr_B + lambda_max
    trace_minus_lambda = tr_B - lambda_max
    
    S_minus_tpl = S - np.identity(3) * trace_plus_lambda
    M = S_minus_tpl * trace_minus_lambda - np.dot(z, z.T)
    m1 = M[0,0:3]
    m2 = M[1,0:3]
    m3 = M[2,0:3]

    # Find the largest e_i.
    e_options = [np.cross(m2, m3), np.cross(m3, m1), np.cross(m1, m2)]
    e_mags    = np.array([linalg.norm(e_options[0]), linalg.norm(e_options[1]), linalg.norm(e_options[2])])
    imax      = np.argmax(e_mags)
    e         = e_options[imax].reshape((3,1))

    if n_obs != 2:
        # This piece borrowed from:
        # https://github.com/muzhig/ESOQ2/blob/master/esoq2p1.py.
        # Has not yet been tested!!!
        n1 = np.array([S_minus_tpl[0,0] - 2 * lambda_max, S_minus_tpl[0,1], S_minus_tpl[2,0]])
        n2 = np.array([S_minus_tpl[0,1], S_minus_tpl[1,1] - 2 * lambda_max, S_minus_tpl[1,2]])
        n3 = np.array([S_minus_tpl[2,0], S_minus_tpl[1,2], S_minus_tpl[2,2] - 2 * lambda_max])
        aa = [m2, m3, m1][imax]
        bb = [n3, n1, n2][imax]
        cc = [m3, m1, m2][imax]
        dd = [n2, n3, n1][imax]
        mm = [m1, m2, m3][imax].reshape((1,3))
        nn = [n1, n2, n3][imax].reshape((1,3))

        v = (np.cross(aa, bb) - np.cross(cc, dd)).reshape((3,1))
        loss = - (mm.dot(e)) / (nn.dot(e) + mm.dot(v))
        trace_minus_lambda  += loss
        e += loss * v
        

    # Find the normalized quaternion (unrotated)
    q = pq.Quat(np.dot(z.T, e),
                *(e * -trace_minus_lambda)).normalized_large()
        
    return q, loss
