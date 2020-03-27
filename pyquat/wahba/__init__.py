"""
This module contains quaternion and attitude utility functions
relating to solutions to what is known as Wahba's problem [0],
namely the problem of finding a rotation matrix which brings
one set of vectors into the best possible alignment with a
second set of vectors.

Reference:
[0] Wahba, Grace. 1965. Problems and Solutions: Problem 65--1.
    SIAM Review 7(3): 409.

"""

import numpy as np
import scipy.linalg as spl

def davenport_matrix(B                   = None,
                     covariance_analysis = False,
                     **attitude_profile_kwargs):
    """Compute the Davenport matrix for a given attitude profile.
    Accepts either an attitude profile matrix or the arguments
    for the attitude_profile_matrix() function. Returns a 4x4
    Davenport matrix K.

    If the covariance_analysis argument is True, a slightly modified K
    is returned, which may be used as part of a least squares approach
    (such as the qEKF or the information matrix approach, SOAR). See
    Eqs. 10--11 in [2] for details.

    If you do not provide an attitude profile matrix to this method,
    you must pass it the appropriate arguments for calling
    attitude_profile_matrix().

    References:

    [0] Davenport, P. 1968. A vector approach to the algebra of
        rotations with applications. NASA Technical Report X-546-65-437.

    [1] De Ruiter, A.; Damaren, C.; Forbes, J. 2013. Spacecraft
        Dynamics and Control: An Introduction, 1st Ed. Wiley, West
        Sussex, U.K. pp. 468-471.

    [2] Ainscough, Zanetti, Christian, Spanos. 2015. Q-method extended
        Kalman filter. Journal of Guidance, Control, and Dynamics
        38(4): 752--760.

    Kwargs:
        B                        attitude profile matrix (if not provided, one
                                 will be computed using attitude_profile_kwargs)
        covariance_analysis      defaults to False for the standard Davenport
                                 matrix; if True, returns the Davenport matrix
                                 utilized for least squares
        attitude_profile_kwargs  additional arguments passed to the
                                 attitude_profile_matrix() method if B is not
                                 provided

    Returns:
        A 4x4 matrix.

    """

    if B is None:
        B  = attitude_profile_matrix(**attitude_profile_kwargs)
    
    S  = B + B.T
    mu = np.trace(B)
    zx = B.T - B
    z  = np.array([[zx[2,1]], [zx[0,2]], [zx[1,0]]])

    if covariance_analysis:
        M = S - np.identity(3) * (2*mu)
        upperleft = 0.0
    else:
        M = S - np.identity(3) * mu
        upperleft = mu
    K = np.vstack((np.hstack((np.array([[upperleft]]), z.T)),
                   np.hstack((z,                       M))))
    return K


def attitude_profile_matrix(q           = None,
                            cov         = None,
                            inverse_cov = None,
                            obs         = None,
                            ref         = None,
                            weights     = None):

    """"There exists a unique orthogonal matrix A which satisfies
   
       A \hat{r}_i = \hat{s}_i for i in (1,2,3)

    which is given by

       A = \sum_{i=1}^{3} \hat{s}_i \hat{r}_i^\top
    
    " [0] and is called the attitude profile matrix. Note that
    observation vector \hat{s}_i is 3x1 and reference vector
    \hat{r}_i^\top is 1x3 in the above. Also permitted is a weight on
    each vector observation describing the relative precision of the
    sensor which provided it; see Eq. 5 from [1] and Eq. 10 from [2].

    Compute the attitude profile matrix B using either vector
    observations and reference vectors or a quaternion and 3x3
    attitude error covariance.

    weights should be positive and sum to 1.0.

    References:

    [0] Shuster, Oh. 1981. Three-axis attitude determination from
        vector observations. Journal of Guidance and Control 4(1):
        70--77.

    [1] Mortari. 1997. ESOQ-2 single-point algorithm for fast optimal
        spacecraft attitude determination. In Advances in the
        Astronautical Sciences.

    [2] Ainscough, Zanetti, Christian, Spanos. 2015. Q-Method extended
        Kalman filter. Journal of Guidance, Control, and Dynamics
        38(4): 752--760.

    Kwargs:
        q    

    """
    if q is not None:
        if inverse_cov is None:
            inverse_cov = spl.inv(cov)

        tr = np.trace(inverse_cov)
        return np.dot(np.identity(3) * tr * 0.5 - inverse_cov, q.to_matrix())

    # If we get this far, q was not provided; instead we're dealing with
    # vector observations.

    # Make sure we have a weights vector since this is a weighted least
    # squares problem.
    n = obs.shape[1]
    if weights is None:
        weights = np.ones(n) / float(n)

    B = np.zeros((3,3))
    for ii in range(0, n):
        B += np.dot(obs[0:3,ii].reshape((3,1)), ref[0:3,ii].reshape((1,3))) * weights[ii]
        
    return B
