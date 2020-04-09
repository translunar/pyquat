"""This module implements Davenport's q-Method [0] (outlined in [1]),
as described and notated in [2] (which is where the equation numbering
appearing in comments originates).

References:

[0] Davenport, P. 1968.

[1] De Ruiter, A.; Damaren, C.; Forbes, J. 2013. Spacecraft Dynamics
    and Control: An Introduction, 1st Ed. Wiley, West Sussex,
    U.K. pp. 468-471.

[2] Ainscough, T.; Zanetti, R.; Christian, J.; Spanos, P. D. 2015.
    Q-Method extended Kalman filter. Journal of Guidance, Control, and
    Dynamics 38(4): 752-760.

"""

import numpy as np
import scipy.linalg as spl

import pyquat as pq
from pyquat.wahba import attitude_profile_matrix
from pyquat.wahba import davenport_matrix

def qekf_measurement_model(T, y, n, w):
    """Generate a weighted least squares measurement model.

    This is eq. 18 from [2].

    Args:
        T   3x3 rotation matrix describing the prior attitude
        y   3xm matrix of unit 3D vector observations
        n   3xm matrix of corresponding unit 3D reference vectors
        w   list of weights corresponding to y and n

    Returns:
        A 3x3 measurement Jacobian.
    """
    H = np.zeros((3, 3))

    for ii in range(y.shape[1]):

        # Measurement Jacobian: eq 18
        yx  = pq.skew(y[0:3,ii])
        Tn  = T.dot(n[0:3,ii])
        Tnx = pq.skew(Tn)
        H += w[ii] * (yx.dot(Tnx) + Tnx.dot(yx))
        
    return H


def quest_measurement_covariance(vector, sigma):
    """Generate the measurement covariance from the QUEST measurement
    model for a given 3D unit vector and sigma.

    These are eqs. 5(a) and 5(b) in [2].

    Args:
        vector  unit vector
        sigma   standard deviation

    Returns:
        A 3x3 covariance matrix.
    """
    
    return (np.identity(3) - vector[0:3].reshape((3, 1)).dot(vector[0:3].reshape((1,3)))) * sigma


def qekf_measurement_covariance(T, y, n, w, sigma_y, sigma_n):
    """Generate a measurement covariance for the z vector by computing
    measurement covariances on the observations and reference vectors.

    The z vector is defined as the weighted sum of the cross-products
    of the observations and references. If

      B = \sum_{i=1}^n w_i y_i n_i^\top

    then

      \left[ z \times \right] = B^\top - B

    or alternatively,

      z = \sum_{i=1}^n w_i ( y_i \times n_i ) .

    This function first generates QUEST measurement model covariances
    for n and y (Eqs. 5(a) and 5(b) from [2]) and then uses these to
    produce a measurement covariance for z (Eq. 25 from [2]).

    The weights can be obtained using compute_weights(), or you can
    supply your own.

    Args:
        T        transformation matrix describing the prior attitude
        y        3xm matrix of unit vector observations
        n        3xm matrix of corresponding unit reference vectors
        a        array of measurement weights corresponding to y and n
        sigma_y  array of standard deviations for each unit vector observation
        sigma_n  array of standard deviations for each unit reference vector

    Returns:
        A weighted 3x3 measurement covariance.

    """
    R = np.zeros((3, 3))
    
    for ii in range(y.shape[1]):
        # QUEST measurement model: eq 5a, 5b
        Rnn = quest_measurement_covariance(n[:,ii], sigma_n[ii])
        Ryy = quest_measurement_covariance(y[:,ii], sigma_y[ii])

        yx  = pq.skew(y[0:3,ii])
        Tn  = T.dot(n[0:3,ii])
        Tnx = pq.skew(Tn)
        
        # Measurement covariance: eq 25
        yxT = yx.dot(T)
        R += (yxT.dot(Rnn.dot(yxT.T)) + Tnx.dot(Ryy.dot(Tnx.T))) * (4 * w[ii]**2)

    return R

        
def compute_weights(sigma_y, sigma_n):
    """Given the sigmas for observations y and reference vectors n, this
    method computes the weights needed for the measurement covariance for
    z.

    This method represents Eq. 6 from [2].

    Args:
        sigma_y  standard deviations for each observation vector y
        sigma_n  standard deviations for each reference vector n

    Returns:
        A numpy array containing as many entries as there are
    observation and reference vectors.

    """
    w = []
    for ii in range(len(sigma_y)):
        # Compute weights: eq 6
        w.append( 1.0 / (sigma_n[ii]**2 + sigma_y[ii]**2) )
    return np.array(w)

        
def qmethod(y, n,
            w       = None,
            sigma_y = None,
            sigma_n = None,
            q_prior = None,
            N_prior = None):
    """Runs Davenport's q-Method as described in [2], using priors as
    needed for a SOAR filter or qEKF. It produces a normalized quaternion
    corresponding to the largest eigenvalue of the Davenport matrix.

    See p. 5 in [2] for additional details.

    You can provide either weights w or sigmas sigma_y and sigma_n.
    Weights are computed from the latter two using compute_weights()
    if weights are not supplied.

    Args:
        y        3xm matrix of 3D unit vector observations
        n        3xm matrix of corresponding 3D unit reference vectors 

    Kwargs:
        w        weights corresponding to each of the m entries of y 
                 and n; can also be computed from sigma_y and sigma_n
                 if those are supplied instead
        sigma_y  sigmas corresponding to each of the m entries in y
        sigma_n  sigmas corresponding to each of the m entries in n
        q_prior  prior attitude, if any
        N_prior  prior information matrix (3x3); this is also the
                 inverse of the prior covariance matrix (you must
                 provide this if you provide q_prior)

    Returns:
        A normalized quaternion.

    """
    if q_prior is None:
        T = np.identity(3)
        q_prior = pq.identity()

        if N_prior is not None:
            raise ValueError("expected prior attitude to be given with prior information")
    else:
        T = q_prior.to_matrix()

        if N_prior is None:
            raise ValueError("expected prior information to be given with prior attitude")

    if w is None:
        w = compute_weights(sigma_y, sigma_n)

    B = attitude_profile_matrix(obs = y, ref = n, weights = w)
    K = davenport_matrix(B, covariance_analysis = True)

    if N_prior is not None:
        # A0 is twice the inverse covariance.
        A0 = N_prior * 2

        # "Condition" K on the prior attitude and prior information
        Xi_prior = q_prior.big_xi()
        K -= Xi_prior.dot(A0.dot(Xi_prior.T))        

    # Get eigenvalues and eigenvectors
    lam, v = spl.eig(K)
    ii = np.argmax(lam) # find the largest eigenvalue

    return pq.Quat(*v[:,ii]).normalized()
