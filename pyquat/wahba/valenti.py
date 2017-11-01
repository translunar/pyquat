"""Implements certain functions for solving an example of the Wahba problem.

Reference:

  [1] Valenti, Dryanovski, Xiao. 2015. Keeping a good attitude: A
      quaternion-based orientation filter for IMUs and MARGs. Sensors
      15(8): 19302-30.

"""

import numpy as np
from math import sqrt

def q_acc(a):
    """Helper function for wahba.valenti(). Computes the q_acc auxiliary
    quaternion using a relatively noiseless measurement a.

    Args:
        a:  relatively noiseless measurement vector with unit norm

    Returns: 
        An auxiliary quaternion q_acc which rotates the gravity
        vector [0,0,1]' such that it points at a; thus, q_acc's
        conjugate identifies an arbitrary frame xyD frame (where D is
        down, and x and y are arbitrary).

    """
    import pyquat as pq
    
    if a[2] >= 0:
        s2x1pay = sqrt(2.0 * (1+a[2]))
        
        return pq.Quat(  sqrt( (1+a[2]) / 2.0 ),
                         a[1] / s2x1pay,
                        -a[0] / s2x1pay,
                         0.0 )
    else:
        s2x1may = sqrt(2.0 * (1-a[2]))
        
        return pq.Quat( -a[1] / s2x1may,
                        -sqrt( (1-a[2]) / 2.0 ),
                         0.0,
                        -a[0] / s2x1may )


def q_mag(l):
    """Helper function for wahba.valenti(). Computes the q_mag auxiliary
    quaternion using a relatively noisy measurement b.

    Args:
        l:  relatively noisy measurement vector with unit norm, which
            has already been rotated into the xyD frame given by q_acc(a).

    Returns: 
        An auxiliary quaternion q_mag which rotates about the
        z-axis to an arbitrary xyD frame (from q_acc()) from a
        "global" NED frame where N is defined as the horizontal
        component of the magnetically northward vector (*not* true
        north).

    """
    import pyquat as pq

    gamma         = l[0]**2 + l[1]**2
    sqrt_gamma    = sqrt(gamma)
    lx_sqrt_gamma = l[0] * sqrt_gamma
    sqrt_2        = sqrt(2.0)

    if l[0] >= 0:
        sgplxsg = sqrt( gamma + lx_sqrt_gamma )
        return pq.Quat( sgplxsg / (sqrt_2 * sqrt_gamma),
                        0.0,
                        0.0,
                       -l[1] / (sqrt_2 * sgplxsg) )

    else:
        sgmlxsg = sqrt(gamma - lx_sqrt_gamma )
        return pq.Quat( l[1] / (sqrt_2 * sgmlxsg),
                        0.0,
                        0.0,
                       -sgmlxsg / (sqrt_2 * sqrt_gamma) )
        
    
    
def q_global_to_local(a, b):
    """Compute a global to local transformation using a relatively
    low-noise measurement a (e.g. from an accelerometer) and a
    relatively noisy measurement b (e.g. from a magnetometer).

    Reference:
        [1] Valenti, Dryanovski, Xiao. 2015. Keeping a good
            attitude: A quaternion-based orientation filter for IMUs and
            MARGs. Sensors 15(8): 19302-30.

    Args:
        a:  relatively noiseless accelerometer measurement (unit norm)
        b:  relatively noisy magnetometer measurement (unit norm)

    Returns:
        A quaternion \f$ q_G^L = q_{acc} \otimes q_{mag} \f$ mapping
        from the global frame G to the local sensor frame L. Note that
        G is an NED frame where D is aligned with gravity, and N points
        toward magnetic north (horizontal component), not true north.
    """

    q_a      = q_acc(a)
    # q_a maps from G to intermediate.
    # Thus q_a_conj maps from intermediate to G, where intermediate
    # is just an about-z rotation from L.
    # Transform vector b into the intermediate frame.
    l        = q_a.conjugated().rotate(b)
    
    # Find the quaternion q_mag which rotates l into a vector lying on
    # the northeast/northwest half-plane.
    q_m      = q_mag(l)

    # Return the mapping from global to local.
    return q_a * q_m
