"""Implements certain functions for solving an example of the Wahba problem.

Reference:

  [1] Valenti, Dryanovski, Xiao. 2015. Keeping a good attitude: A
      quaternion-based orientation filter for IMUs and MARGs. Sensors
      15(8): 19302-30.

"""

import numpy as np
from math import sqrt

from pyquat import valenti_q_mag as q_mag
from pyquat import valenti_q_acc as q_acc
from pyquat import valenti_dq_mag as dq_mag
from pyquat import valenti_dq_acc as dq_acc
    
    
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
    # q_a maps from L to intermediate.
    # Thus q_a_conj maps from intermediate to L, where intermediate
    # is just an about-z rotation from G.
    # Transform vector b into the intermediate frame.
    l        = q_a.conjugated().rotate(b)
    
    # Find the quaternion q_mag which rotates l into a vector lying on
    # the northeast/northwest half-plane.
    q_m      = q_mag(l)

    # Return the mapping from global to local.
    return q_a * q_m
