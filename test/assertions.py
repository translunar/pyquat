import unittest
import numpy as np
import math

from .context import pq
from .context import pq_esoq

class QuaternionTest(unittest.TestCase):
    def assert_almost_equal_components(self, q1, q2, **kwargs):
        self.assertAlmostEqual(q1.w, q2.w, **kwargs)
        self.assertAlmostEqual(q1.x, q2.x, **kwargs)
        self.assertAlmostEqual(q1.y, q2.y, **kwargs)
        self.assertAlmostEqual(q1.z, q2.z, **kwargs)

    def assert_equal_as_matrix(self, q, m, **kwargs):
        """ convert a quaternion to a matrix and compare it to m """
        np.testing.assert_array_equal(q.to_matrix(), m, **kwargs)

    def assert_equal_as_quat(self, q, m, **kwargs):
        self.assertEqual(q, pq.Quat.from_matrix(m).normalized(), **kwargs)
        
    def assert_almost_equal_as_matrix(self, q, m, **kwargs):
        """ convert a quaternion to a matrix and compare it to m """
        np.testing.assert_array_almost_equal(q.to_matrix(), m, **kwargs)

    def assert_almost_equal_as_quat(self, q, m, **kwargs):
        self.assert_almost_equal_components(q, pq.Quat.from_matrix(m).normalized(), **kwargs)

    def assert_equal(self, q1, q2, **kwargs):
        self.assertEqual(q1, q2, **kwargs)

    def assert_not_equal(self, q1, q2, **kwargs):
        self.assertNotEqual(q1, q2, **kwargs)

    def assert_almost_equal(self, q1, q2, **kwargs):
        np.testing.assert_array_almost_equal(np.array([0.0]), np.array([math.acos(q1.dot(q2))]), **kwargs)

    def assert_not_almost_equal(self, q1, q2, decimal=7,err_msg='',verbose=True):
        # This is a hack.
        actual  = np.array([q1.dot(q2)])
        desired = np.array([1.0])
        
        def _build_err_msg():
            header = ('Arrays are almost equal to %d decimals' % decimal)
            return np.testing.build_err_msg([actual, desired], err_msg, verbose=verbose,
                                 header=header)
    
        try:
            self.assert_almost_equal(q1, q2, decimal=decimal)
        except AssertionError:
            return None
        raise AssertionError(_build_err_msg())
        

    def assert_esoq2_two_observations_correct(self, ref, obs, **kwargs):
        """
        Tests esoq2. Requires that ref and obs be 3x2 matrices (where each column
        is a ref or obs vector, respectively).
        """

        # First, compute the quaternion mapping the ref frame to the obs frame.
        B = pq_esoq.attitude_profile_matrix(obs = obs, ref = ref)
        irot = pq_esoq.sequential_rotation(B)
        K = pq_esoq.davenport_matrix(B)
        q, loss = pq_esoq.esoq2(K, n_obs = 2)
        q_ref_to_obs = pq_esoq.sequential_rotation(q = q, irot = irot)

        T_ref_to_obs = q_ref_to_obs.to_matrix()

        obs_result = np.dot(T_ref_to_obs, ref)
        np.testing.assert_array_almost_equal(obs, obs_result, **kwargs)

        
        

        
