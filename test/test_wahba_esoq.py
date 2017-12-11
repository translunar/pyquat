import numpy as np
from scipy import linalg
from test.assertions import QuaternionTest
import math
import unittest

from .context import pq, pq_esoq
        
class TestWahbaESOQ(QuaternionTest):
    def test_attitude_profile_matrix_from_quaternion(self):
        """attitude_profile_matrix() doesn't raise errors when given a quaternion measurement and a covariance"""
        q   = pq.identity()
        cov = np.identity(3)
        B   = pq_esoq.attitude_profile_matrix(q, cov)
        # Needs actual test here

    def test_attitude_profile_matrix(self):
        """attitude_profile_matrix() doesn't raise errors when given observation and reference vectors"""
        ref = np.array([[1.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 1.0]])
        obs = np.array([[0.0, 0.0],
                        [1.0, 0.0],
                        [0.0, 1.0]])
        B = pq_esoq.attitude_profile_matrix(obs = obs, ref = ref)
        # Needs actual test here


    def test_davenport_matrix_from_quaternion(self):
        """The Davenport matrix produced by a quaternion is the same as that produced by attitude_profile_matrix() called on that same quaternion"""
        q   = pq.identity()
        cov = np.identity(3)
        K1  = pq_esoq.davenport_matrix(q = q, cov = cov)

        B   = pq_esoq.attitude_profile_matrix(q, cov)
        K2  = pq_esoq.davenport_matrix(B)
        np.testing.assert_array_equal(K1, K2)

    def test_davenport_matrix(self):
        """davenport_matrix() called on the output of attitude_profile_matrix() produces a matrix of the correct shape"""
        ref = np.array([[1.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 1.0]])
        obs = np.array([[0.0, 0.0],
                        [1.0, 0.0],
                        [0.0, 1.0]])
        B = pq_esoq.attitude_profile_matrix(obs = obs, ref = ref)
        K = pq_esoq.davenport_matrix(B)
        self.assertEqual(K.shape[0], 4)
        self.assertEqual(K.shape[1], 4)

    def test_davenport_eigenvalues(self):
        """davenport_eigenvalues() produces eigenvalues in the appropriate range"""
        ref = np.array([[1.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 1.0]])
        obs = np.array([[0.0, 0.0],
                        [1.0, 0.0],
                        [0.0, 1.0]])
        B = pq_esoq.attitude_profile_matrix(obs = obs, ref = ref)
        irot = pq_esoq.sequential_rotation(B)
        K = pq_esoq.davenport_matrix(B)
        l = pq_esoq.davenport_eigenvalues(K, B, n_obs = 2)
        self.assertLessEqual(-1.0 - 1e-6, l[3])
        self.assertLessEqual(l[3], l[2])
        self.assertLessEqual(l[2], l[1])
        self.assertLessEqual(l[1], l[0])
        self.assertLessEqual(l[0], 1.0 + 1e-6)

    def test_trace_adj(self):
        """The adjugate trace is as expected"""
        K = np.array([[-1.0, 0, 2, -2],
                      [0, 3, 0, 0],
                      [2, 0, -1, -2],
                      [-2, 0, -2, -1]])
        ta1 = pq_esoq.trace_adj(K)
        ta2 = np.matrix(K).getH().trace()[0,0]
        self.assertEqual(ta1, ta2)

    def test_esoq2_0_rotation(self):
        """ESOQ2 works properly with no rotation between ref and obs"""
        #Test borrowed from https://github.com/muzhig/ESOQ2/blob/master/test.cpp
        ref = np.array([[1.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 1.0]])
        obs = np.array([[1.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 1.0]])
        self.assert_esoq2_two_observations_correct(ref = ref, obs = obs, decimal=12)

    def test_esoq2_90_z_rotation(self):
        """ESOQ2 works properly with a 90-degree about-z rotation between ref and obs"""

        # Test borrowed from https://github.com/muzhig/ESOQ2/blob/master/test.cpp

        ref = np.array([[1.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 1.0]])
        obs = np.array([[0.0, 0.0],
                        [1.0, 0.0],
                        [0.0, 1.0]])
        self.assert_esoq2_two_observations_correct(ref = ref, obs = obs, decimal=12)

    def test_esoq2_90_x_rotation(self):
        """ESOQ2 works properly with a 90-degree about-x rotation between ref and obs"""
        
        # Test borrowed from https://github.com/muzhig/ESOQ2/blob/master/test.cpp

        ref = np.array([[1.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 1.0]])
        obs = np.array([[1.0, 0.0],
                        [0.0, -1.0],
                        [0.0, 0.0]])
        self.assert_esoq2_two_observations_correct(ref = ref, obs = obs, decimal=12)

    def test_esoq2_90_y_rotation(self):
        """ESOQ2 works properly with a 90-degree about-y rotation between ref and obs"""
        
        # Test borrowed from https://github.com/muzhig/ESOQ2/blob/master/test.cpp
        ref = np.array([[1.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 1.0]])
        obs = np.array([[0.0, 1.0],
                        [0.0, 0.0],
                        [-1.0, 0.0]])
        self.assert_esoq2_two_observations_correct(ref = ref, obs = obs, decimal=12)

        
    def test_esoq2_actual_rotation(self):
        """Verifies that ESOQ2 produces the expected result for an arbitrary set"""
        ref_d = np.array([[-0.13745816],
                          [ 0.44258304],
                          [ 0.88612951]])
        ref_e = np.array([[-0.27904739],
                          [ 0.8410828 ],
                          [-0.46337056]])
        obs_d = np.array([[-0.65361096],
                          [ 0.38250994],
                          [ 0.65305348]])
        obs_e = np.array([[ 0.37343209],
                          [ 0.91352261],
                          [-0.16132242]])

        ref   = np.hstack((ref_d, ref_e))
        obs   = np.hstack((obs_d, obs_e))
        self.assert_esoq2_two_observations_correct(ref = ref, obs = obs, decimal=8)        

if __name__ == '__main__':
    unittest.main()
