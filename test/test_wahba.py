import numpy as np
from scipy import linalg
import pyquat as pq
from pyquat import Quat
import pyquat.wahba as pqw
from assertions import QuaternionTest
import math
import unittest
from esoq2p1 import esoq2p1
        
class TestWahba(QuaternionTest):
    def test_attitude_profile_matrix_from_quaternion(self):
        q   = pq.identity()
        cov = np.identity(3)
        B   = pqw.attitude_profile_matrix(q, cov)
        # Needs actual test here

    def test_attitude_profile_matrix(self):
        ref = np.array([[1.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 1.0]])
        obs = np.array([[0.0, 0.0],
                        [1.0, 0.0],
                        [0.0, 1.0]])
        B = pqw.attitude_profile_matrix(obs = obs, ref = ref)
        # Needs actual test here


    def test_davenport_matrix_from_quaternion(self):
        q   = pq.identity()
        cov = np.identity(3)
        K1  = pqw.davenport_matrix(q = q, cov = cov)

        B   = pqw.attitude_profile_matrix(q, cov)
        K2  = pqw.davenport_matrix(B)
        np.testing.assert_array_equal(K1, K2)

    def test_davenport_matrix(self):
        ref = np.array([[1.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 1.0]])
        obs = np.array([[0.0, 0.0],
                        [1.0, 0.0],
                        [0.0, 1.0]])
        B = pqw.attitude_profile_matrix(obs = obs, ref = ref)
        K = pqw.davenport_matrix(B)
        self.assertEqual(K.shape[0], 4)
        self.assertEqual(K.shape[1], 4)

    def test_davenport_eigenvalues(self):
        return
        ref = np.array([[1.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 1.0]])
        obs = np.array([[0.0, 0.0],
                        [1.0, 0.0],
                        [0.0, 1.0]])
        B = pqw.attitude_profile_matrix(obs = obs, ref = ref)
        irot = pqw.sequential_rotation(B)
        K = pqw.davenport_matrix(B)
        l = pqw.davenport_eigenvalues(K, n_obs = 2)
        self.assertLessEqual(-1.0 - 1e-6, l[3])
        self.assertLessEqual(l[3], l[2])
        self.assertLessEqual(l[2], l[1])
        self.assertLessEqual(l[1], l[0])
        self.assertLessEqual(l[0], 1.0 + 1e-6)

    def test_trace_adj(self):
        K = np.array([[-1.0, 0, 2, -2],
                      [0, 3, 0, 0],
                      [2, 0, -1, -2],
                      [-2, 0, -2, -1]])
        ta1 = pqw.trace_adj(K)
        ta2 = np.matrix(K).getH().trace()[0,0]
        self.assertEqual(ta1, ta2)

    def test_esoq2_0_rotation(self):
        """
        Test borrowed from https://github.com/muzhig/ESOQ2/blob/master/test.cpp
        """
        ref = np.array([[1.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 1.0]])
        obs = np.array([[1.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 1.0]])
        
        q, loss = pqw.esoq2(obs = obs, ref = ref)
        q1, loss1 = esoq2p1(obs, ref, np.ones(2) * 0.5)
        q1 = Quat(q1[0], q1[1], q1[2], q1[3])
        self.assert_equal(q, pq.identity())
        self.assert_equal(q, q1)

    def test_esoq2_90_z_rotation(self):
        return
        """
        Test borrowed from https://github.com/muzhig/ESOQ2/blob/master/test.cpp
        """
        ref = np.array([[1.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 1.0]])
        obs = np.array([[0.0, 0.0],
                        [1.0, 0.0],
                        [0.0, 1.0]])
        q1, loss1 = esoq2p1(obs, ref, np.ones(2) * 0.5)
        q, loss   = pqw.esoq2(obs = obs, ref = ref)
        q1 = Quat(q1[3], q1[0], q1[1], q1[2])
        self.assert_equal(q, q1)

    def test_esoq2_90_x_rotation(self):
        return
        """
        Test borrowed from https://github.com/muzhig/ESOQ2/blob/master/test.cpp
        """
        ref = np.array([[1.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 1.0]])
        obs = np.array([[1.0, 0.0],
                        [0.0, -1.0],
                        [0.0, 0.0]])
        q1, loss1 = esoq2p1(obs, ref, np.ones(2) * 0.5)
        q, loss   = pqw.esoq2(obs = obs, ref = ref)
        q1 = Quat(q1[3], q1[0], q1[1], q1[2])
        self.assert_equal(q, q1)

    def test_esoq2_90_y_rotation(self):
        return
        """
        Test borrowed from https://github.com/muzhig/ESOQ2/blob/master/test.cpp
        """
        ref = np.array([[1.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 1.0]])
        obs = np.array([[0.0, 1.0],
                        [0.0, 0.0],
                        [-1.0, 0.0]])
        q1, loss1 = esoq2p1(obs, ref, np.ones(2) * 0.5)
        q, loss   = pqw.esoq2(obs = obs, ref = ref)
        q1 = Quat(q1[3], q1[0], q1[1], q1[2])
        self.assert_equal(q, q1)


if __name__ == '__main__':
    unittest.main()
