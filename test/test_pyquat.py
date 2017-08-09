import unittest
import numpy as np
from numpy import linalg
import pyquat as pq
from pyquat import Quat
from assertions import QuaternionTest
import math

class TestPyquat(QuaternionTest):

    def test_mean(self):
        self.assertEqual(
            pq.mean(np.array([[1.0, 0.0, 0.0, 0.0],
                              [1.0, 0.0, 0.0, 0.0]]).T), 
            Quat(1.0, 0.0, 0.0, 0.0))
        
        self.assert_almost_equal_components(
            pq.mean(np.array([[1.0, 0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0, 0.0]]).T),
            Quat(0.70710678118654757, 0.70710678118654757, 0.0, 0.0),
            delta=1e-12)
        
        self.assert_almost_equal_components(
            pq.mean(np.array([Quat(1.0, 0.0, 0.0, 0.0),
                              Quat(0.0, 1.0, 0.0, 0.0)])),
            Quat(0.70710678118654757, 0.70710678118654757, 0.0, 0.0),
            delta=1e-12)

    def test_identity(self):
        self.assert_equal_as_matrix(
            pq.identity(),
            np.identity(3))
        self.assert_equal_as_quat(
            pq.identity(),
            np.identity(3))

    def test_symmetric_matrix_conversion(self):
        q = Quat(0.4, -0.3, 0.2, -0.1)
        q.normalize()
        self.assert_almost_equal_as_quat(
            q,
            q.to_matrix())
        self.assert_almost_equal_as_quat(
            q.conjugate(),
            q.to_matrix().T)

    def test_symmetric_rotation_vector_conversion(self):
        q = Quat(0.4, -0.3, 0.2, -0.1)
        q.normalize()
        self.assert_almost_equal_components(q, Quat.from_rotation_vector(q.to_rotation_vector()))
        

    def test_symmetric_angle_axis_conversion(self):
        q = Quat(0.4, -0.3, 0.2, -0.1)
        q.normalize()
        phi = q.to_rotation_vector()
        angle = linalg.norm(phi)
        phi_hat = phi / angle
        self.assert_almost_equal_components(q, Quat.from_angle_axis(angle, phi_hat[0], phi_hat[1], phi_hat[2]))

    def test_symmetric_conjugate(self):
        q = Quat(0.4, -0.3, 0.2, -0.1)
        qT = q.conjugated()
        self.assert_equal(q, qT.conjugated())

    def test_small_rotation_vector(self):
        v = np.zeros((3,1)) * 3.0 / math.sqrt(3.0)
        q = Quat.from_rotation_vector(v)
        self.assert_equal(q, pq.identity())
        T = pq.rotation_vector_to_matrix(v)
        np.testing.assert_array_equal(T, q.to_matrix())

    def test_large_rotation_vector(self):
        v = np.array([[3.0, 2.0, 1.0]]).T
        T1 = Quat.from_rotation_vector(v).to_matrix()
        T2 = pq.rotation_vector_to_matrix(v)
        np.testing.assert_array_almost_equal(T1, T2)

    def test_multiplication(self):
        qAB = Quat(0.4, -0.3, 0.2, -0.1)
        qAB.normalize()
        qBC = Quat(0.2, 0.3, -0.4, 0.5)
        qBC.normalize()
        qAC = qBC * qAB
        qAC.normalize()

        tAB = qAB.to_matrix()
        tBC = qBC.to_matrix()
        tAC = np.dot(tBC, tAB)
        self.assert_almost_equal_as_matrix(qAC, tAC)

    def test_normalize(self):
        q0 = Quat(4.0, 3.0, 2.0, 1.0)
        q1 = Quat(4.0, 3.0, 2.0, 1.0)
        q2 = q1.normalized()

        # Test that normalized() changed q2 and not q1
        self.assert_not_equal(q1, q2)
        self.assert_equal(q0, q1)

        # Now test that normalize() changes q1
        q1.normalize()
        self.assert_not_equal(q0, q1)
        self.assert_equal(q1, q2)

        # Now test that normalize does what we expect it to do.
        v = np.array([[4.0, 3.0, 2.0, 1.0]]).T
        v /= linalg.norm(v)
        q3 = Quat(v[0], v[1], v[2], v[3])
        self.assert_equal(q1, q3)

        # Now test that normalize handles invalid quaternions.
        q3 = Quat(0.0, 0.0, 0.0, 0.0)
        self.assert_equal(q3.normalized(), pq.identity()) # out-of-place test
        self.assert_equal(q3, Quat(0.0, 0.0, 0.0, 0.0))
        q3.normalize()
        self.assert_equal(q3, pq.identity()) # in-place test
    

    def test_conjugate(self):
        q0 = Quat(4.0, -3.0, -2.0, -1.0)
        q1 = Quat(4.0,  3.0,  2.0,  1.0)

        # Test out-of-place
        self.assert_equal(q0, q1.conjugated())
        self.assert_not_equal(q0, q1)

        # Test in-place
        q1.conjugate()
        self.assert_equal(q0, q1)
        
if __name__ == '__main__':
    unittest.main()
