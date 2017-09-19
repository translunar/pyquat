import numpy as np
from scipy import linalg
import pyquat as pq
from pyquat import Quat
from assertions import QuaternionTest
import math
import unittest

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

    def test_skew(self):
        v = np.array([[1.0, 2.0, 3.0]]).T
        vx = pq.skew(v)
        np.testing.assert_array_equal(vx, np.array([[ 0.0, -3.0,  2.0],
                                                    [ 3.0,  0.0, -1.0],
                                                    [-2.0,  1.0,  0.0]]))

    def test_propagate(self):
        dt = 0.01
        q0 = Quat(1.0, 0.0, 0.0, 0.0)
        w0 = np.array([[0.0, 0.0, 1.0]]).T
        q1 = pq.propagate(q0, w0, dt)
        phi1 = q1.to_rotation_vector()
        phi2 = w0 * dt
        q2   = pq.from_rotation_vector(phi2)
        self.assert_equal(q1, q2)

        # Test quaternion propagation
        q3 = pq.propagate(q0, w0, dt)
        self.assert_equal(q1, q3)

        # Compare the directed cosine matrix result
        T0   = q0.to_matrix()
        w0x  = pq.skew(w0)
        Tdot = np.dot(T0, w0x)
        T1   = np.identity(3) - Tdot * dt # dT = (I - [phi x])
        self.assert_almost_equal_as_quat(q1, T1)

    def test_0_propagate(self):
        dt = 0.01
        q0 = Quat(1.0, 0, 0, 0)
        w0 = np.zeros((3,1))
        q1 = pq.propagate(q0, w0, dt)
        self.assert_equal(q0, q1)

    def test_rk4_integration(self):
        dt = 0.05
        q = Quat(1.0, 2.0, 3.0, 4.0).normalized()
        w = np.array([[0.03, 0.02, 0.01]]).T
        J = np.diag([200.0, 200.0, 100.0])
        J_inv = linalg.inv(J)

        # Test propagation using RK4
        q1, w1 = pq.step_rk4(q,  w,  dt,     J = J, J_inv = J_inv)
        qa, wa = pq.step_rk4(q,  w,  dt*0.5, J = J, J_inv = J_inv)
        q2, w2 = pq.step_rk4(qa, wa, dt*0.5, J = J, J_inv = J_inv)

    def test_expm(self):
        B1    = 13/51.0
        dt    = 0.05
        w     = np.array([[0.03, 0.02, 0.01]]).T
        J     = np.diag([200.0, 200.0, 100.0])
        J_inv = linalg.inv(J)
        wk1   = pq.wdot(w, J)
        qk1   = pq.state_transition_matrix(w)
        expm1 = pq.expm(w, dt * B1)
        expm2 = linalg.expm(qk1 * (B1 * dt))
        np.testing.assert_array_almost_equal(expm1, expm2)
        
    def test_cg_integration(self):
        """
        CG3 and CG4 integration
        """
        dt = 0.1
        q = Quat(1.0, 2.0, 3.0, 4.0).normalized()
        w = np.array([[0.03, 0.02, 0.01]]).T
        J = np.diag([200.0, 200.0, 100.0])

        # Test propagation using RK4
        q1, w1 = pq.step_rk4(q, w, dt, J = J)

        # Test propagation using CG3
        q2, w2 = pq.step_cg3(q, w, dt, J = J)

        # Test propagation using CG4
        q3, w3 = pq.step_cg4(q, w, dt, J = J)

        self.assert_almost_equal(q2, q3)
        self.assert_almost_equal(q1, q2)
        np.testing.assert_array_almost_equal(w2, w3)
        np.testing.assert_array_almost_equal(w1, w2)

    def test_integration_handles_zero(self):
        dt = 0.1
        q = pq.identity()
        w = np.zeros((3,1))
        J = np.diag([200.0, 200.0, 100.0])
        q1, w1 = pq.step_rk4(q, w, dt, J = J)
        q2, w2 = pq.step_cg3(q, w, dt, J = J)
        q3, w3 = pq.step_cg4(q, w, dt, J = J)

        self.assert_equal(q, q1)
        self.assert_equal(q, q2)
        self.assert_equal(q, q3)

        
if __name__ == '__main__':
    unittest.main()
