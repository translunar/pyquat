import numpy as np
from scipy import linalg
from test.assertions import QuaternionTest
import math
import unittest

from .context import pq      

class TestPyquat(QuaternionTest):
    """Tests basic functionality on pyquat and the pyquat Quaternion type 'Quat' written in C"""

    def test_mean(self):
        """The mean quaternion is computed properly"""
        self.assert_equal(
            pq.mean(np.array([[1.0, 0.0, 0.0, 0.0],
                              [1.0, 0.0, 0.0, 0.0]]).T), 
            pq.Quat(1.0, 0.0, 0.0, 0.0))
        
        self.assert_almost_equal_components(
            pq.mean(np.array([[1.0, 0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0, 0.0]]).T),
            pq.Quat(0.70710678118654757, 0.70710678118654757, 0.0, 0.0),
            delta=1e-12)
        
        self.assert_almost_equal_components(
            pq.mean(np.array([pq.Quat(1.0, 0.0, 0.0, 0.0),
                              pq.Quat(0.0, 1.0, 0.0, 0.0)])),
            pq.Quat(0.70710678118654757, 0.70710678118654757, 0.0, 0.0),
            delta=1e-12)

    def test_identity(self):
        """The identity quaternion is properly constructed"""
        self.assert_equal_as_matrix(
            pq.identity(),
            np.identity(3))
        self.assert_equal_as_quat(
            pq.identity(),
            np.identity(3))

    def test_symmetric_matrix_conversion(self):
        """Conversion to and from a matrix is symmetric"""
        q = pq.Quat(0.4, -0.3, 0.2, -0.1)
        q.normalize()
        self.assert_almost_equal_as_quat(
            q,
            q.to_matrix())
        self.assert_almost_equal_as_quat(
            q.conjugated(),
            q.to_matrix().T)

    def test_symmetric_rotation_vector_conversion(self):
        """Conversion to and from a rotation vector is symmetric"""
        q = pq.Quat(0.4, -0.3, 0.2, -0.1)
        q.normalize()
        self.assert_almost_equal_components(q, pq.Quat.from_rotation_vector(q.to_rotation_vector()))
        

    def test_symmetric_angle_axis_conversion(self):
        """Conversion to and from an angle-axis is symmetric"""
        q = pq.Quat(0.4, -0.3, 0.2, -0.1)
        q.normalize()
        phi = q.to_rotation_vector()
        angle = linalg.norm(phi)
        phi_hat = phi / angle
        self.assert_almost_equal_components(q, pq.Quat.from_angle_axis(angle, phi_hat[0], phi_hat[1], phi_hat[2]))

    def test_symmetric_conjugate(self):
        """Quaternion conjugation is symmetric"""
        q = pq.Quat(0.4, -0.3, 0.2, -0.1)
        qT = q.conjugated()
        self.assert_equal(q, qT.conjugated())

    def test_small_rotation_vector(self):
        """Construction directly from a small rotation vector produces the same quaternion as conversion of a rotation vector to a matrix and then to a quaternion"""
        v = np.zeros((3,1)) * 3.0 / math.sqrt(3.0)
        q = pq.Quat.from_rotation_vector(v)
        self.assert_equal(q, pq.identity())
        T = pq.rotation_vector_to_matrix(v)
        np.testing.assert_array_equal(T, q.to_matrix())

    def test_large_rotation_vector(self):
        """Construction directly from a large rotation vector produces an almost-identical quaternion as conversion of a rotation vector to a matrix and then to a quaternion
        """        
        v = np.array([[3.0, 2.0, 1.0]]).T
        T1 = pq.Quat.from_rotation_vector(v).to_matrix()
        T2 = pq.rotation_vector_to_matrix(v)
        np.testing.assert_array_almost_equal(T1, T2)

    def test_multiplication(self):
        """Quaternion multiplication produces the same result as attitude matrix multiplication"""
        qAB = pq.Quat(0.4, -0.3, 0.2, -0.1)
        qAB.normalize()
        qBC = pq.Quat(0.2, 0.3, -0.4, 0.5)
        qBC.normalize()
        qAC = qBC * qAB
        qAC.normalize()

        tAB = qAB.to_matrix()
        tBC = qBC.to_matrix()
        tAC = np.dot(tBC, tAB)
        self.assert_almost_equal_as_matrix(qAC, tAC)

    def test_normalize(self):
        """Normalization works properly for typical values """
        q0 = pq.Quat(4.0, 3.0, 2.0, 1.0)
        q1 = pq.Quat(4.0, 3.0, 2.0, 1.0)
        q5 = pq.Quat(4.0, 3.0, 2.0, 1.0)
        q2 = q1.normalized()

        # Test that normalized() changed q2 and not q1
        self.assert_not_equal(q1, q2)
        self.assert_equal(q0, q1)

        # Test that normalized_large() works
        q4 = q1.normalized_large()
        self.assert_equal(q2, q4)

        # Now test that normalize() changes q1
        q1.normalize()
        self.assert_not_equal(q0, q1)
        self.assert_equal(q1, q2)

        # Test that normalize_large() changes q4 correctly
        q5.normalize_large()
        self.assert_equal(q1, q5)

        # Now test that normalize does what we expect it to do.
        v = np.array([[4.0, 3.0, 2.0, 1.0]]).T
        v /= linalg.norm(v)
        q3 = pq.Quat(v[0], v[1], v[2], v[3])
        self.assert_equal(q1, q3)

        # Now test that normalize handles invalid quaternions.
        q3 = pq.Quat(0.0, 0.0, 0.0, 0.0)
        self.assert_equal(q3.normalized(), pq.identity()) # out-of-place test
        self.assert_equal(q3, pq.Quat(0.0, 0.0, 0.0, 0.0))
        q3.normalize()
        self.assert_equal(q3, pq.identity()) # in-place test

    def test_normalize_large(self):
        """Overflow is avoided in normalization"""
        q_max = 3.9545290113758423e+256
        q0 = pq.Quat(-9.6241008572232875e+255,
                   q_max,
                  -2.3364730154155227e+255,
                  -2.2751942430616868e+256)
        v0 = q0.to_vector()
        v1 = v0 / q_max
        v1_mag = math.sqrt(v1[0,0]**2 + v1[1,0]**2 + v1[2,0]**2 + v1[3,0]**2)
        v2 = v1 / v1_mag
        q1 = pq.Quat(v2[0], v2[1], v2[2], v2[3]) 
        q2 = q0.normalized_large()
        self.assert_equal(q1, q2)

    def test_conjugate(self):
        """In-place and out-of-place conjugation works as expected"""
        q0 = pq.Quat(4.0, -3.0, -2.0, -1.0)
        q1 = pq.Quat(4.0,  3.0,  2.0,  1.0)

        # Test out-of-place
        self.assert_equal(q0, q1.conjugated())
        self.assert_not_equal(q0, q1)

        # Test in-place
        q1.conjugate()
        self.assert_equal(q0, q1)

    def test_skew(self):
        """The skew matrix is produced from a 3x1 vector correctly in C code"""
        w = np.array([[0.03, 0.02, 0.01]]).T

        # old method:
        skv = np.roll(np.roll(np.diag(w.flatten()), 1, 1), -1, 0)
        wx  = skv - skv.T
        
        np.testing.assert_array_equal(wx, pq.skew(w))        

    def test_propagate(self):
        """Simple propagation with small angular velocities produces the same result for matrices and quaternions"""
        dt = 0.01
        q0 = pq.Quat(1.0, 0.0, 0.0, 0.0)
        w0 = np.array([[3.0, 2.0, 1.0]]).T
        q1 = pq.propagate(q0, w0, dt)
        phi1 = q1.to_rotation_vector()
        phi2 = w0 * dt
        q2   = pq.from_rotation_vector(phi2)
        self.assert_almost_equal(q1, q2, decimal=6)

        # Test quaternion propagation
        q3 = pq.propagate(q0, w0, dt)
        self.assert_equal(q1, q3)

        # Compare the directed cosine matrix result
        T0   = q0.to_matrix()
        w0x  = pq.skew(w0)
        Tdot = np.dot(T0, w0x)
        T1   = np.identity(3) - Tdot * dt # dT = (I - [phi x])
        qT1  = pq.from_matrix(T1)
        self.assert_almost_equal(q1, qT1)

        # Compare to the additive result
        q4 = pq.propagate_additively(q0, w0, dt)
        self.assert_almost_equal(q4, qT1)

    def test_0_propagate(self):
        """Quaternion propagation with a 0 angular velocity does not lead to a zero division
        """
        dt = 0.01
        q0 = pq.Quat(1.0, 0, 0, 0)
        w0 = np.zeros((3,1))
        q1 = pq.propagate(q0, w0, dt)

        self.assert_equal(q0, q1)

    def test_rk4_integration(self):
        """Runge-Kutta 4 integration does not produce errors"""
        dt = 0.05
        q = pq.Quat(1.0, 2.0, 3.0, 4.0).normalized()
        w = np.array([[0.03, 0.02, 0.01]]).T
        J = np.diag([200.0, 200.0, 100.0])
        J_inv = linalg.inv(J)

        # Test propagation using RK4
        q1, w1 = pq.step_rk4(q,  w,  dt,     J = J, J_inv = J_inv)
        qa, wa = pq.step_rk4(q,  w,  dt*0.5, J = J, J_inv = J_inv)
        q2, w2 = pq.step_rk4(qa, wa, dt*0.5, J = J, J_inv = J_inv)

    def test_expm(self):
        """The closed form expression for expm from a quaternion and angular velocity produces the same result as linalg.expm()"""
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
        """CG3, CG4, and RK4 integration produce reasonably similar results"""
        dt = 0.1
        q = pq.Quat(1.0, 2.0, 3.0, 4.0).normalized()
        w = np.array([[0.3, 0.2, 0.1]]).T
        J = np.diag([200.0, 200.0, 100.0])

        # Test propagation using RK4
        q1, w1 = pq.step_rk4(q, w, dt, J = J, w_dynamics = pq.wdot)

        # Test propagation using CG3
        q2, w2 = pq.step_cg3(q, w, dt, J = J, w_dynamics = pq.wdot)

        # Test propagation using CG4
        q3, w3 = pq.step_cg4(q, w, dt, J = J, w_dynamics = pq.wdot)

        self.assert_almost_equal(q2, q3)
        self.assert_almost_equal(q1, q2)
        np.testing.assert_array_almost_equal(w2, w3)
        np.testing.assert_array_almost_equal(w1, w2)

        # Test propagation without angular velocity dynamics using each method
        q4, w4 = pq.step_rk4(q, w, dt, w_dynamics = None)
        q5, w5 = pq.step_cg3(q, w, dt, w_dynamics = None)
        q6, w6 = pq.step_cg4(q, w, dt, w_dynamics = None)

        self.assert_almost_equal(q4, q5)
        self.assert_almost_equal(q5, q6)
        np.testing.assert_array_equal(w, w4)
        np.testing.assert_array_equal(w, w5)
        np.testing.assert_array_equal(w, w6)

    def test_cg_integration_tiny_steps(self):
        """CG4 integration produces reasonably similar results over many small steps as it would over one big step"""

        # This test mostly exists to show that integration is not very accurate.
        
        q0 = pq.Quat(1.0, 2.0, 3.0, 4.0).normalized()
        
        w = np.array([[-1.0, -3.0, -4.0]]).T * (math.pi / 180.0)
        
        dt_small = 0.001
        dt_large = 0.1
        t        = 1.0

        dt_total = 0.0
        q_large = q0.copy()
        while dt_total < t:
            q_large, w_out = pq.step_cg4(q_large, w, dt_large, w_dynamics = None)
            dt_total += dt_large
        q_large.normalize()
            
        dt_total = 0.0
        q_small  = q0.copy()
        while dt_total < t:
            q_small, w_out = pq.step_cg4(q_small, w, dt_small, w_dynamics = None)
            dt_total += dt_small
        q_small.normalize()

        self.assert_not_almost_equal(q_large, q0)
        self.assert_not_almost_equal(q_small, q0)

        self.assert_almost_equal(q_small, q_large, decimal=2) # FAIRLY INACCURATE!

    def test_integration_handles_zero(self):
        """CG3, CG4, and RK4 integration correctly hadnle an angular velocity of  0"""
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

    def test_big_omega(self):
        """The C big_omega() method produces the expected result"""
        w = np.array([[0.03, 0.02, 0.01]]).T
        W = pq.big_omega(w)

        W1 = np.array([[0.0,    -w[0,0], -w[1,0], -w[2,0]],
                       [w[0,0],  0.0,     w[2,0], -w[1,0]],
                       [w[1,0], -w[2,0],  0.0,     w[0,0]],
                       [w[2,0],  w[1,0], -w[0,0],  0.0]])
        
        np.testing.assert_array_equal(W, W1)

    def test_tobytes_and_fromstring(self):
        """Conversion to and from strings is symmetric"""
        q1 = pq.Quat(1.0, 2.0, 3.0, 4.0).normalized()
        s  = q1.tobytes()
        q2 = pq.fromstring(s, dtype=np.float64, count=4)
        self.assert_equal(q1, q2)

    def test_fromstring_with_sep(self):
        """Conversion from a string with separator character works properly"""
        q1 = pq.fromstring("1.0 2.0 3.0 4.0", sep=' ').normalized()
        q2 = pq.Quat(1.0, 2.0, 3.0, 4.0).normalized()
        self.assert_equal(q1, q2)

    def test_copy(self):
        """Copies the quaternion"""
        q1 = pq.Quat(1.0, 2.0, 3.0, 4.0).normalized()
        q2 = q1.copy()
        q2.w = 0.0
        q2.normalize()
        self.assert_not_equal(q1, q2)
        self.assert_equal(q1, pq.Quat(1.0, 2.0, 3.0, 4.0).normalized())

    def test_dot(self):
        """Dot product of two quaternions should be the dot product of their
        respective to_vector() methods"""
        q1 = pq.Quat(1.0, 0.0, 0.0, 0.0)
        q2 = pq.Quat(1.0, 1.0, 0.0, 0.0).normalized()
        self.assertEqual(q1.dot(q2),
                         q1.to_vector().reshape(4).dot(q2.to_vector().reshape(4)))
        
    def test_lerp(self):
        """Linear interpolation works as expected for a 45-degree angle"""
        q1 = pq.Quat(1.0, 0.0, 0.0, 0.0)
        q2 = pq.Quat(1.0, 1.0, 0.0, 0.0).normalized()

        # Check half-way point equidistant from inputs
        q3 = q1.lerp(q2, 0.5)
        self.assertAlmostEqual(q3.dot(q1), q3.dot(q2))

        # Check that each input is 22.5 degrees from the output
        q3n = q3.normalized()
        self.assertAlmostEqual(math.acos(q3n.dot(q2)) * 180.0 / math.pi, 22.5)
        self.assertAlmostEqual(math.acos(q3n.dot(q1)) * 180.0 / math.pi, 22.5)

        # Check that q1 lerped with itself gives itself.
        q4 = q1.lerp(q1.copy(), 1.0)
        self.assert_equal(q1, q4)
        q4 = q1.lerp(q1.copy(), 0.0)
        self.assert_equal(q1, q4)
        q4 = q1.lerp(q1.copy(), 0.5)
        self.assert_equal(q1, q4)

        # Check end-ranges of the t parameter
        q5 = q1.lerp(q2, 0.0)
        self.assert_equal(q5, q1)

        q6 = q1.lerp(q2, 1.0)
        self.assert_equal(q6, q2)

    def test_slerp(self):
        """Spherical linear interpolation works as expected for a 45-degree angle"""
        q1 = pq.Quat(1.0, 0.0, 0.0, 0.0)
        q2 = pq.Quat(1.0, 1.0, 0.0, 0.0).normalized()

        # Check half-way point equidistant from inputs
        q3 = q1.slerp(q2, 0.5)
        self.assertAlmostEqual(q3.dot(q1), q3.dot(q2))

        # Check that each input is 22.5 degrees from the output
        self.assertAlmostEqual(math.acos(q3.dot(q2)) * 180.0 / math.pi, 22.5)
        self.assertAlmostEqual(math.acos(q3.dot(q1)) * 180.0 / math.pi, 22.5)

        # Check that q1 lerped with itself gives itself.
        q4 = q1.slerp(q1.copy(), 1.0)
        self.assert_equal(q1, q4)
        q4 = q1.slerp(q1.copy(), 0.0)
        self.assert_equal(q1, q4)
        q4 = q1.slerp(q1.copy(), 0.5)
        self.assert_equal(q1, q4)

        # Check end-ranges of the t parameter
        q5 = q1.slerp(q2, 0.0)
        self.assert_equal(q5, q1)

        q6 = q1.slerp(q2, 1.0)
        self.assert_equal(q6, q2)

    def test_slerp_with_lerp(self):
        """slerp should resort to lerp when a lerp_threshold is set"""
        q1  = pq.Quat(1.0, 0.0, 0.0, 0.0)
        q2  = pq.Quat(1.0, 0.1, 0.0, 0.0).normalized()
        dot = q1.dot(q2)

        lerp  = q1.lerp(q2, 0.5).normalized()
        slerp = q1.slerp(q2, 0.5)
        mixed = q1.slerp(q2, 0.5, lerp_threshold = 0.9)
        self.assert_equal(lerp, mixed)
        self.assert_not_equal(lerp, slerp)
       
    

    def test_rotate(self):
        """Rotation of a vector using a quaternion is equivalent to standard matrix-vector multiplication over many random test quaternions and vectors (to 14 decimal places)"""
        import pyquat.random as pqr
        
        for ii in range(0, 100):
            q = pqr.rand()
            T = q.to_matrix()
            v = pqr.uniform_random_axis() * 10.0 # vector of length 10
            qvq = q.rotate(v)
            Tv  = T.dot(v)
            np.testing.assert_almost_equal(Tv, qvq, decimal=14)

         
        
if __name__ == '__main__':
    unittest.main()
