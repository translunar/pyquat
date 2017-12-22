import numpy as np
from scipy import linalg
import pyquat as pq
from pyquat import Quat
import pyquat.wahba.valenti as pqv
import pyquat.random as pqr
from test.assertions import QuaternionTest
import math
import unittest
        
class TestWahbaValenti(QuaternionTest):
    def test_q_acc_points_down(self):
        """ Tests that q_acc produces an arbitrary quaternion which rotates the gravity vector [0,0,1]' (down) to point at a measurement"""
        for ii in range(0,100):
            a0 = pqr.uniform_random_axis().reshape(3)
            d  = np.array([0.0, 0.0, 1.0])
        
            q  = pqv.q_acc(a0)
            a1 = q.rotate(d)
            np.testing.assert_almost_equal(a0, a1, decimal=14)

            q  = pqv.dq_acc(a0)
            a1 = q.rotate(d)
            np.testing.assert_almost_equal(a0, a1, decimal=14)

    def test_q_mag_points_north(self):
        """ Tests that q_mag produces an arbitrary quaternion which points a random vector at [1,0,0]' (magnetic north); also tests that q_global_to_local gives the same result"""
        for ii in range(0, 100):
            a  = pqr.uniform_random_axis().reshape(3)
            b  = pqr.uniform_random_axis().reshape(3)
        
            while a.dot(b) > 0.95: # make sure b isn't the same direction as a
                b = pqr.uniform_random_axis().reshape(3)
        
            qa = pqv.q_acc(a) # qa rotates a to being 0,0,1 (so into an xyD frame)
            l  = qa.conjugated().rotate(b)
            qm = pqv.q_mag(l)

            dqm = pqv.dq_mag(l)

            # Test that qm is about-z only
            self.assertEqual(qm.x, 0.0)
            self.assertEqual(qm.y, 0.0)

            # Do the same for dq_mag
            self.assertEqual(dqm.x, 0.0)
            self.assertEqual(dqm.y, 0.0)
        
            q_G_to_L  = qa * qm
            q_L_to_G  = q_G_to_L.conjugated()

            # Test that q_G_to_L still points gravity at a measurement
            # (really actually unnecessary since qm is about-z only)
            d         = np.array([0.0, 0.0, 1.0])
            a1        = q_G_to_L.rotate(d)
            np.testing.assert_almost_equal(a, a1, decimal=14)

            # Test that q_L_to_G rotates b so its horizontal component is
            # toward [1,0,0]'
            northish = q_G_to_L.conjugated().rotate(b)
            self.assertAlmostEqual(northish[1], 0.0, places=14)
            self.assertGreaterEqual(northish[0], 0.0)

            # Test that q_global_to_local gives the same result.
            q_G_to_L_0 = pqv.q_global_to_local(a, b)
            self.assert_equal(q_G_to_L, q_G_to_L_0)
        
if __name__ == '__main__':
    unittest.main()
