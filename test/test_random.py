import numpy as np
import pyquat as pq
import pyquat.random as pqr
from assertions import QuaternionTest
import math
import unittest
from scipy import linalg

class TestRandom(QuaternionTest):
    def test_uniform_random_axis(self):
        """
        Just tests that a unit axis is generated, not that it is
        uniform.
        """
        v = pqr.uniform_random_axis()
        uv = v / linalg.norm(v)
        np.testing.assert_array_almost_equal(v, uv, decimal=12)

    def test_randu(self):
        """
        Test that the random numbers generated are not outside
        the correct range (-1,+1).
        """
        for ii in range(0,100):
            v = pqr.randu()
            self.assertLessEqual(-1.0, v)
            self.assertGreaterEqual(1.0, v)

    def test_rand(self):
        """ Just tests that a unit quaternion is returned. """
        qr = pqr.rand()
        self.assertEqual(type(qr), pq.Quat)

        v = qr.to_vector()
        self.assertEqual(linalg.norm(v), 1.0)

        for ii in range(0,4):
            self.assertLessEqual(-1.0, v[ii])
            self.assertGreaterEqual(1.0, v[ii])
        

        
if __name__ == '__main__':
    unittest.main()
