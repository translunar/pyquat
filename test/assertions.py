import unittest
import numpy as np
import pyquat as pq
from pyquat import Quat

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
        self.assertEqual(q, Quat.from_matrix(m), **kwargs)
        
    def assert_almost_equal_as_matrix(self, q, m, **kwargs):
        """ convert a quaternion to a matrix and compare it to m """
        np.testing.assert_array_almost_equal(q.to_matrix(), m, **kwargs)

    def assert_almost_equal_as_quat(self, q, m, **kwargs):
        self.assert_almost_equal_components(q, Quat.from_matrix(m), **kwargs)
