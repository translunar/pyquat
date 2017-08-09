import unittest
import numpy as np
import pyquat as pq
from pyquat import Quat
from assertions import QuaternionTest

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

if __name__ == '__main__':
    unittest.main()
