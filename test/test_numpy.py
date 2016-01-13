import unittest
import numpy as np
import pyquat

class TestNumpyMethods(unittest.TestCase):

    def test_mean(self):
        self.assertEqual(pyquat.mean(np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]).T), 
                         pyquat.Quat(1.0, 0.0, 0.0, 0.0))
        self.assertEqual(pyquat.mean(np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]).T),
                         pyquat.Quat(0.70710678118654757, 0.70710678118654757, 0.0, 0.0))
        self.assertEqual(pyquat.mean(np.array([pyquat.Quat(1.0, 0.0, 0.0, 0.0), pyquat.Quat(0.0, 1.0, 0.0, 0.0)])),
                         pyquat.Quat(0.70710678118654757, 0.70710678118654757, 0.0, 0.0))


if __name__ == '__main__':
    unittest.main()
