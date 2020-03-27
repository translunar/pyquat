import numpy as np

from .context import wahba

from test.assertions import QuaternionTest

class TestWahba(QuaternionTest):

    def test_attitude_profile_matrix(self):
        """attitude_profile_matrix() doesn't raise errors when given observation and reference vectors"""
        ref = np.array([[1.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 1.0]])
        obs = np.array([[0.0, 0.0],
                        [1.0, 0.0],
                        [0.0, 1.0]])
        B = wahba.attitude_profile_matrix(obs = obs, ref = ref)
        # Needs actual test here


    def test_davenport_matrix(self):
        """davenport_matrix() called on the output of attitude_profile_matrix() produces a valid Davenport K matrix"""
        ref = np.array([[1.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 1.0]])
        obs = np.array([[0.0, 0.0],
                        [1.0, 0.0],
                        [0.0, 1.0]])
        B = wahba.attitude_profile_matrix(obs = obs, ref = ref)
        K = wahba.davenport_matrix(B)
        self.assertEqual(K.shape[0], 4)
        self.assertEqual(K.shape[1], 4)

        # Test q-method davenport matrix
        K_qekf = wahba.davenport_matrix(B, covariance_analysis = True)
        self.assertEqual(K_qekf[0,0], 0.0)
        np.testing.assert_equal(K_qekf[0,1:3], K[0,1:3])
        for ii in range(0,3):
            self.assertNotEqual(K_qekf[ii,ii], K[ii,ii])
