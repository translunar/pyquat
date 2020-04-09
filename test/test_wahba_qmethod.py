import numpy as np
import numpy.random as npr
import scipy.linalg as spl

from test.assertions import QuaternionTest

from .context import pq, esoq, wahba, qmethod
        
class TestWahbaQMethod(QuaternionTest):
    
    def test_qmethod(self):
        q_inrtl_to_body_list = [
            pq.identity(),
            pq.Quat(1.0, -2.0, 3.0, 4.0).normalized(),
            pq.Quat(0.0, 1.0, 0.0, 0.0),
            pq.Quat(0.0, 0.0, 1.0, 0.0),
            pq.Quat(0.0, 0.0, 0.0, 1.0)]

        # Setup misalignments using skew matrices.
    
        # Our truth will be corrupted to reference by some error in our
        # knowledge about the world.

        # We will then corrupt those reference vectors by observation
        # errors (because our instruments are kind of shitty).
    
        ref_misalign     = npr.randn(3) * 1e-6
        sun_obs_misalign = npr.randn(3) * 1e-5
        mag_obs_misalign = npr.randn(3) * 1e-5
    
        T_ref_err = np.identity(3) #- pq.skew(ref_misalign)
        T_sun_obs_err = np.identity(3) - pq.skew(sun_obs_misalign)
        T_mag_obs_err = np.identity(3) - pq.skew(mag_obs_misalign)
    
        mag_truth = np.array([0.0, 0.1, 1.0])
        mag_truth /= spl.norm(mag_truth)

        sun_truth = np.array([0.5, 0.5, 0.02])
        sun_truth /= spl.norm(sun_truth)

        for qib in q_inrtl_to_body_list:
            Tib = qib.to_matrix()

            mag_ref = T_ref_err.dot(mag_truth)
            mag_ref /= spl.norm(mag_ref)
            sun_ref = T_ref_err.dot(sun_truth)
            sun_ref /= spl.norm(sun_ref)

            mag_obs  = T_mag_obs_err.dot(Tib.dot(mag_ref))
            mag_obs /= spl.norm(mag_obs)
            sun_obs  = T_sun_obs_err.dot(Tib.dot(sun_ref))
            sun_obs /= spl.norm(sun_obs)

            P_prior = np.identity(3) * np.pi**2
            N_prior = spl.inv(P_prior)


            # Assemble arguments to pass to qmethod()
            sigma_y = [1e-2, 1e-2]
            sigma_n = [1e-2, 1e-2]
            weights = qmethod.compute_weights(sigma_y, sigma_n)
            qmethod_args = (np.vstack((sun_obs, mag_obs)).T,
                            np.vstack((sun_ref, mag_ref)).T,
                            weights)            

            # Test qmethod with priors
            q_posterior = qmethod.qmethod(*qmethod_args,
                                          q_prior = pq.identity(),
                                          N_prior = N_prior)

            # Test qmethod without priors
            q_naive     = qmethod.qmethod(*qmethod_args)

            for q in (q_posterior, q_naive):
                dq_body = qib * q.conjugated()
                self.assertLess(np.abs(np.arccos(dq_body.w)), 1e-4)
