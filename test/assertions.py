import typing
from typing import TYPE_CHECKING, Any, Optional

import unittest
import numpy as np
import math

if TYPE_CHECKING:
    import pyquat.pyquat as pq
    import pyquat.pyquat.esoq as esoq
    import pyquat.pyquat.wahba as wahba
else:
    import pyquat as pq
    import pyquat.wahba.esoq as esoq
    import pyquat.wahba as wahba

class QuaternionTest(unittest.TestCase):
    def assert_almost_equal_components(
            self,
            q1: pq.Quat,
            q2: pq.Quat,
            places: Optional[int] = None,
            msg: Optional[str] = None,
            delta: Optional[Any] = None,
        ) -> None:

        self.assertAlmostEqual(q1.w, q2.w, msg=msg, delta=delta, places=places)
        self.assertAlmostEqual(q1.x, q2.x, msg=msg, delta=delta, places=places)
        self.assertAlmostEqual(q1.y, q2.y, msg=msg, delta=delta, places=places)
        self.assertAlmostEqual(q1.z, q2.z, msg=msg, delta=delta, places=places)

    def assert_equal_as_matrix(
            self,
            q: pq.Quat,
            m: np.ndarray[Any, np.dtype[np.float64]],
            err_msg: str = '',
            verbose: bool = True,
            strict: bool = False,
        ) -> None:
        """ convert a quaternion to a matrix and compare it to m """
        np.testing.assert_array_equal(q.to_matrix(), m, err_msg=err_msg, verbose=verbose, strict=strict)

    def assert_equal_as_quat(
            self,
            q: pq.Quat,
            m: np.ndarray[Any, np.dtype[np.float64]],
            err_msg: str = '',
            verbose: bool = True,
            strict: bool = False,
        ) -> None:

        np.testing.assert_array_equal(
            q.to_vector(),
            pq.Quat.from_matrix(m).normalized().to_vector(),
            err_msg=err_msg,
            verbose=verbose,
            strict=strict,
        )
        
    def assert_almost_equal_as_matrix(
            self,
            q: pq.Quat,
            m: np.ndarray[Any, np.dtype[np.float64]],
            decimal: int = 6,
            err_msg: str = '',
            verbose: bool = True,
        ) -> None:

        """ convert a quaternion to a matrix and compare it to m """
        np.testing.assert_array_almost_equal(
            q.to_matrix(),
            m,
            decimal=decimal,
            err_msg=err_msg,
            verbose=verbose,
        )

    def assert_almost_equal_as_quat(
            self,
            q: pq.Quat,
            m: np.ndarray[Any, np.dtype[np.float64]],
            places: int = 7,
            msg: Optional[str] = None,
        ) -> None:

        self.assert_almost_equal_components(
            q,
            pq.Quat.from_matrix(m).normalized(),
            places=places,
            msg=msg,
        )

    def assert_equal(
            self,
            q1: pq.Quat,
            q2: pq.Quat,
            err_msg: str = '',
            verbose: bool = True,
            strict: bool = False,
        ) -> None:
        np.testing.assert_array_equal(
            q1.to_vector(),
            q2.to_vector(),
            err_msg=err_msg,
            verbose=verbose,
            strict=strict,
        )

    def assert_not_equal(
            self,
            q1: pq.Quat,
            q2: pq.Quat,
            err_msg: str = '',
            verbose: bool = True,
            strict: bool = False,
        ) -> None:

        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_equal,
            q1.to_vector(),
            q2.to_vector(),
            err_msg=err_msg,
            verbose=verbose,
            strict=strict,
        )

    def assert_almost_equal(
            self,
            q1: pq.Quat,
            q2: pq.Quat,
            decimal: int = 6,
            err_msg: str = '',
            verbose: bool = True,
        ) -> None:
        dot = q1.dot(q2)
        if dot > 1.0:  dot = 1.0
        if dot < -1.0: dot = -1.0
        np.testing.assert_array_almost_equal(
            np.array([0.0]),
            np.array([math.acos(dot)]),
            decimal=decimal,
            err_msg=err_msg,
            verbose=verbose,
        )

    def assert_not_almost_equal(
            self,
            q1: pq.Quat,
            q2: pq.Quat,
            decimal: int = 7,
            err_msg: str = '',
            verbose: bool = True
        ) -> None:
        # This is a hack.
        dot     = q1.dot(q2)
        if dot > 1.0:  dot = 1.0
        if dot < -1.0: dot = -1.0
        actual  = np.array([dot])
        desired = np.array([1.0])
        
        def _build_err_msg() -> str:
            header = ('Arrays are almost equal to %d decimals' % decimal)
            return np.testing.build_err_msg([actual, desired], err_msg, verbose=verbose,
                                 header=header)
    
        try:
            self.assert_almost_equal(q1, q2, decimal=decimal, err_msg=err_msg, verbose=verbose)
        except AssertionError:
            return None
        raise AssertionError(_build_err_msg())
        

    def assert_esoq2_two_observations_correct(
            self,
            ref: np.ndarray[Any, np.dtype[np.float64]],
            obs: np.ndarray[Any, np.dtype[np.float64]],
            decimal: int = 6,
            err_msg: str = '',
            verbose: bool = True,
        ) -> None:
        """
        Tests esoq2. Requires that ref and obs be 3x2 matrices (where each column
        is a ref or obs vector, respectively).
        """

        # First, compute the quaternion mapping the ref frame to the obs frame.
        B = wahba.attitude_profile_matrix(obs = obs, ref = ref)
        irot = esoq.sequential_rotation_helper(B)
        K = wahba.davenport_matrix(B)
        q, loss = esoq.esoq2(K, B, n_obs = 2)
        q_ref_to_obs = esoq.quat_sequential_rotation(q = q, irot = irot)

        T_ref_to_obs = q_ref_to_obs.to_matrix()

        obs_result = np.dot(T_ref_to_obs, ref)
        np.testing.assert_array_almost_equal(
            obs,
            obs_result,
            decimal=decimal,
            err_msg=err_msg,
            verbose=verbose,             
        )

