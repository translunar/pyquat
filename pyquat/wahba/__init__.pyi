from typing import overload, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pyquat._pyquat import Quat
else:
    from _pyquat import Quat

import numpy as np

@overload
def attitude_profile_matrix(
    q: Quat,
    cov: Optional[np.ndarray[Any, np.dtype[np.float64]]] = None,
    inverse_cov: Optional[np.ndarray[Any, np.dtype[np.float64]]] = None
) -> np.ndarray[Any, np.dtype[np.float64]]:
    ...

@overload
def attitude_profile_matrix(
    obs: np.ndarray[Any, np.dtype[np.float64]],
    ref: np.ndarray[Any, np.dtype[np.float64]],
    weights: Optional[np.ndarray[Any, np.dtype[np.float64]]] = None
) -> np.ndarray[Any, np.dtype[np.float64]]:
    ...