import numpy as np
from typing import Any, Optional, overload, TYPE_CHECKING

if TYPE_CHECKING:
    class Quat: ...

@overload
def qdot(
        q: np.ndarray[Any, np.dtype[np.float64]],
        w: np.ndarray[Any, np.dtype[np.float64]],
        big_w: Optional[np.ndarray[Any, np.dtype[np.float64]]] = None
    ) -> np.ndarray[Any, np.dtype[np.float64]]: ...

@overload
def qdot(
        q: Quat,
        w: np.ndarray[Any, np.dtype[np.float64]],
        big_w: Optional[np.ndarray[Any, np.dtype[np.float64]]] = None
    ) -> np.ndarray[Any, np.dtype[np.float64]]: ...

@overload
def angle_vector_cov(
        ary: np.ndarray[Any, np.dtype[np.float64]]
    ) -> np.ndarray[Any, np.dtype[np.float64]]: ...

@overload
def angle_vector_cov(
        ary: np.ndarray[Any, np.dtype[np.object_]]
    ) -> np.ndarray[Any, np.dtype[np.float64]]: ...

@overload
def cov(
        ary: np.ndarray[Any, np.dtype[np.float64]]
    ) -> np.ndarray[Any, np.dtype[np.float64]]: ...

@overload
def cov(
        ary: np.ndarray[Any, np.dtype[np.object_]]
    ) -> np.ndarray[Any, np.dtype[np.float64]]: ...