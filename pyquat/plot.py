import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from typing import TYPE_CHECKING, Any, Dict, Optional, List

if TYPE_CHECKING:
    from pyquat._pyquat import Quat
else:
    from _pyquat import Quat

def prepare_quaternion_array_for_plotting(
        q_ary: np.ndarray[Any, np.dtype[np.object_]],
        rotate_axis: str = 'x'
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
    """ Converting all attitudes to points on a sphere """
    ary = None
    if len(q_ary.shape) == 1:
        ary = np.empty((3, q_ary.shape[0]))
        for i, q in enumerate(q_ary):
            ary[:,i] = q.to_unit_vector(rotate_axis)[:,0]
    elif len(q_ary.shape) == 2:
        ary = np.empty((3, q_ary.shape[1]))
        for i, q in enumerate(q_ary):
            ary[:,i] = q[0].to_unit_vector(rotate_axis)[:,0]
    else:
        raise ValueError("expected 1- or 2-D array")
    return ary


def scatter(
        q_ary: np.ndarray[Any, np.dtype[np.object_]],
        rotate_axis: str = 'x',
        fig: Optional[mpl.figure.Figure] = None,
        axes: Optional[mpl.axes.Axes] = None,
        scatter_kwargs: Dict[str,Any] = {},
    ) -> mpl.collections.PathCollection:
    """ Plot an array of quaternions as scatter points on a sphere """

    if fig is None:
        fig = plt.figure()
    if axes is None:
        axes = fig.add_subplot(111, projection='3d')

    ary = prepare_quaternion_array_for_plotting(q_ary, rotate_axis = rotate_axis)

    return axes.scatter(ary[0,:], ary[1,:], ary[2,:], **scatter_kwargs)


def plot(
        q_ary: np.ndarray[Any, np.dtype[np.object_]],
        t: Optional[np.ndarray[Any, np.dtype[np.float64]]] = None,
        rotate_axis: str = 'x',
        fig: Optional[mpl.figure.Figure] = None,
        axes: Optional[mpl.axes.Axes] = None,
        plot_kwargs: Dict[str,Any] = {}
    ) -> List[mpl.lines.Line2D]:
    """
    Plot an array of quaternions using lines on the surface of a sphere.

    Optionally, you can provide a t argument to plot as a timeseries,
    where the lowest t is displayed red, and the highest t is
    displayed blue.

    """

    if fig is None:
        fig = plt.figure()
    if axes is None:
        axes = fig.add_subplot(111, projection='3d')

    ary = prepare_quaternion_array_for_plotting(q_ary, rotate_axis=rotate_axis)

    if t is not None:
        tc = (t - t.min()) / (t.max() - t.min())
        c = np.vstack((tc, np.zeros_like(tc), -tc + 1.0))

        lines: List[mpl.lines.Line2D] = []
        for i in range(1,ary.shape[1]):
            lines.extend(axes.plot(ary[0,i-1:i+1], ary[1,i-1:i+1], ary[2,i-1:i+1], c = tuple(c[:,i]), **plot_kwargs))
        return lines
        
    else:
        return axes.plot(ary[0,:], ary[1,:], ary[2,:], **plot_kwargs)

def plot_frame(
        q: Quat,
        r: np.ndarray[Any, np.dtype[np.float64]] = np.zeros((3,1)),
        fig: Optional[mpl.figure.Figure] = None,
        axes: Optional[mpl.axes.Axes] = None,
        axis_size: float = 1.0
    ) -> mpl.axes.Axes:
    """
    Plot a quaternion as a coordinate frame, with red, green, and blue 
    referring to x, y, and z.

    Also accepts an optional position to move the frame to (post-rotation).
    """
    if fig is None:
        fig = plt.figure()
    if axes is None:
        axes = fig.add_subplot(111, projection='3d')

    xh = np.zeros((3,2))
    yh = np.zeros_like(xh)
    zh = np.zeros_like(xh)
    xh[0,1] = axis_size
    yh[1,1] = axis_size
    zh[2,1] = axis_size

    # Transform the frame (rotate it and then move it)
    T = q.to_matrix()
    txh = np.dot(T, xh) + r
    tyh = np.dot(T, yh) + r
    tzh = np.dot(T, zh) + r
    
    axes.plot(txh[0,:], txh[1,:], txh[2,:], c='r')
    axes.plot(tyh[0,:], tyh[1,:], tyh[2,:], c='g')
    axes.plot(tzh[0,:], tzh[1,:], tzh[2,:], c='b')
    
    return axes
