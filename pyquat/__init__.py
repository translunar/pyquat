from _pyquat import *
    
import math
import numpy as np
from scipy import linalg
import warnings

QUAT_SMALL = 1e-8


def fromstring(*args, **kwargs):
    """
    Shortcut for pyquat.Quat.from_vector(numpy.fromstring()).  If you
    don't provide a 'sep' argument, this method will supply the
    argument count=4 to numpy.fromstring() regardless of what you
    provided for it.
    """
    if 'sep' in kwargs and kwargs['sep'] == '':
        kwargs['count'] = 4
        
    return Quat(*(np.fromstring(*args, **kwargs)))

def qdot(q, w, big_w = None):
    """
    Compute dq/dt given some angular velocity w and initial quaternion q.
    """
    if big_w is None:
        big_w = big_omega(w)
    if isinstance(q, Quat):
        return np.dot(big_w * 0.5, q.to_vector())
    else:
        return np.dot(big_w * 0.5, q)

def wdot(w, J, J_inv = None):
    """
    Compute dw/dt given some angular velocity w and moment of inertia J.
    """
    if J_inv is None:
        J_inv = linalg.inv(J)
    return np.dot(J_inv, np.dot(skew(np.dot(J, w)), w))

def state_transition_matrix(w, big_w = None):
    """
    Generate a state transition matrix for a quaternion based on some
    angular velocity w.
    """
    if big_w is None:
        big_w = big_omega(w)
    return big_w * 0.5

def change(*args, **kwargs):
    warnings.warn("deprecated", DeprecationWarning)
    return propagate(*args, **kwargs)

def propagate(q, w, dt):
    """
    Change a quaternion q by some angular velocity w over some small
    timestep dt.
    """

    # Find magnitude of angular velocity (in r/s)
    w_norm = linalg.norm(w)
    if w_norm < QUAT_SMALL:
        return q.copy()
    return Quat(*(np.dot(expm(w, dt), q.to_vector())))

def matrix_propagate(T, w, dt, r = 1):
    """Propagate an attitude matrix T forward by some angular velocity w
    over time step dt. This method uses a Taylor expansion of degree r
    where r is between 1 and 4 inclusive.

    Note that there's minimal computational difference between orders
    2 and 3, as each involves a second 3x3 matrix multiplication. The
    step to 4th order is also quite small, involving only an
    additional vector dot product. In most cases, you will probably
    want to use r = 1 or r = 4.

    Args:
        T:  transformation matrix (3x3)
        w:  angular velocity vector (length 3)
        dt: time step size
        r:  Taylor expansion degree (between 1 and 4 inclusive)

    Returns:
        A 3x3 matrix giving the updated transformation.
    """

    wt   = w*dt
    wtx  = skew(wt)
    
    exp  = np.identity(3) + wtx
    if r >= 2:
        wtx2 = wtx.dot(wtx)
        if r == 2:
            exp += wtx2 * 0.5
        elif r >= 3:
            if r == 3:
                exp += wtx2 * (0.5 - wt / 6.0)
            elif r == 4:
                wt2   = wt.T.dot(wt)
                exp += wtx2 * (0.5 - wt / 6.0 - wt2 / 24.0)
            else:
                raise(NotImplemented, "degree must be between 1 and 4 inclusive")
        else:
            raise(NotImplemented, "degree must be between 1 and 4 inclusive")

    return exp.T.dot(T)
        
    

def propagate_additively(q, w, dt):
    """Change a quaternion q by some angular velocity w over some small
    timestep dt, using additive propagation (q1 = q0 + dq/dt * dt)"""
    
    q_vector = q.to_vector()
    q_vector += qdot(q_vector, w) * dt
    return Quat(*q_vector)

    
def cov(ary):
    """Compute the covariance of an array of quaternions, where each
    column represents a quaternion.
    """
    # If the user supplies an array of N quaternions, convert it to a 4xN array,
    # since we need it in this form to get its covariance.
    if ary.dtype == np.dtype(Quat):
        a = np.empty((4, max(ary.shape)), dtype=np.double)
        q_ary = ary.T
        for i, q in enumerate(q_ary.flatten()):
            a[:,i] = q.to_vector()[:,0]
        ary = a
            
    # Compute the covariance of the supplied quaternions.
    return np.cov(ary)

def mean(ary, covariance = None):
    """
    Compute the average quaternion using Markey, Cheng, Craissidis, and Oshman (2007)
    
    This method takes a 4xN array and computes the average using eigenvalue decomposition.
    """
    if covariance == None:
        covariance = cov(ary)

    # Compute their eigenvalues and eigenvectors
    eigenvalues, eigenvectors = linalg.eig(covariance)
    max_index = np.argmax(eigenvalues)
    q = eigenvectors[max_index]
    mean = Quat(q[0], q[1], q[2], q[3])
    mean.normalize()
    return mean

def mean_and_cov(ary):
    c = cov(ary)
    m = mean(ary, covariance=c)
    return (m,c)

def angle_vector_cov(ary):
    """
    Compute the covariance of an array of quaternions, like cov(), except use the attitude vector
    representation of each.
    """

    if ary.dtype == np.dtype(Quat):
        a = np.empty((3, max(ary.shape)), dtype=np.double)
        q_ary = ary.T
        for i, q in enumerate(q_ary.flatten()):
            a[:,i] = q.to_angle_vector()[:,0]
        ary = a
    elif ary.dtype == np.double:
        a = np.empty((3, ary.shape[1]), dtype=np.double)
        q_ary = ary.T
        for i, q in enumerate(q_ary):
            a[:,i] = Quat(q[0], q[1], q[2], q[3]).to_angle_vector()[:,0]
        ary = a

    return np.cov(ary)

def from_rotation_vector(v):
    """
    Shortcut for Quat.from_rotation_vector(v).
    """
    return Quat.from_rotation_vector(v)

def from_matrix(m):
    """
    Shortcut for Quat.from_matrix(v).
    """
    return Quat.from_matrix(m)

def uv_to_matrix(u, v):
    """Construct a transformation matrix from a pair of two vectors (which must not
    be parallel).

    Reference:
        * Wertz (1978). Spacecraft Attitude Determination and Control. p. 424.
    
    Args:
        u:  length 3 vector
        v:  length 3 vector

    Returns:
        A 3x3 transformation matrix that rotates vectors from the original frame
    to the one created by the two vectors.
    """

    # Normalize both inputs
    uh = u / linalg.norm(u) # q in Eq. 12-39a
    vh = v / linalg.norm(v)

    if np.abs(np.dot(uh, vh)) - 1.0 <= QUAT_SMALL:
        raise ArgumentError()

    rh = np.cross(uh, vh)
    rh /= linalg.norm(rh) # Eq. 12-39b
    sh = np.cross(uh, rh) # Eq. 12-39c
    sh /= linalg.norm(sh)

    M = np.stack((uh, rh, sh))
    return M


def ref_obs_to_matrix(ur, vr, u, v):
    """Construct a transformation matrix that maps from the reference frame to the
    frame formed by the observations. Each pair of vectors is passed to the method
    uv_to_matrix().
    
    Reference:
        * Wertz (1978). Spacecraft Attitude Determination and Control. pp. 424--5.

    Args:
        ur: length 3 reference vector
        vr: length 3 reference vector
        u:  length 3 observation vector
        v:  length 3 observation vector

    Returns:
        A 3x3 transformation matrix rotating from the reference frame to the
    observation frame.
    """
    T_orig_to_obs = uv_to_matrix(u, v)
    T_ref_to_orig = uv_to_matrix(ur, vr).T
    return T_orig_to_obs.dot(T_ref_to_orig)


def from_uv(*args):
    """Turn vectors u, v into a transformation matrix and from there into
    a quaternion. See also: uv_to_matrix().
    """
    return from_matrix(uv_to_matrix(*args))


def from_ref_obs(*args):
    """See ref_obs_to_matrix(). Produces a quaternion instead of a matrix."""
    return from_matrix(ref_obs_to_matrix(*args))


def step_rk4(q, w, dt, w_dynamics = None, q_dynamics = qdot, J = None, J_inv = None):
    """
    Use a standard Runge-Kutta 4-step / 4th-order integration to step
    the quaternion forward in time.
    """
    if linalg.norm(w) <= QUAT_SMALL:
        return (q, w)
    
    #C1 = 0.0
    #C2 = 0.5
    #C3 = 0.5
    #C4 = 1.0
    A21 = 0.5
    A32 = 0.5 #A31 = 0.0
    #A43 = 1.0; A42 = A41 = 0.0
    #B1 = 1/6.0
    #B2 = 1/3.0
    #B3 = B2
    #B4 = B1

    q1  = q.to_vector()
    
    if w_dynamics is None:
        qk1 = q_dynamics(q1, w)
        q2  = q1 + qk1 * A21 * dt
        qk2 = q_dynamics(q2, w)
        q3  = q1 + qk2 * A32 * dt # + qk1 * A31 * dt
        qk3 = q_dynamics(q3, w)
        q4  = q1 + qk3 * dt # + A42 * qk2 * dt + A41 * qk1 * dt
        qk4 = q_dynamics(q4, w)

        w_next = np.array(w)
    else:
        if J is None:
            J = np.identity(3)
            J_inv = J
        elif J_inv is None:
            J_inv = linalg.inv(J)
                 
        w1  = w # + C1 * dt
        wk1 = w_dynamics(w1, J, J_inv)
        qk1 = q_dynamics(q1, w1) #q = q1, w = w1

        q2  = q1 + qk1 * A21 * dt
        w2  = w  + wk1 * A21 * dt
        wk2 = w_dynamics(w2, J, J_inv)
        qk2 = q_dynamics(q2, w2)


        q3  = q1 + qk2 * A32 * dt # + qk1 * A31 * dt
        w3  = w  + wk2 * A32 * dt # + wk1 * A31 * dt
        wk3 = w_dynamics(w3, J, J_inv)
        qk3 = q_dynamics(q3, w3)

        q4  = q1 + qk3 * dt # + A42 * qk2 * dt + A41 * qk1 * dt
        w4  = w  + wk3 * dt # same
        wk4 = w_dynamics(w4, J, J_inv)
        qk4 = q_dynamics(q4, w4)
        
        w_next = w1 + dt * (wk1 + wk2*2 + wk3*2 + wk4) / 6.0

    q_next = q1 + dt * (qk1 + qk2*2 + qk3*2 + qk4) / 6.0
    
    return (Quat(*q_next).normalized(), w_next)

def step_cg3(
        q,
        w,
        dt,
        w_dynamics                = None,
        q_state_transition_matrix = state_transition_matrix,
        J                         = None,
        J_inv                     = None):
    """
    Use a 3-stage, third-order Crouch-Grossman integration for 
    propagating a quaternion and a Runge-Kutta integration for 
    propagating angular velocity.

    This method returns a tuple containing the resulting quaternion
    and omega.
    """
    if linalg.norm(w) <= QUAT_SMALL:
        return (q, w)
    
    B1 = 13/51.0
    B2 = -2/3.0
    B3 = 24/17.0
    A21 = 0.75
    A31 = 119/216.0
    A32 = 17/108.0
    C1 = 0.0
    C2 = 3/4.0
    C3 = 17/24.0

    q1  = q.to_vector()
    
    if w_dynamics is None:
        w_next = np.array(w)
        
        expm3 = expm(w, dt * B3)
        expm2 = expm(w, dt * B2)
        expm1 = expm(w, dt * B1)        

    else:
        if J is None:
            J = np.identity(3)
            J_inv = J
        elif J_inv is None:
            J_inv = linalg.inv(J)
        
        w1  = w
        wk1 = w_dynamics(w1, J, J_inv)

        w2  = w1 + A21 * wk1 * dt
        wk2 = w_dynamics(w2, J, J_inv)

        w3  = w1 + (wk1 * A31 + wk2 * A32) * dt
        wk3 = w_dynamics(w3, J, J_inv)

        expm3 = expm(w3, dt * B3)
        expm2 = expm(w2, dt * B2)
        expm1 = expm(w1, dt * B1)

        w_next = w + (wk1*B1 + wk2*B2 + wk3*B3) * dt
    
    q_next = np.dot(expm3, np.dot(expm2, np.dot(expm1, q1)))

    return (Quat(*q_next), w_next)

def step_cg4(
        q,
        w,
        dt,
        w_dynamics                = None,
        q_state_transition_matrix = state_transition_matrix,
        J                         = None,
        J_inv                     = None):
    """
    Use a 5-stage, fourth-order Crouch-Grossman integration for 
    propagating a quaternion and a Runge-Kutta integration for 
    propagating angular velocity.

    This method returns a tuple containing the resulting quaternion
    and omega.
    """
    if linalg.norm(w) <= QUAT_SMALL:
        return (q, w)
    
    B1 =   0.1370831520630755
    B2 =  -0.0183698531564020
    B3 =   0.7397813985370780
    B4 =  -0.1907142565505889
    B5 =   0.3322195591068374
    A21 =  0.8177227988124852
    A31 =  0.3199876375476427
    A32 =  0.0659864263556022
    A41 =  0.9214417194464946
    A42 =  0.4997857776773573
    A43 = -1.0969984448371582
    A51 =  0.3552358559023322
    A52 =  0.2390958372307326
    A53 =  1.3918565724203246
    A54 = -1.1092979392113565
    C1  =  0.0
    C2  =  0.8177227988124852
    C3  =  0.3859740639032449
    C4  =  0.3242290522866937
    C5  =  0.8768903263420429

    q1  = q.to_vector()
    
    if w_dynamics is None: # don't propagate omega
        expm5 = expm(w, dt * B5)
        expm4 = expm(w, dt * B4)
        expm3 = expm(w, dt * B3)
        expm2 = expm(w, dt * B2)
        expm1 = expm(w, dt * B1)
    
        w_next = np.array(w)
        
    else:
        if J is None:
            J = numpy.identity(3)
            J_inv = J
        elif J_inv is None:
            J_inv = linalg.inv(J)

        w1  = w
        wk1 = w_dynamics(w1, J, J_inv)

        w2  = w1 + A21 * wk1 * dt
        wk2 = w_dynamics(w2, J, J_inv)

        w3  = w1 + (wk1 * A31 + wk2 * A32) * dt
        wk3 = w_dynamics(w3, J, J_inv)

        w4  = w1 + (wk1 * A41 + wk2 * A42 + wk3 * A43) * dt
        wk4 = w_dynamics(w4, J, J_inv)
        
        w5  = w1 + (wk1 * A51 + wk2 * A52 + wk3 * A53 + wk4 * A54) * dt
        wk5 = w_dynamics(w5, J, J_inv)

        expm5 = expm(w5, dt * B5)
        expm4 = expm(w4, dt * B4)
        expm3 = expm(w3, dt * B3)
        expm2 = expm(w2, dt * B2)
        expm1 = expm(w1, dt * B1)
    
        w_next = w + (wk1*B1 + wk2*B2 + wk3*B3 + wk4*B4 + wk5*B5) * dt

    # propagate the quaternion
    q_next = np.dot(expm5,
                    np.dot(expm4,
                           np.dot(expm3,
                                  np.dot(expm2,
                                         np.dot(expm1, q1)))))


    return (Quat(*q_next), w_next)


def interp(x, xp, fp,
           lerp_threshold = 1.0,
           left           = 'slerp',
           right          = 'slerp'):
    """Corresponds roughly to numpy.interp. Attempts to interpolate the
    quaternion series qp occurring at independent variable series tp (e.g.
    times) at other values given by t.

    Note that left and right, when set to 'slerp', ignore lerp_threshold.

    Args:
        x:               time or times at which to interpolate
        xp:              times corresponding to qp values
        fp:              quaternion series
        lerp_threshold:  dot product threshold for deciding between lerp
                         and slerp (default is 1.0, always uses slerp; for
                         always use lerp, choose 0.0)
        left:            'lerp' or 'slerp' (default 'slerp')
        right:           'lerp' or 'slerp' (default 'slerp')

    Returns:
        If x is a single value, returns a single quaternion. Otherwise,
        returns a list of quaternions of same size as x.
    """

    indices = np.searchsorted(xp, x)
    f       = []

    if isinstance(x, (float, int)):
        x = [x]

    jj = 0
    for ii in indices:
        if ii == 0: # <= xp (left)
            t = (x[jj] - xp[0]) / (xp[1] - xp[0])
            if left == 'slerp':
                f.append(fp[0].slerp(fp[1], t, lerp_threshold = 1.0))
            else:
                f.append(fp[0].lerp(fp[1], t))

        elif ii == len(fp): # >= xp (right)
            t = (x[jj] - xp[ii-2]) / (xp[ii-1] - xp[ii-2])
            if right == 'slerp':
                f.append(fp[ii-2].slerp(fp[ii-1], t, lerp_threshold = 1.0))
            else:
                f.append(fp[ii-2].lerp(fp[ii-1], t))

        else: # all others
            t = (x[jj] - xp[ii-1]) / (xp[ii] - xp[ii-1])
            f.append(fp[ii-1].slerp(fp[ii], t, lerp_threshold = lerp_threshold))

        jj += 1

    return f
