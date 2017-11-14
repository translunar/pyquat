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

def propagate_additively(q, w, dt):
    """Change a quaternion q by some angular velocity w over some small
    timestep dt, using additive propagation (q1 = q0 + dq/dt * dt)"""
    
    q_vector = q.to_vector()
    q_vector += qdot(q_vector, w) * dt
    return Quat(*q_vector)    

    
def cov(ary):

    """
    Compute the covariance of an array of quaternions, where each
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

