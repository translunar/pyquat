#include "pyquat.h"

/*
 * Local function declarations.
 */
static int       pyquat_Quat_init(pyquat_Quat* self, PyObject* args);
static PyObject* pyquat_Quat_from_rotation_vector(PyObject* self, PyObject* args);
static PyObject* pyquat_Quat_from_angle_axis(PyObject* type, PyObject* args, PyObject* kwargs);
static PyObject* pyquat_Quat_from_matrix(PyObject* type, PyObject* args);
static PyObject* pyquat_Quat_repr(PyObject* self);
static PyObject* pyquat_Quat_mul(PyObject* self, PyObject* args);
static PyObject* pyquat_Quat_inplace_normalize(PyObject* self);
static PyObject* pyquat_Quat_inplace_normalize_large(PyObject* self);
static PyObject* pyquat_Quat_inplace_conjugate(PyObject* self);
static PyObject* pyquat_Quat_normalize(PyObject* self);
static PyObject* pyquat_Quat_normalize_large(PyObject* self);
static PyObject* pyquat_Quat_conjugate(PyObject* self);
static PyObject* pyquat_Quat_to_rotation_vector(PyObject* self);
static void      to_matrix(pyquat_Quat* q, double* T);
static PyObject* pyquat_Quat_to_matrix(PyObject* self);
static PyObject* pyquat_Quat_to_unit_vector(PyObject* self, PyObject* args);
static PyObject* pyquat_Quat_to_vector(PyObject* self);
static PyObject* pyquat_Quat_to_big_xi_matrix(PyObject* self);
static PyObject* pyquat_Quat_copy(PyObject* self);
static PyObject* pyquat_identity(PyObject* self);
static int       pyquat_Quat_compare(PyObject* left, PyObject* right);
static PyObject* pyquat_Quat_tobytes(PyObject* self, PyObject* args, PyObject* kwargs);
static PyObject* pyquat_rotation_vector_to_matrix(PyObject* self, PyObject* args);
static PyObject* pyquat_big_omega(PyObject* self, PyObject* args);
static PyObject* pyquat_skew(PyObject* self, PyObject* args);
static PyObject* pyquat_expm(PyObject* self, PyObject* args);
static PyObject* pyquat_Quat_lerp(PyObject* self, PyObject* args);
static PyObject* pyquat_Quat_slerp(PyObject* self, PyObject* args, PyObject* kwargs);
static PyObject* pyquat_Quat_dot(PyObject* self, PyObject* args);
static PyObject* pyquat_Quat_rotate(PyObject* self, PyObject* args);
static PyObject* pqw_valenti_q_acc(PyObject* self, PyObject* arg);
static PyObject* pqw_valenti_q_mag(PyObject* self, PyObject* arg);
static PyObject* pqw_valenti_dq_acc(PyObject* self, PyObject* arg);
static PyObject* pqw_valenti_dq_mag(PyObject* self, PyObject* arg);

static int not_double_array(PyArrayObject* v) {
  if (v->descr->type_num != NPY_DOUBLE) {
    PyErr_SetString(PyExc_ValueError,
                    "array must be of type Float.");
    return 1;
  }
  return 0;
}


static int not_double_vector(PyArrayObject* v) {
  if (v->descr->type_num != NPY_DOUBLE || v->nd != 1) {
    PyErr_SetString(PyExc_ValueError,
                    "array must be of type Float and 1-dimensional (n).");
    return 1;
  }
  return 0;
}


static int not_double_matrix(PyArrayObject* m) {
  if (m->descr->type_num != NPY_DOUBLE || m->nd != 2) {
    PyErr_SetString(PyExc_ValueError,
                    "array must be of type Float and 2-dimensional (n x m).");
    return 1;
  }
  return 0;
}


/** @brief Check that v is 3 in length.
 */
static int not_length_3(PyArrayObject* v) {
  if (v->dimensions[0] != 3) {
    PyErr_SetString(PyExc_ValueError,
                    "array must be length 3");
    return 1;
  }
  return 0;
}

/** @brief Check that v is 2 in length.
 */
static int not_length_2(PyArrayObject* v) {
  if (v->dimensions[0] != 2) {
    PyErr_SetString(PyExc_ValueError,
                    "array must be length 2");
    return 1;
  }
  return 0;
}

/** @brief Check that m is exactly 3x3. Assumes you've already
 **        checked that it is a matrix.
 */
static int not_3x3(PyArrayObject* m) {
  if (m->dimensions[0] != 3 || m->dimensions[1] != 3) {
    PyErr_SetString(PyExc_ValueError,
                    "array must be 3x3");
    return 1;
  }
  return 0;
}


static int not_3x1(PyArrayObject* m) {
  if (m->dimensions[0] != 3 || m->dimensions[1] != 1) {
    PyErr_SetString(PyExc_ValueError,
                    "array must be 3x1");
    return 1;
  }
  return 0;
}


static int not_double_matrix_or_vector(PyArrayObject* m) {
  if (m->descr->type_num != NPY_DOUBLE || m->nd > 2) {
    PyErr_SetString(PyExc_ValueError,
                    "array must be of type Float and 1- or 2-dimensional (n or n x m).");
    return 1;
  }
  return 0;
}


static int not_double_nx1_or_length_n(PyArrayObject* m, int n) {
  if (m->descr->type_num != NPY_DOUBLE) {
    PyErr_SetString(PyExc_ValueError,
                    "array must be of type Float");
    return 1;
  } else if (m->dimensions[0] != n &&
             (m->nd == 1 || (m->nd == 2 && m->dimensions[1] != 1))) {
    PyErr_SetString(PyExc_ValueError,
                    "array must be a vector of length n or matrix of shape nx1");
    return 1;
  }
  return 0;
}


static PyMethodDef pyquat_methods[] = {
  /* pyquat namespace */				     
  {"identity", (PyCFunction)pyquat_identity, METH_NOARGS, "create an identity quaternion (1.0, 0.0, 0.0, 0.0)"},
  {"rotation_vector_to_matrix", (PyCFunction)pyquat_rotation_vector_to_matrix, METH_VARARGS, "convert a rotation vector directly to a directed-cosine matrix, skipping the quaternion"},
  {"big_omega", (PyCFunction)pyquat_big_omega, METH_VARARGS, "compute the 4x4 Omega matrix for some angular velocity"},
  {"skew", (PyCFunction)pyquat_skew, METH_VARARGS, "compute the 3x3 cross-product (skew symmetric) matrix for some vector"},
  {"expm", (PyCFunction)pyquat_expm, METH_VARARGS, "compute the 4x4 matrix exponential for quaternion propagation for some angular velocity and time step"},
  /* pyquat.wahba.valenti namespace */
  {"valenti_q_mag", (PyCFunction)pqw_valenti_q_mag, METH_O, "compute a quaternion mapping from a z-down frame to an x-magnetic-north/z-down frame"},
  {"valenti_q_acc", (PyCFunction)pqw_valenti_q_acc, METH_O, "compute a quaternion mapping from an arbitrary unknown frame to a z-down frame"},
  {"valenti_dq_mag", (PyCFunction)pqw_valenti_dq_mag, METH_O, "compute a correction quaternion mapping from a z-down frame to an x-magnetic-north/z-down frame"},
  {"valenti_dq_acc", (PyCFunction)pqw_valenti_dq_acc, METH_O, "compute a correction quaternion mapping from an arbitrary unknown frame to a z-down frame"},    
  {NULL, NULL, 0, NULL}  /* Sentinel */
};


static PyMemberDef pyquat_Quat_members[] = {
  {"s",  T_DOUBLE, offsetof(pyquat_Quat, s),                    0, "scalar component"  },
  {"vx", T_DOUBLE, offsetof(pyquat_Quat, v),                    0, "vector x component"},
  {"vy", T_DOUBLE, offsetof(pyquat_Quat, v) + sizeof(double),   0, "vector y component"},
  {"vz", T_DOUBLE, offsetof(pyquat_Quat, v) + 2*sizeof(double), 0, "vector z component"},
  {"w",  T_DOUBLE, offsetof(pyquat_Quat, s),                    0, "scalar component"  },
  {"x",  T_DOUBLE, offsetof(pyquat_Quat, v),                    0, "vector x component"},
  {"y",  T_DOUBLE, offsetof(pyquat_Quat, v) + sizeof(double),   0, "vector y component"},
  {"z",  T_DOUBLE, offsetof(pyquat_Quat, v) + 2*sizeof(double), 0, "vector z component"},
  {NULL}  /* Sentinel */
};


static PyMethodDef pyquat_Quat_methods[] = {
  {"to_rotation_vector", (PyCFunction)pyquat_Quat_to_rotation_vector, METH_NOARGS, "convert to a unit axis divided by the angle of rotation in radians"},
  {"from_angle_axis", (PyCFunction)pyquat_Quat_from_angle_axis, METH_CLASS | METH_VARARGS | METH_KEYWORDS, "create a quaternion from an angle and an axis of rotation"},
  {"from_rotation_vector", (PyCFunction)pyquat_Quat_from_rotation_vector, METH_CLASS | METH_VARARGS, "create a quaternion from a non-unit axis, treating its magnitude as the angle about the normalized axis"},  
  {"from_matrix", (PyCFunction)pyquat_Quat_from_matrix, METH_CLASS | METH_VARARGS, "create a quaternion from a 3x3 directional cosine matrix"},
  {"to_matrix", (PyCFunction)pyquat_Quat_to_matrix, METH_NOARGS, "convert to a transformation matrix"},
  {"to_vector", (PyCFunction)pyquat_Quat_to_vector, METH_NOARGS, "convert to a 4x1 vector"},
  {"to_unit_vector", (PyCFunction)pyquat_Quat_to_unit_vector, METH_VARARGS, "convert to a 3x1 unit vector representing the attitude on the surface of a unit sphere"},
  {"normalize", (PyCFunction)pyquat_Quat_inplace_normalize, METH_NOARGS, "in-place normalize the quaternion"},
  {"normalize_large", (PyCFunction)pyquat_Quat_inplace_normalize_large, METH_NOARGS, "in-place normalize the quaternion, avoiding overflow"},
  {"conjugate", (PyCFunction)pyquat_Quat_inplace_conjugate, METH_NOARGS, "in-place conjugate the quaternion"},
  {"normalized", (PyCFunction)pyquat_Quat_normalize, METH_NOARGS, "normalize the quaternion"},
  {"normalized_large", (PyCFunction)pyquat_Quat_normalize_large, METH_NOARGS, "normalize the quaternion, avoiding overflow"},
  {"conjugated", (PyCFunction)pyquat_Quat_conjugate, METH_NOARGS, "copy and conjugate the quaternion"},
  {"copy", (PyCFunction)pyquat_Quat_copy, METH_NOARGS, "copy the quaternion"},  
  {"tobytes", (PyCFunction)pyquat_Quat_tobytes, METH_VARARGS | METH_KEYWORDS, "equivalent of numpy.ndarray.tobytes(), but for pyquat.Quat"},
  {"lerp", (PyCFunction)pyquat_Quat_lerp, METH_VARARGS, "linear interpolation between two quaternions"},
  {"slerp", (PyCFunction)pyquat_Quat_slerp, METH_VARARGS | METH_KEYWORDS, "spherical linear interpolation or linear interpolation between two quaternions depending on whether the dot product exceeds the parameter 'lerp_threshold'"},
  {"dot", (PyCFunction)pyquat_Quat_dot, METH_VARARGS, "dot product of two quaternions as if they are 4D vectors"},
  {"rotate", (PyCFunction)pyquat_Quat_rotate, METH_VARARGS, "rotate a vector"},
  {"big_xi", (PyCFunction)pyquat_Quat_to_big_xi_matrix, METH_NOARGS, "build a Xi matrix"},
  {NULL, NULL, 0, NULL}  /* Sentinel */
};


static PyNumberMethods pyquat_Quat_as_number = {
  0,                  /*nb_add*/
  0,                  /*nb_subtract*/
  pyquat_Quat_mul,    /*nb_multiply*/
  0,                  /*nb_divide*/
  0,                  /*nb_remainder*/
  0,                  /*nb_divmod*/
  0,                  /*nb_power*/
  0,                  /*nb_negative*/
  0,                  /*nb_positive*/
  0,                  /*nb_absolute*/
  0,                  /*nb_nonzero*/
  0,                  /*nb_invert*/
  0,                  /*nb_lshift*/
  0,                  /*nb_rshift*/
  0,                  /*nb_and*/
  0,                  /*nb_xor*/
  0,                  /*nb_or*/
  0,                  /*nb_coerce*/
  0,                  /*nb_int*/
  0,                  /*nb_long*/
  0,                  /*nb_float*/
  0,                  /* nb_oct */
  0,                  /* nb_hex */
  0,                  /* nb_inplace_add */
  0,                  /* nb_inplace_subtract */
  0,                  /* nb_inplace_multiply */
  0,                  /* nb_inplace_divide */
  0,                  /* nb_inplace_remainder */
  0,                  /* nb_inplace_power */
  0,                  /* nb_inplace_lshift */
  0,                  /* nb_inplace_rshift */
  0,                  /* nb_inplace_and */
  0,                  /* nb_inplace_xor */
  0,                  /* nb_inplace_or */
  0,                  /* nb_floor_divide */
  0,                  /* nb_true_divide */
  0,                  /* nb_inplace_floor_divide */
  0,                  /* nb_inplace_true_divide */
};


static PyTypeObject pyquat_QuatType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "pyquat.Quat",            /*tp_name*/
  sizeof(pyquat_Quat),       /*tp_basicsize*/
  0,                         /*tp_itemsize   -- variable-length objects like lists and strings, does not apply */
  0,                         /*tp_dealloc*/
  0,                         /*tp_print*/
  0,                         /*tp_getattr*/
  0,                         /*tp_setattr*/
  &pyquat_Quat_compare,      /*tp_compare*/
  &pyquat_Quat_repr,         /*tp_repr*/
  &pyquat_Quat_as_number,    /*tp_as_number*/
  0,                         /*tp_as_sequence*/
  0,                         /*tp_as_mapping*/
  0,                         /*tp_hash */
  0,                         /*tp_call*/
  0,                         /*tp_str*/
  0,                         /*tp_getattro*/
  0,                         /*tp_setattro*/
  0,                         /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
  "Quaternion type",         /*tp_doc */
  0,                         /*tp_traverse */
  0,                         /*tp_clear */
  0,                         /*tp_richcompare */
  0,                         /*tp_weaklistoffset */
  0,                         /*tp_iter */
  0,                         /*tp_iternext */
  pyquat_Quat_methods,       /*tp_methods */
  pyquat_Quat_members,       /*tp_members */
  0,                         /*tp_getset */
  0,                         /*tp_base */
  0,                         /*tp_dict */
  0,                         /*tp_descr_get */
  0,                         /*tp_descr_set */
  0,                         /*tp_dictoffset */
  (initproc)pyquat_Quat_init,/*tp_init */
  0,                         /*tp_alloc */
  0 //pyquat_Quat_new,           /*tp_new */
};


/* Initialize the pyquat module and add pyquat.Quat to it.
 *
 */
static PyObject* pyquat_init(void) {
  PyObject* m;

  pyquat_QuatType.tp_new = PyType_GenericNew;
  if (PyType_Ready(&pyquat_QuatType) < 0)
    return NULL;  
  
  MOD_DEF(m, "_pyquat",
          "Quaternion module with fast unit (right) quaternion math written in C.",
          pyquat_methods);
  
  if (m == NULL)
    return NULL;

  // Import NumPy to prevent a segfault when we call a function that uses NumPy API.
  import_array1(NULL);

  // Create the Quat class in the pyquat module.
  Py_INCREF(&pyquat_QuatType);
  PyModule_AddObject(m, "Quat", (PyObject *)&pyquat_QuatType);

  return m;
}

#ifdef IS_PY3K
PyMODINIT_FUNC PyInit__pyquat(void) { return pyquat_init(); }
#else
PyMODINIT_FUNC init_pyquat(void) { pyquat_init(); }
#endif


/** @brief Negate a quaternion in-place.
 **
 * @detail This function does no pointer checks. It is generally used
 *         to ensure that output quaternions do not have a negative
 *         scalar component.
 *
 * @param[in,out]  q   quaternion to negate
 */
inline static void negate(pyquat_Quat* q) {
  q->s    = -q->s;
  q->v[0] = -q->v[0];
  q->v[1] = -q->v[1];
  q->v[2] = -q->v[2];

  return;
}


static int pyquat_Quat_init(pyquat_Quat* self, PyObject* args) {
  
  double scalar, vx, vy, vz;
  
  if (!PyArg_ParseTuple(args, "dddd", &scalar, &vx, &vy, &vz))
    return -1;

  // Read the scalar and vector components of the quaternion.
  if (scalar >= 0) {
    self->s    = scalar;
    self->v[0] = vx;
    self->v[1] = vy;
    self->v[2] = vz;
  } else {
    self->s    = -scalar;
    self->v[0] = -vx;
    self->v[1] = -vy;
    self->v[2] = -vz;
  }

  return 0;
}


static PyObject* pyquat_Quat_repr(PyObject* obj) {
  pyquat_Quat* self = (pyquat_Quat*)(obj);
  return PyUnicode_FromFormat("pyquat.Quat(\%s, \%s, \%s, \%s)", 
                              PyOS_double_to_string(self->s, 'g', 17, 0, NULL),
                              PyOS_double_to_string(self->v[0], 'g', 17, 0, NULL),
                              PyOS_double_to_string(self->v[1], 'g', 17, 0, NULL),
                              PyOS_double_to_string(self->v[2], 'g', 17, 0, NULL));
}


/** @brief Quaternion multiplication operation usually represented by 
 **        \f$\mathrm{p}\otimes\mathrm{q}\f$.
 *
 * @detail This function does not do any pointer checks!
 *
 * @param[in]      p       left-hand operand quaternion (pointer)
 * @param[in]      q       right-hand operand quaternion (pointer)
 * @param[in,out]  result  pre-allocated result quaternion (pointer)
 */
inline static void otimes(pyquat_Quat* p, pyquat_Quat* q, pyquat_Quat* result) {
  result->s    = p->s * q->s - (p->v[0] * q->v[0] + p->v[1] * q->v[1] + p->v[2] * q->v[2]);
  result->v[0] = p->s * q->v[0] + q->s * p->v[0] - (p->v[1] * q->v[2] - p->v[2] * q->v[1]);
  result->v[1] = p->s * q->v[1] + q->s * p->v[1] - (p->v[2] * q->v[0] - p->v[0] * q->v[2]);
  result->v[2] = p->s * q->v[2] + q->s * p->v[2] - (p->v[0] * q->v[1] - p->v[1] * q->v[0]);
}


static PyObject * pyquat_Quat_mul(PyObject* self, PyObject* arg) {

  // Expects the one argument to be a pyquat_Quat
  if (!PyObject_IsInstance(arg, (PyObject*)&pyquat_QuatType)) {
    PyErr_SetString(PyExc_ValueError, "expected quaternion");
    return NULL;
  }

  pyquat_Quat* result = (pyquat_Quat*) PyObject_New(pyquat_Quat, &pyquat_QuatType);
  Py_CheckAlloc(result);

  otimes((pyquat_Quat*)self, (pyquat_Quat*)arg, result);

  // restrict to positive scalar portion of the 4D sphere
  if (result->s < 0) negate(result);

  return (PyObject*)(result);
}



/** @brief Quaternion multiplication operation usually represented by 
 **        \f$\mathrm{p}\otimes\mathrm{q}\f$; however, for this version,
 **        the scalar component of the right-hand quaternion is 0.
 *
 * @detail This function does not do any pointer checks!
 *
 * @param[in]      p       left-hand operand quaternion (pointer)
 * @param[in]      vq      right-hand operand quaternion's vector component (pointer)
 * @param[in,out]  result  pre-allocated result quaternion (pointer)
 */
inline static void otimes_vector(pyquat_Quat* p, double* vq, pyquat_Quat* result) {
  result->s    =                  - p->v[0] * vq[0] - p->v[1] * vq[1] - p->v[2] * vq[2];
  result->v[0] =  p->s * vq[0]    - p->v[1] * vq[2] + p->v[2] * vq[1];
  result->v[1] =  p->s * vq[1]    - p->v[2] * vq[0] + p->v[0] * vq[2];
  result->v[2] =  p->s * vq[2]    - p->v[0] * vq[1] + p->v[1] * vq[0];
}


/** @brief Quaternion multiplication operation usually represented by 
 **        \f$\mathrm{p}\otimes\mathrm{q}\f$; however, for this version,
 **        the scalar component of the output quaternion is 0, and
 **        we treat q as conjugated.
 *
 * @detail This function does not do any pointer checks!
 *
 * @param[in]      p       left-hand operand quaternion (pointer)
 * @param[in]      q       right-hand operand quaternion (pointer) which
 *                         we treat as conjugated
 * @param[in,out]  result  pre-allocated result vector of size 3 (pointer)
 */
inline static void otimes_conj_vector_out(pyquat_Quat* p, pyquat_Quat* q, double* result) {
  result[0] = -p->s * q->v[0] + q->s * p->v[0] + p->v[1] * q->v[2] - p->v[2] * q->v[1];
  result[1] = -p->s * q->v[1] + q->s * p->v[1] + p->v[2] * q->v[0] - p->v[0] * q->v[2];
  result[2] = -p->s * q->v[2] + q->s * p->v[2] + p->v[0] * q->v[1] - p->v[1] * q->v[0];
}



/** @brief Rotate a vector using a quaternion using the operation 
 **        \f$\mathrm{q}\otimes\mathrm{v}_q\otimes\mathrm{q}^{-1}\f$,
 **        where \f$\mathrm{v}_q\f$ is the 3D vector written as a quaternion
 **        (with 0 in the scalar component).
 *
 * @detail This function does not do any pointer checks!
 *
 * FIXME: This function needs to be fixed.
 *
 * @param[in]      p       left-hand operand quaternion (pointer)
 * @param[in]      qv      operand 3-component vector (pointer)
 * @param[in,out]  result  pre-allocated result 3-component vector (pointer)
 */
inline static void rotate_vector(pyquat_Quat* p, double* qv, double* result) {

  result[0] = (p->v[0]*p->v[0] + p->s*p->s - p->v[2]*p->v[2] + p->v[1]*p->v[1]) * qv[0]
    + 2 * p->s * p->v[2] * qv[1];
  
  /*result[0] = (p->s*p->s + p->v[0]*p->v[0] - p->v[1]*p->v[1] - p->v[2]*p->v[2]) * qv[0]
    + 2*(p->v[0]*p->v[1] - p->s*p->v[2]) * qv[1]
    + 2*(p->v[0]*p->v[2] + p->s*p->v[1]) * qv[2];*/
  
  result[1] = 2*(p->v[0]*p->v[1] + p->s*p->v[2]) * qv[0]
    + (p->s*p->s - p->v[0]*p->v[0] + p->v[1]*p->v[1] - p->v[2]*p->v[2]) * qv[1]
    + 2*(p->v[1]*p->v[2] - p->s*p->v[0]) * qv[2];
  
  result[2] = 2*(p->v[0]*p->v[2] - p->s*p->v[1]) * qv[0]
    + 2*(p->v[1]*p->v[2] + p->s*p->v[0]) * qv[1]
    + (p->s*p->s - p->v[0]*p->v[0] - p->v[1]*p->v[1] + p->v[2]*p->v[2]) * qv[2];
}


static PyObject * pyquat_Quat_rotate(PyObject* self, PyObject* args) {

  PyArrayObject* vec;
  if (PyArg_ParseTuple(args, "O!|:rotate", &PyArray_Type, &vec)) {
    if (not_double_nx1_or_length_n(vec, 3)) return NULL;
    
    PyArrayObject* result = (PyArrayObject*)PyArray_SimpleNew(vec->nd, vec->dimensions, NPY_DOUBLE);
    Py_CheckAlloc(result);

    pyquat_Quat t; // temporary quaternion
    
    double* v = (double*)vec->data;

    // FIXME: combine these two functions into one single more efficient function.
    otimes_vector((pyquat_Quat*)self, (double*)vec->data, &t);
    otimes_conj_vector_out(&t, (pyquat_Quat*)self, (double*)result->data);

    return (PyObject*)result;
  }
  return NULL;
}


static double abs_max_component(pyquat_Quat* q) {
  double qw = fabs(q->s);
  double qx = fabs(q->v[0]);
  double qy = fabs(q->v[1]);
  double qz = fabs(q->v[2]);
  if (qw > qx) {
    if (qy > qz) return qw > qy ? qw : qy;
    else         return qw > qz ? qw : qz;
  } else {
    if (qy > qz) return qx > qy ? qx : qy;
    else         return qx > qz ? qx : qz;
  }
}


static void normalize_large(pyquat_Quat* result, pyquat_Quat* q) {
  double q_max = abs_max_component(q);

  result->s    = q->s    / q_max;
  result->v[0] = q->v[0] / q_max;
  result->v[1] = q->v[1] / q_max;
  result->v[2] = q->v[2] / q_max;

  double scaled_mag = sqrt(result->s    * result->s +
                           result->v[0] * result->v[0] +
                           result->v[1] * result->v[1] +
                           result->v[2] * result->v[2]);

  if (scaled_mag > PYQUAT_SMALL) {
    result->s    /= scaled_mag;
    result->v[0] /= scaled_mag;
    result->v[1] /= scaled_mag;
    result->v[2] /= scaled_mag;
  } else { // can't normalize, so just use identity
    result->s    = 1.0;
    result->v[0] = result->v[1] = result->v[2] = 0.0;
  }
}


static PyObject* pyquat_Quat_inplace_normalize_large(PyObject* self) {
  pyquat_Quat* q = (pyquat_Quat*)(self);
  normalize_large(q, q);

  Py_INCREF(self);
  return self;
}


static PyObject* pyquat_Quat_normalize_large(PyObject* self) {
  pyquat_Quat* q      = (pyquat_Quat*)(self);

  // allocate a quaternion for the result
  pyquat_Quat* result = (pyquat_Quat*) PyObject_New(pyquat_Quat, &pyquat_QuatType);
  if (!result) {
    PyErr_NoMemory();
    return NULL;
  }

  normalize_large(result, q);
  return (PyObject*)result;
}


static void normalize(pyquat_Quat* result, pyquat_Quat* q) {
  double q_mag = sqrt(q->s * q->s + q->v[0] * q->v[0] + q->v[1] * q->v[1] + q->v[2] * q->v[2]);
  if (q_mag > PYQUAT_SMALL) {
    result->s     = q->s / q_mag;
    result->v[0]  = q->v[0] / q_mag;
    result->v[1]  = q->v[1] / q_mag;
    result->v[2]  = q->v[2] / q_mag;
  } else { // can't normalize, so just use identity
    result->s     = 1.0;
    result->v[0]  = result->v[1] = result->v[2] = 0.0;
  }
}


static PyObject* pyquat_Quat_inplace_normalize(PyObject* self) {
  pyquat_Quat* q = (pyquat_Quat*)(self);
  normalize(q, q);

  Py_INCREF(self);
  return self;
}


static PyObject* pyquat_Quat_normalize(PyObject* self) {
  pyquat_Quat* q      = (pyquat_Quat*)(self);

  // allocate a quaternion for the result
  pyquat_Quat* result = (pyquat_Quat*) PyObject_New(pyquat_Quat, &pyquat_QuatType);
  if (!result) {
    PyErr_NoMemory();
    return NULL;
  }

  normalize(result, q);
  return (PyObject*)result;
}


static PyObject* pyquat_Quat_inplace_conjugate(PyObject* self) {

  pyquat_Quat* q = (pyquat_Quat*)(self);

  q->v[0] = -q->v[0];
  q->v[1] = -q->v[1];
  q->v[2] = -q->v[2];

  Py_INCREF(self);

  return self;
}


static PyObject* pyquat_Quat_conjugate(PyObject* self) {

  pyquat_Quat* q      = (pyquat_Quat*)(self);
  pyquat_Quat* result = (pyquat_Quat*) PyObject_New(pyquat_Quat, &pyquat_QuatType);
  Py_CheckAlloc(result);

  result->s    = q->s;
  result->v[0] = -q->v[0];
  result->v[1] = -q->v[1];
  result->v[2] = -q->v[2];

  return (PyObject*)result;
}


static PyObject* pyquat_Quat_copy(PyObject* self) {

  pyquat_Quat* q      = (pyquat_Quat*)(self);
  pyquat_Quat* result = (pyquat_Quat*) PyObject_New(pyquat_Quat, &pyquat_QuatType);
  Py_CheckAlloc(result);

  result->s    = q->s;
  result->v[0] = q->v[0];
  result->v[1] = q->v[1];
  result->v[2] = q->v[2];

  return (PyObject*)result;
}



static PyObject* pyquat_Quat_from_angle_axis(PyObject* type,
                                             PyObject* args,
                                             PyObject* kwargs)
{
  static char* keywords[] = {"angle", "x", "y", "z", "theta", NULL};

  double z, theta, phi, x, y;
  if (PyArg_ParseTupleAndKeywords(args, kwargs, "d|dddd:from_angle_axis",
                                   keywords,
                                   &phi, &x, &y, &z, &theta))
  {
    pyquat_Quat* q = (pyquat_Quat*) PyObject_New(pyquat_Quat, &pyquat_QuatType);
    Py_CheckAlloc(q);

    if (kwargs) {
      PyObject* theta_str = PyUnicode_FromString(keywords[4]);
      if (PyDict_Contains(kwargs, theta_str)) { // Overwrite any values for x and y
        x = sqrt(1.0 - z * z) * cos(theta);
        y = sqrt(1.0 - z * z) * sin(theta);
      }
      Py_DECREF(theta_str); // okay, done with that.
    }

    double half_angle = phi / 2.0;

    // Instantiate the quaternion.
    q->s    = cos(half_angle);
    q->v[0] = x * sin(half_angle);
    q->v[1] = y * sin(half_angle);
    q->v[2] = z * sin(half_angle);
    
    return (PyObject*)q;
  }

  return NULL;
}


/** \brief Helper function for converting a rotation vector to a pyquat
 */
static void from_rotation_vector(pyquat_Quat* q, double* phi) {
  double mag        = sqrt(phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2]);
  double half_angle = mag / 2.0;
  if (mag < PYQUAT_SMALL) { // very small angle: use identity
    q->s = 1.0;
    q->v[0] = q->v[1] = q->v[2] = 0.0;
  } else {
    q->s    = cos(half_angle);
    q->v[0] = (phi[0] / mag) * sin(half_angle);
    q->v[1] = (phi[1] / mag) * sin(half_angle);
    q->v[2] = (phi[2] / mag) * sin(half_angle);
  }
}


static PyObject* pyquat_Quat_from_rotation_vector(PyObject* type,
                                                  PyObject* args)
{
  PyArrayObject* ary;
  if (PyArg_ParseTuple(args, "O!|:from_rotation_vector", &PyArray_Type, &ary)) {
    if (not_double_nx1_or_length_n(ary, 3)) return NULL;
    PyArrayObject* v_ary = PyArray_GETCONTIGUOUS(ary);
    
    pyquat_Quat* q = (pyquat_Quat*) PyObject_New(pyquat_Quat, &pyquat_QuatType);
    Py_CheckAlloc(q);
    from_rotation_vector(q, (double*)v_ary->data);
        
    return (PyObject*)q;
  }

  return NULL;
}


static PyObject* pyquat_Quat_to_rotation_vector(PyObject* self) {
  npy_intp dims[2] = {3,1};
  
  pyquat_Quat* q = (pyquat_Quat*)(self);

  double vec_mag = 2.0 * acos(q->s);
  double a       = sin(vec_mag / 2.0);
  PyArrayObject* ary  = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
  Py_CheckAlloc(ary);

  double* vec    = (double*)ary->data;
  
  if (a > PYQUAT_SMALL) {
    vec[0] = q->v[0] * vec_mag / a;
    vec[1] = q->v[1] * vec_mag / a;
    vec[2] = q->v[2] * vec_mag / a;
  } else {
    vec[0] = vec[1] = vec[2] = 0.0;
  }

  return (PyObject*)ary;
}


/** \brief Helper function for converting a pyquat to a matrix
 */
static void to_matrix(pyquat_Quat* q, double* T) {
  T[0] = 1.0 - 2.0 * (q->v[2] * q->v[2] + q->v[1] * q->v[1]); 
  T[1] =       2.0 * (q->v[1] * q->v[0] +    q->s * q->v[2]);
  T[2] =       2.0 * (q->v[2] * q->v[0] -    q->s * q->v[1]);
  T[3] =       2.0 * (q->v[1] * q->v[0] -    q->s * q->v[2]);
  T[4] = 1.0 - 2.0 * (q->v[2] * q->v[2] + q->v[0] * q->v[0]);
  T[5] =       2.0 * (q->v[2] * q->v[1] +    q->s * q->v[0]);
  T[6] =       2.0 * (q->v[2] * q->v[0] +    q->s * q->v[1]);
  T[7] =       2.0 * (q->v[2] * q->v[1] -    q->s * q->v[0]);
  T[8] = 1.0 - 2.0 * (q->v[1] * q->v[1] + q->v[0] * q->v[0]); 
}


/** \brief Helper function for converting a matrix to a pyquat
 */
static void from_matrix(pyquat_Quat* q, double* T) {
  int i = 0;
  double tr = T[0] + T[4] + T[8]; // diagonals
  q->s = tr;
  for (int n = 0; n < 3; ++n) {
    if (T[n*3+n] > q->s) {
      i = n + 1;
      q->s = T[n*3+n];
    }
  }

  tr = sqrt(1.0 + 2.0 * q->s - tr);
  for (int n = 1; n <= 3; ++n) {
    int k = (n % 3) + 1;
    int j = 6 - n - k;
    if (i == 0 || n == i) {
      q->v[n-1]     = q->s = (T[(k-1)*3 + j-1] - T[(j-1)*3 + k-1]) / tr;
    } else {
      q->v[j+k-i-1] = (T[(k-1)*3 + j-1] + T[(j-1)*3 + k-1]) / tr;
    }
  }

  if (i == 0) q->s      = tr;
  else        q->v[i-1] = tr;

  tr = 0.5;
  if (q->s < 0.0)    tr = -tr;

  q->s    *= tr;
  q->v[0] *= tr;
  q->v[1] *= tr;
  q->v[2] *= tr;
}


/** @brief Helper function for pyquat_big_xi() */
static void to_big_xi(pyquat_Quat* q, double* Xi) {
  // scalar row
  Xi[0]  = -q->v[0];
  Xi[1]  = -q->v[1];
  Xi[2]  = -q->v[2];

  // vector portion
  Xi[3]  =  q->s;
  Xi[4]  = -q->v[2];
  Xi[5]  =  q->v[1];

  Xi[6]  =  q->v[2];
  Xi[7]  =  q->s;
  Xi[8]  = -q->v[0];

  Xi[9]  = -q->v[1];
  Xi[10] =  q->v[0];
  Xi[11] =  q->s;
}


/** @brief Create a Xi matrix from a quaternion.
 *
 * See Eq. 307 in [1], originally from [0].
 *
 * References:
 *
 * [0] Cayley, A. 1843. On the motion of rotation of a solid
 *     body. Cambridge Mathematics Journal 3(1843): 224-232.
 *
 * [1] Shuster, M. 1993. A survey of attitude representations.
 *     Journal of the Astronautical Sciences 41(4): 439-519.
 **/
static PyObject* pyquat_Quat_to_big_xi_matrix(PyObject* self) {
  npy_intp dims[2] = {4,3};
  
  pyquat_Quat* q = (pyquat_Quat*)(self);

  PyArrayObject* ary = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
  Py_CheckAlloc(ary);

  double* Xi = (double*)ary->data;
  to_big_xi(q, Xi);

  return (PyObject*)ary;
}



static PyObject* pyquat_Quat_to_matrix(PyObject* self) {
  npy_intp dims[2] = {3,3};
  
  pyquat_Quat* q = (pyquat_Quat*)(self);

  PyArrayObject* ary  = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
  Py_CheckAlloc(ary);

  double* T = (double*)ary->data;
  to_matrix(q, T);

  return (PyObject*)ary;
}


static PyObject* pyquat_Quat_to_vector(PyObject* self) {
  npy_intp dims[2] = {4,1};

  pyquat_Quat* q = (pyquat_Quat*)self;

  PyArrayObject* ary  = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
  Py_CheckAlloc(ary);

  double* vec = (double*)ary->data;

  vec[0] = q->s;
  vec[1] = q->v[0];
  vec[2] = q->v[1];
  vec[3] = q->v[2];

  return ary;
}


static PyObject* pyquat_Quat_to_unit_vector(PyObject* self, PyObject* args) {
  npy_intp dims[2] = {3,1};

  pyquat_Quat* q = (pyquat_Quat*)self;


  char axis = 'x';
  if (args) {
    if (!PyArg_ParseTuple(args, "c", &axis)) {
      PyErr_SetString(PyExc_IOError, "expected axis designation");
      return NULL;
    }
  }


  PyArrayObject* ary = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
  Py_CheckAlloc(ary);

  // Let's suppose we're multiplying q.to_matrix() by (1,0,0), or x-hat.
  // That's equivalent to the first column of q.to_matrix().
  double* vec = (double*)ary->data;
  if (axis == 'x') {
    vec[0] = 1.0 - 2.0 * (q->v[2] * q->v[2] + q->v[1] * q->v[1]);
    vec[1] =       2.0 * (q->v[1] * q->v[0] -    q->s * q->v[2]);
    vec[2] =       2.0 * (q->v[2] * q->v[0] +    q->s * q->v[1]);
  } else if (axis == 'y') {
    vec[0] =       2.0 * (q->v[1] * q->v[0] +    q->s * q->v[2]);
    vec[1] = 1.0 - 2.0 * (q->v[2] * q->v[2] + q->v[0] * q->v[0]);
    vec[2] =       2.0 * (q->v[2] * q->v[1] -    q->s * q->v[0]);
  } else if (axis == 'z') {
    vec[0] =       2.0 * (q->v[2] * q->v[0] -    q->s * q->v[1]);
    vec[1] =       2.0 * (q->v[2] * q->v[1] +    q->s * q->v[0]);
    vec[2] = 1.0 - 2.0 * (q->v[1] * q->v[1] + q->v[0] * q->v[0]);
  } else {
    PyErr_SetString(PyExc_IOError, "expected axis designation to be in (x, y, or z)");
    // return anyway
  }
  
  return ary;
}


static PyObject* pyquat_Quat_tobytes(PyObject* self,
                                     PyObject* args,
                                     PyObject* kwargs)
{
  static char *keywords[] = { "order", NULL };

  unsigned char order;
  if (PyArg_ParseTupleAndKeywords(args, kwargs, "|b:tobytes", keywords, &order)) {

    pyquat_Quat* q = (pyquat_Quat*)self;
    PyObject* ret = PyBytes_FromStringAndSize((char*)(&(q->s)), (Py_ssize_t) sizeof(double)*4);
    Py_CheckAlloc(ret);

    return ret;
  }
  return NULL;
}



static PyObject* pyquat_Quat_from_matrix(PyObject* type,
					 PyObject* args)
{
  PyArrayObject* ary;
  if (PyArg_ParseTuple(args, "O!|:from_matrix", &PyArray_Type, &ary)) {
    if (not_double_matrix(ary) || not_3x3(ary)) return NULL;
    PyArrayObject* mat = PyArray_GETCONTIGUOUS(ary);
    
    pyquat_Quat* q = (pyquat_Quat*) PyObject_New(pyquat_Quat, &pyquat_QuatType);
    Py_CheckAlloc(q);
    from_matrix(q, (double*)mat->data);
        
    return (PyObject*)q;
  }

  return NULL;
}


static PyObject* pyquat_rotation_vector_to_matrix(PyObject* self, PyObject* args) {
  PyArrayObject* ary;
  if (PyArg_ParseTuple(args, "O!|:rotation_vector_to_matrix", &PyArray_Type, &ary)) {

    if (not_double_nx1_or_length_n(ary, 3)) return NULL;
    PyArrayObject* vec = PyArray_GETCONTIGUOUS(ary);
    
    // First allocate a place to put it, and check that allocation was successful.
    npy_intp dims[2] = {3,3};
    PyArrayObject* mat = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    Py_CheckAlloc(mat);
    double* T = (double*)mat->data;
    
    
    double* v    = (double*)vec->data;
    double v_mag = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    double c     = cos(v_mag);
    double s     = sin(v_mag);
    double vu[3];
    if (v_mag < PYQUAT_SMALL) { // just set to identity if angle is infinitesmally small
      T[0] = T[4] = T[8] = 1.0;
      T[1] = T[2] = T[3] = T[5] = T[6] = T[7] = 0.0;
    } else {
      vu[0] = v[0] / v_mag;
      vu[1] = v[1] / v_mag;
      vu[2] = v[2] / v_mag;

      T[0]  = c + vu[0]*vu[0] * (1.0 - c);
      T[1]  = vu[0]*vu[1] * (1.0 - c) + vu[2] * s;
      T[2]  = vu[0]*vu[2] * (1.0 - c) - vu[1] * s;
      T[3]  = vu[0]*vu[1] * (1.0 - c) - vu[2] * s;
      T[4]  = c + vu[1]*vu[1] * (1.0 - c);
      T[5]  = vu[1]*vu[2] * (1.0 - c) + vu[0] * s;
      T[6]  = vu[0]*vu[2] * (1.0 - c) + vu[1] * s;
      T[7]  = vu[1]*vu[2] * (1.0 - c) - vu[0] * s;
      T[8]  = c + vu[2]*vu[2] * (1.0 - c);
    }

    return mat;
  }

  return NULL;
}


static void skew(double* v, double* vx) {
  vx[0] =  0.0;
  vx[1] = -v[2];
  vx[2] =  v[1];
  
  vx[3] =  v[2];
  vx[4] =  0.0;
  vx[5] = -v[0];

  vx[6] = -v[1];
  vx[7] =  v[0];
  vx[8] =  0.0;
}


static PyObject* pyquat_skew(PyObject* self, PyObject* args) {
  PyArrayObject* ary;
  if (PyArg_ParseTuple(args, "O!|:skew", &PyArray_Type, &ary)) {
    if (not_double_nx1_or_length_n(ary, 3)) return NULL;
    PyArrayObject* vec = PyArray_GETCONTIGUOUS(ary);

    // First allocate a place to put it, and check that allocation was successful.
    npy_intp dims[2] = {3,3};
    PyArrayObject* mat = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    Py_CheckAlloc(mat);

    double* vx   = (double*)mat->data;
    double* v    = (double*)vec->data;
    skew(v, vx);

    return PyArray_Return(mat);
  }

  return NULL;
}


/* Helper function for pyquat_big_omega() */
static void big_omega(double* w, double* W) {
  W[0]  =  0.0;
  W[1]  = -w[0];
  W[2]  = -w[1];
  W[3]  = -w[2];
    
  W[4]  =  w[0];
  W[5]  =  0.0;
  W[6]  =  w[2];
  W[7]  = -w[1];
    
  W[8]  =  w[1];
  W[9]  = -w[2];
  W[10] =  0.0;
  W[11] =  w[0];

  W[12] =  w[2];
  W[13] =  w[1];
  W[14] = -w[0];
  W[15] =  0.0;
}


static PyObject* pyquat_big_omega(PyObject* self, PyObject* args) {
  PyArrayObject* ary;
  if (PyArg_ParseTuple(args, "O!|:big_omega", &PyArray_Type, &ary)) {
    if (not_double_nx1_or_length_n(ary, 3)) return NULL;
    PyArrayObject* vec = PyArray_GETCONTIGUOUS(ary);

    // First allocate a place to put it, and check that allocation was successful.
    npy_intp dims[2] = {4,4};
    PyArrayObject* mat = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    Py_CheckAlloc(mat);
    
    double* w = (double*)vec->data;
    double* W = (double*)mat->data;
    big_omega(w, W);

    return PyArray_Return(mat);
  }

  return NULL;
}



/* Closed-form matrix exponential for an angular velocity and time step */
static void expm(double* w, double dt, double* M) {

  // Compute the magnitude of the angular velocity, then use that to get the
  // total angle over the time step dt.
  double w_mag = sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]);
  double theta = 0.5 * dt * w_mag;

  double c        = cos(theta);
  double s_over_w = sin(theta) / w_mag;

  // I*cos(theta) + Omega(w) * sin(theta) / mag(w)
  M[0]  =  c;
  M[1]  = -w[0] * s_over_w;
  M[2]  = -w[1] * s_over_w;
  M[3]  = -w[2] * s_over_w;

  M[4]  =  w[0] * s_over_w;
  M[5]  =  c;
  M[6]  =  w[2] * s_over_w;
  M[7]  = -w[1] * s_over_w;

  M[8]  =  w[1] * s_over_w;
  M[9]  = -w[2] * s_over_w;
  M[10] =  c;
  M[11] =  w[0] * s_over_w;

  M[12] =  w[2] * s_over_w;
  M[13] =  w[1] * s_over_w;
  M[14] = -w[0] * s_over_w;
  M[15] =  c;
}


static PyObject* pyquat_expm(PyObject* self, PyObject* args) {
  PyArrayObject* ary;
  double dt;
  if (PyArg_ParseTuple(args, "O!d|:big_omega", &PyArray_Type, &ary, &dt)) {
    if (not_double_nx1_or_length_n(ary, 3)) return NULL;
    PyArrayObject* vec = PyArray_GETCONTIGUOUS(ary);

    // First allocate a place to put it, and check that allocation was successful.
    npy_intp dims[2] = {4,4};
    PyArrayObject* mat = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    Py_CheckAlloc(mat);

    double* w = (double*)vec->data;
    double* M = (double*)mat->data;
    expm(w, dt, M);

    return mat;
  }

  return NULL;
}


static PyObject* pyquat_identity(PyObject* self) {
  pyquat_Quat* q =  (pyquat_Quat*) PyObject_New(pyquat_Quat, &pyquat_QuatType);

  q->s = 1.0;
  q->v[0] = q->v[1] = q->v[2] = 0.0;

  return (PyObject*)(q);
}


static int pyquat_Quat_compare(PyObject* left, PyObject* right) {

  // Expects both to be pyquat.Quat
  if (!PyObject_IsInstance(right, (PyObject*)&pyquat_QuatType)) {
    PyErr_SetString(PyExc_IOError, "expected quaternion");
    return -1;
  }

  if (left == right) return 0;

  pyquat_Quat* l = (pyquat_Quat*)left;
  pyquat_Quat* r = (pyquat_Quat*)right;

  // q == -q and q == q
  if (l->s == r->s && 
      l->v[0] == r->v[0] && 
      l->v[1] == r->v[1] &&
      l->v[2] == r->v[2]) {
    return 0;
  } else if (l->s == -r->s && 
      l->v[0] == -r->v[0] && 
      l->v[1] == -r->v[1] &&
      l->v[2] == -r->v[2]) {
    return 0;
  } else {
    return -1;
  }
}


/** @brief Compute the dot product between two quaternions as if they
 **        were 4-vectors.
 *
 * @detail This function does not perform any checks to see if the
 *         quaternions are allocated. That should be done before
 *         calling it.
 *
 * @param[in] p  pointer to the first operand
 * @param[in] q  pointer to the second operand
 *
 * @returns the dot product, a double.
 */
static double QdotQ(pyquat_Quat* p, pyquat_Quat* q) {
  return p->s*q->s + p->v[0]*q->v[0] + p->v[1]*q->v[1] + p->v[2]*q->v[2];
}


/** @brief Linear interpolation of two quaternions with no safety
 **        checks.
 *
 * @param[in]     q0      pointer to the first operand
 * @param[in]     q1      pointer to the second operand
 * @param[in]     t       interpolation coefficient between [0.0, 1.0]
 * @param[in,out] result  pre-allocated result of the computation
 *
 */
static void lerp(pyquat_Quat* q0, pyquat_Quat* q1, double t, pyquat_Quat* result) {
  result->s    = q0->s    + t*(q1->s    - q0->s   );
  result->v[0] = q0->v[0] + t*(q1->v[0] - q0->v[0]);
  result->v[1] = q0->v[1] + t*(q1->v[1] - q0->v[1]);
  result->v[2] = q0->v[2] + t*(q1->v[2] - q0->v[2]);
}


/** @brief Spherical linear interpolation of two quaternions with no
 **        safety checks.
 *
 * @param[in]     q0      pointer to the first operand
 * @param[in]     q1      pointer to the second operand
 * @param[in]     t       interpolation coefficient between [0.0, 1.0]
 * @param[in]     dot     dot product of q0 and q1
 * @param[in,out] result  pre-allocated result of the computation
 *
 */
static void slerp(pyquat_Quat* q0,
                  pyquat_Quat* q1,
                  double       t,
                  double       dot,
                  pyquat_Quat* result)
{
  double theta_0 = acos(dot);   // angle between input vectors
  double theta   = theta_0 * t; // angle between q0 and result
  double st      = sin(theta);
  double ct      = cos(theta);

  pyquat_Quat q2;
  q2.s    = q1->s    - q0->s    * dot;
  q2.v[0] = q1->v[0] - q0->v[0] * dot;
  q2.v[1] = q1->v[1] - q0->v[1] * dot;
  q2.v[2] = q1->v[2] - q0->v[2] * dot;
  normalize(&q2, &q2); // q0, q2 are now an orthonormal basis
  
  result->s    = q0->s    * ct   +   q2.s    * st;
  result->v[0] = q0->v[0] * ct   +   q2.v[0] * st;
  result->v[1] = q0->v[1] * ct   +   q2.v[1] * st;
  result->v[2] = q0->v[2] * ct   +   q2.v[2] * st;
}


/** @brief Spherical linear interpolation of two quaternions, or linear
 **        interpolation beneath a certain threshold; otherwise, no safety
 **        checks.
 *
 * @param[in]     q0              pointer to the first operand
 * @param[in]     q1              pointer to the second operand
 * @param[in]     t               interpolation coefficient between [0.0, 1.0]
 * @param[in]     lerp_threshold  dot product threshold at/beneath which 
 *                                we should use lerp+normalize instead of 
 *                                slerp
 * @param[in,out] result          pre-allocated result of the computation
 *
 */
static void slerp_or_lerp(pyquat_Quat* q0,
                          pyquat_Quat* q1,
                          double       t,
                          double       lerp_threshold,
                          pyquat_Quat* result)
{
  // Compute the dot product between the two quaternions as 4D vectors.
  double dot = QdotQ(q0, q1);
  
  if (lerp_threshold < 1.0 && fabs(dot) > lerp_threshold) {
    // Angle between the quaternions is too small, so we might as well just use LERP.
    lerp(q0, q1, t, result);
    normalize(result, result);
  } else if (dot < 0) {
    // Negative dot product means quaternions have opposite handedness.
    pyquat_Quat nq1;
    nq1.s    = -q1->s;
    nq1.v[0] = -q1->v[0];
    nq1.v[1] = -q1->v[1];
    nq1.v[2] = -q1->v[2];

    if (dot < -1) dot = 1.0;
    slerp(q0, &nq1, t, dot, result);
  } else {
    // Proceed with slerp
    if (dot > 1) dot = 1.0; // clamp
    slerp(q0, q1, t, dot, result);
  }

}


static PyObject* pyquat_Quat_lerp(PyObject* self, PyObject* args) {
  pyquat_Quat* q0 = (pyquat_Quat*)self;
  pyquat_Quat* q1;
  double       t;
  if (PyArg_ParseTuple(args, "O!d:lerp",
                       &pyquat_QuatType, &q1,
                       &t)) {

    if (t == 0) {
      return pyquat_Quat_copy(q0);
    } else if (t == 1) {
      return pyquat_Quat_copy(q1);
    }

    // allocate result quaternion
    pyquat_Quat* q = (pyquat_Quat*)PyObject_New(pyquat_Quat, &pyquat_QuatType);
    Py_CheckAlloc(q);

    // Perform linear interpolation.
    lerp(q0, q1, t, q);

    // Return the result.
    return (PyObject*)q;
  }

  return NULL;
}


static PyObject* pyquat_Quat_slerp(PyObject* self,
                                   PyObject* args,
                                   PyObject* kwargs) {
  pyquat_Quat* q0 = (pyquat_Quat*)self;
  pyquat_Quat* q1;
  double       t;
  double       lerp_threshold = 1.0;


  static char* keywords[] = {"rhs", "t", "lerp_threshold", NULL};
  
  if (PyArg_ParseTupleAndKeywords(args, kwargs, "O!d|d:slerp", keywords,
                                  &pyquat_QuatType, &q1,
                                  &t,
                                  &lerp_threshold)) {
    if (t == 0) {
      return pyquat_Quat_copy(q0);
    } else if (t == 1) {
      return pyquat_Quat_copy(q1);
    }

    // allocate result quaternion
    pyquat_Quat* q = (pyquat_Quat*)PyObject_New(pyquat_Quat, &pyquat_QuatType);
    Py_CheckAlloc(q);

    // Perform spherical linear interpolation or linear interpolation according
    // to lerp_threshold.
    slerp_or_lerp(q0, q1, t, lerp_threshold, q);

    // Return the result.
    return (PyObject*)q;
  }
  
  return NULL;
}


static PyObject* pyquat_Quat_dot(PyObject* self, PyObject* args) {
  pyquat_Quat* q0 = (pyquat_Quat*)self;
  pyquat_Quat* q1;
  double       t;
  if (PyArg_ParseTuple(args, "O!:dot",
                       &pyquat_QuatType, &q1)) {

    // Compute dot product.
    return PyFloat_FromDouble(QdotQ(q0, q1));
  }

  return NULL;
}


/*
 * API methods that should really go in a separate module eventually.
 */

/** @brief Helper for pqw_valenti_q_mag
 *
 * @param[in]     l  relatively noisey measurement vector with unit norm,
 *                   which has already been rotated into the xyD frame
 *                   given by q_acc(a)
 * @param[in,out] q  result quaternion, already allocated
 */
void valenti_q_mag(double l[3], pyquat_Quat* q) {

  double gamma         = l[0]*l[0] + l[1]*l[1];

  if (l[0] >= 0) {
    double gplxsg = gamma + l[0] * sqrt(gamma);
    q->s    = sqrt( gplxsg ) / sqrt(2.0 * gamma);
    q->v[0] = 0.0;
    q->v[1] = 0.0;
    q->v[2] = -l[1] / sqrt( 2.0 * gplxsg );
  } else {
    double gmlxsg = gamma - l[0] * sqrt(gamma);
    q->s    = l[1] / sqrt(2.0 * gmlxsg);
    q->v[0] = 0.0;
    q->v[1] = 0.0;
    q->v[2] = -sqrt( gmlxsg ) / sqrt(2.0 * gamma);
  }
}


/** @brief Helper for pqw_valenti_dq_mag
 *
 * @param[in]     l  relatively noisy measurement vector with unit norm,
 *                   which has already been rotated into the xyD frame
 *                   given by q_acc(a)
 * @param[in,out] q  result quaternion, already allocated
 */
void valenti_dq_mag(double l[3], pyquat_Quat* q) {

  double gamma         = l[0]*l[0] + l[1]*l[1];
  double gplxsg = gamma + l[0] * sqrt(gamma);
  q->s    = sqrt( gplxsg ) / sqrt(2.0 * gamma);
  q->v[0] = 0.0;
  q->v[1] = 0.0;
  q->v[2] = -l[1] / sqrt( 2.0 * gplxsg );
}


/** @brief Helper for pqw_valenti_dq_acc
 *
 * @param[in]     a  relatively noisy measurement vector with unit norm
 * @param[in,out] q  result quaternion q_acc, already allocated
 */
void valenti_dq_acc(double a[3], pyquat_Quat* q) {

  double s2x1pay = sqrt(2.0 * (1.0 + a[2]));
    
  q->s    =  sqrt( (1.0 + a[2]) / 2.0 );
  q->v[0] =  a[1] / s2x1pay;
  q->v[1] = -a[0] / s2x1pay;
  q->v[2] =  0.0;
}

/** @brief Helper for pqw_valenti_q_acc
 *
 * @param[in]     a  relatively noisy measurement vector with unit norm
 * @param[in,out] q  result quaternion q_acc, already allocated
 */
void valenti_q_acc(double a[3], pyquat_Quat* q) {

  if (a[2] >= 0) {
    double s2x1pay = sqrt(2.0 * (1.0 + a[2]));
    
    q->s    =  sqrt( (1.0 + a[2]) / 2.0 );
    q->v[0] =  a[1] / s2x1pay;
    q->v[1] = -a[0] / s2x1pay;
    q->v[2] =  0.0;
  } else {
    double s2x1may = sqrt(2.0 * (1.0 - a[2]));
    
    q->s    = -a[1] / s2x1may;
    q->v[0] = -sqrt( (1.0 - a[2]) / 2.0 );
    q->v[1] = 0.0;
    q->v[2] = -a[0] / s2x1may;
  }
}


static PyObject* pqw_valenti_q_mag(PyObject* self, PyObject* arg) {
  if (!PyObject_IsInstance(arg, (PyObject*)&PyArray_Type)) {
    PyErr_SetString(PyExc_ValueError, "expected numpy array");
    return NULL;
  }

  if (not_double_nx1_or_length_n((PyArrayObject*)arg, 3)) return NULL;
  PyArrayObject* ary_in = PyArray_GETCONTIGUOUS(arg);

  double* v = (double*)ary_in->data;

  pyquat_Quat* q = (pyquat_Quat*) PyObject_New(pyquat_Quat, &pyquat_QuatType);
  Py_CheckAlloc(q);

  valenti_q_mag(v, q);

  return (PyObject*)q;
}


static PyObject* pqw_valenti_dq_mag(PyObject* self, PyObject* arg) {
  if (!PyObject_IsInstance(arg, (PyObject*)&PyArray_Type)) {
    PyErr_SetString(PyExc_ValueError, "expected numpy array");
    return NULL;
  }

  if (not_double_nx1_or_length_n((PyArrayObject*)arg, 3)) return NULL;
  PyArrayObject* ary_in = PyArray_GETCONTIGUOUS(arg);

  double* v = (double*)ary_in->data;

  pyquat_Quat* q = (pyquat_Quat*) PyObject_New(pyquat_Quat, &pyquat_QuatType);
  Py_CheckAlloc(q);

  valenti_dq_mag(v, q);

  return (PyObject*)q;
}



static PyObject* pqw_valenti_q_acc(PyObject* self, PyObject* arg) {
  if (!PyObject_IsInstance(arg, (PyObject*)&PyArray_Type)) {
    PyErr_SetString(PyExc_ValueError, "expected numpy array");
    return NULL;
  }

  if (not_double_nx1_or_length_n((PyArrayObject*)arg, 3)) return NULL;
  PyArrayObject* ary_in = PyArray_GETCONTIGUOUS(arg);

  double* v = (double*)ary_in->data;

  pyquat_Quat* q = (pyquat_Quat*) PyObject_New(pyquat_Quat, &pyquat_QuatType);
  Py_CheckAlloc(q);

  valenti_q_acc(v, q);

  return (PyObject*)q;
}


static PyObject* pqw_valenti_dq_acc(PyObject* self, PyObject* arg) {
  if (!PyObject_IsInstance(arg, (PyObject*)&PyArray_Type)) {
    PyErr_SetString(PyExc_ValueError, "expected numpy array");
    return NULL;
  }

  if (not_double_nx1_or_length_n((PyArrayObject*)arg, 3)) return NULL;
  PyArrayObject* ary_in = PyArray_GETCONTIGUOUS(arg);

  double* v = (double*)ary_in->data;

  pyquat_Quat* q = (pyquat_Quat*) PyObject_New(pyquat_Quat, &pyquat_QuatType);
  Py_CheckAlloc(q);

  valenti_dq_acc(v, q);

  return (PyObject*)q;
}
