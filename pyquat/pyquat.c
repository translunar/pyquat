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
static PyObject* pyquat_Quat_inplace_conjugate(PyObject* self);
static PyObject* pyquat_Quat_normalize(PyObject* self);
static PyObject* pyquat_Quat_conjugate(PyObject* self);
static PyObject* pyquat_Quat_to_rotation_vector(PyObject* self);
static void      to_matrix(pyquat_Quat* q, double* T);
static PyObject* pyquat_Quat_to_matrix(PyObject* self);
static PyObject* pyquat_Quat_to_unit_vector(PyObject* self, PyObject* args);
static PyObject* pyquat_Quat_to_vector(PyObject* self);
static PyObject* pyquat_identity(PyObject* self);
static int       pyquat_Quat_compare(PyObject* left, PyObject* right);
static PyObject* pyquat_rotation_vector_to_matrix(PyObject* self, PyObject* args);

/*
static PyObject * pyquat_Quat_new(PyTypeObject* type, PyObject* args) {
  pyquat_Quat* self = (pyquat_Quat *)type->tp_alloc(type, 0);  
  return (PyObject *)(self);
}
*/


static PyMethodDef pyquat_methods[] = {
  {"identity", (PyCFunction)pyquat_identity, METH_NOARGS, "create an identity quaternion (1.0, 0.0, 0.0, 0.0)"},
  {"rotation_vector_to_matrix", (PyCFunction)pyquat_rotation_vector_to_matrix, METH_VARARGS, "convert a rotation vector direction to a directed-cosine matrix, skipping the quaternion"},
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
  {"conjugate", (PyCFunction)pyquat_Quat_inplace_conjugate, METH_NOARGS, "in-place conjugate the quaternion"},
  {"normalized", (PyCFunction)pyquat_Quat_normalize, METH_NOARGS, "normalize the quaternion"},
  {"conjugated", (PyCFunction)pyquat_Quat_conjugate, METH_NOARGS, "copy and conjugate the quaternion"},
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
  PyObject_HEAD_INIT(NULL)   // No semicolon! Don't add one
  0,                         /*ob_size       -- historical artifact, not used any longer */
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
PyMODINIT_FUNC init_pyquat(void) {
  PyObject* m;

  pyquat_QuatType.tp_new = PyType_GenericNew;
  if (PyType_Ready(&pyquat_QuatType) < 0)
    return;

  // Define the pyquat module.
  m = Py_InitModule3("_pyquat", pyquat_methods,
         "Quaternion module with fast unit (right) quaternion math written in C.");
  if (m == NULL)
    return;

  // Import NumPy to prevent a segfault when we call a function that uses NumPy API.
  import_array();

  // Create the Quat class in the pyquat module.
  Py_INCREF(&pyquat_QuatType);
  PyModule_AddObject(m, "Quat", (PyObject *)&pyquat_QuatType);
}


static int pyquat_Quat_init(pyquat_Quat* self, PyObject* args) {
  
  double scalar, vx, vy, vz;
  
  if (!PyArg_ParseTuple(args, "dddd", &scalar, &vx, &vy, &vz))
    return -1;

  // Read the scalar and vector components of the quaternion.
  self->s = scalar;
  self->v[0] = vx;
  self->v[1] = vy;
  self->v[2] = vz;

  return 0;
}


static PyObject* pyquat_Quat_repr(PyObject* obj) {
  pyquat_Quat* self = (pyquat_Quat*)(obj);
  return PyString_FromFormat("Quat{{\%s, \%s, \%s, \%s}}", 
                             PyOS_double_to_string(self->s, 'g', 17, 0, NULL),
                             PyOS_double_to_string(self->v[0], 'g', 17, 0, NULL),
                             PyOS_double_to_string(self->v[1], 'g', 17, 0, NULL),
                             PyOS_double_to_string(self->v[2], 'g', 17, 0, NULL));
}


static PyObject * pyquat_Quat_mul(PyObject* self, PyObject* arg) {

  // Expects the one argument to be a pyquat_Quat
  if (!PyObject_IsInstance(arg, (PyObject*)&pyquat_QuatType)) {
    PyErr_SetString(PyExc_IOError, "expected quaternion");
    return NULL;
  }

  pyquat_Quat* rhs    = (pyquat_Quat*)(arg);
  pyquat_Quat* lhs    = (pyquat_Quat*)(self);
  pyquat_Quat* result = (pyquat_Quat*) PyObject_New(pyquat_Quat, &pyquat_QuatType);
  if (result == NULL) {
    PyErr_NoMemory();
    return NULL;    
  }
  
  result->s    = lhs->s * rhs->s - (lhs->v[0] * rhs->v[0] + lhs->v[1] * rhs->v[1] + lhs->v[2] * rhs->v[2]);
  result->v[0] = lhs->s * rhs->v[0] + rhs->s * lhs->v[0] - (lhs->v[1] * rhs->v[2] - lhs->v[2] * rhs->v[1]);
  result->v[1] = lhs->s * rhs->v[1] + rhs->s * lhs->v[1] - (lhs->v[2] * rhs->v[0] - lhs->v[0] * rhs->v[2]);
  result->v[2] = lhs->s * rhs->v[2] + rhs->s * lhs->v[2] - (lhs->v[0] * rhs->v[1] - lhs->v[1] * rhs->v[0]);

  return (PyObject*)(result);
}

static PyObject* pyquat_Quat_inplace_normalize(PyObject* self) {

  pyquat_Quat* q = (pyquat_Quat*)(self);

  double q_mag = sqrt(q->s * q->s + q->v[0] * q->v[0] + q->v[1] * q->v[1] + q->v[2] * q->v[2]);
  if (q_mag > PYQUAT_SMALL) {
    q->s    /= q_mag;
    q->v[0] /= q_mag;
    q->v[1] /= q_mag;
    q->v[2] /= q_mag;
  } else { // cannot normalize, so just use identity
    q->s = 1.0;
    q->v[0] = q->v[1] = q->v[2] = 0.0;
  }

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
  if (result == NULL) {
    PyErr_NoMemory();
    return NULL;    
  }

  result->s    = q->s;
  result->v[0] = -q->v[0];
  result->v[1] = -q->v[1];
  result->v[2] = -q->v[2];

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
    if (!q) {
      PyErr_NoMemory();
      return NULL;
    }

    if (kwargs) {
      PyObject* theta_str = PyString_FromString(keywords[4]);
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
    pyquat_Quat* q = (pyquat_Quat*) PyObject_New(pyquat_Quat, &pyquat_QuatType);
    if (!q) {
      PyErr_NoMemory();
      return NULL;
    }
    from_rotation_vector(q, (double*)ary->data);
        
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

  // Check that allocation was successful
  if (ary == NULL) {
    PyErr_NoMemory();
    return NULL;
  }

  double* vec    = (double*)ary->data;
  
  if (a > PYQUAT_SMALL) {
    vec[0] = q->v[0] * vec_mag / a;
    vec[1] = q->v[1] * vec_mag / a;
    vec[2] = q->v[2] * vec_mag / a;
  } else {
    vec[0] = vec[1] = vec[2] = 0.0;
  }

  return PyArray_Return(ary);
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



static PyObject* pyquat_Quat_to_matrix(PyObject* self) {
  npy_intp dims[2] = {3,3};
  
  pyquat_Quat* q = (pyquat_Quat*)(self);

  PyArrayObject* ary  = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
  // Check that allocation was successful
  if (ary == NULL) {
    PyErr_NoMemory();
    return NULL;
  }

  double* T = (double*)ary->data;
  to_matrix(q, T);


  return PyArray_Return(ary);
}


static PyObject* pyquat_Quat_to_vector(PyObject* self) {
  npy_intp dims[2] = {4,1};

  pyquat_Quat* q = (pyquat_Quat*)self;

  PyArrayObject* ary  = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
  // Check that allocation was successful
  if (ary == NULL) {
    PyErr_NoMemory();
    return NULL;
  }

  double* vec = (double*)ary->data;

  vec[0] = q->s;
  vec[1] = q->v[0];
  vec[2] = q->v[1];
  vec[3] = q->v[2];

  return PyArray_Return(ary);
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
  // Check that allocation was successful
  if (ary == NULL) {
    PyErr_NoMemory();
    return NULL;
  }

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
  
  return PyArray_Return(ary);
}


static PyObject* pyquat_Quat_from_matrix(PyObject* type,
					 PyObject* args)
{
  PyArrayObject* ary;
  if (PyArg_ParseTuple(args, "O!|:from_matrix", &PyArray_Type, &ary)) {
    pyquat_Quat* q = (pyquat_Quat*) PyObject_New(pyquat_Quat, &pyquat_QuatType);
    if (!q) {
      PyErr_NoMemory();
      return NULL;
    }
    from_matrix(q, (double*)ary->data);
        
    return (PyObject*)q;
  }

  return NULL;
}


static PyObject* pyquat_rotation_vector_to_matrix(PyObject* self, PyObject* args) {
  PyArrayObject* ary;
  if (PyArg_ParseTuple(args, "O!|:rotation_vector_to_matrix", &PyArray_Type, &ary)) {

    // First allocate a place to put it, and check that allocation was successful.
    npy_intp dims[2] = {3,3};
    PyArrayObject* mat = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!mat) {
      PyErr_NoMemory();
      return NULL;
    }
    double* T = (double*)mat->data;
    
    
    double* v    = (double*)ary->data;
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

    return PyArray_Return(mat);
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
