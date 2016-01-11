#include "pyquat.h"


static int pyquat_Quat_init(pyquat_Quat* self, PyObject* args);
static PyObject * pyquat_Quat_mul(PyObject* self, PyObject* args);
static PyObject * pyquat_Quat_inplace_normalize(PyObject* self);
static PyObject * pyquat_Quat_inplace_conjugate(PyObject* self);
static PyObject * pyquat_Quat_to_angle_vector(PyObject* self);
static PyObject * pyquat_Quat_to_matrix(PyObject* self);

/*
static PyObject * pyquat_Quat_new(PyTypeObject* type, PyObject* args) {
  pyquat_Quat* self = (pyquat_Quat *)type->tp_alloc(type, 0);  
  return (PyObject *)(self);
}
*/


static PyMethodDef pyquat_methods[] = {
  {NULL, NULL, 0, NULL}  /* Sentinel */
};




static PyMemberDef pyquat_Quat_members[] = {
  {"s",  T_DOUBLE, offsetof(pyquat_Quat, s),                    0, "scalar component"  },
  {"vx", T_DOUBLE, offsetof(pyquat_Quat, v),                    0, "vector x component"},
  {"vy", T_DOUBLE, offsetof(pyquat_Quat, v) + sizeof(double),   0, "vector y component"},
  {"vz", T_DOUBLE, offsetof(pyquat_Quat, v) + 2*sizeof(double), 0, "vector z component"},
  {NULL}  /* Sentinel */
};


static PyMethodDef pyquat_Quat_methods[] = {
  {"to_angle_vector", (PyCFunction)pyquat_Quat_to_angle_vector, METH_NOARGS, "convert to a unit axis divided by the angle of rotation in radians"},
  {"to_matrix", (PyCFunction)pyquat_Quat_to_matrix, METH_NOARGS, "convert to a transformation matrix"},
  {"normalize", (PyCFunction)pyquat_Quat_inplace_normalize, METH_NOARGS, "in-place normalize the unit quaternion"},
  {"conjugate", (PyCFunction)pyquat_Quat_inplace_conjugate, METH_NOARGS, "in-place conjugate the unit quaternion"},
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
  "pyquat.Quat",             /*tp_name*/
  sizeof(pyquat_Quat),       /*tp_basicsize*/
  0,                         /*tp_itemsize   -- variable-length objects like lists and strings, does not apply */
  0,                         /*tp_dealloc*/
  0,                         /*tp_print*/
  0,                         /*tp_getattr*/
  0,                         /*tp_setattr*/
  0,                         /*tp_compare*/
  0,                         /*tp_repr*/
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
  0,		             /*tp_traverse */
  0,		             /*tp_clear */
  0,		             /*tp_richcompare */
  0,		             /*tp_weaklistoffset */
  0,		             /*tp_iter */
  0,		             /*tp_iternext */
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
PyMODINIT_FUNC initpyquat(void) {
  PyObject* m;

  pyquat_QuatType.tp_new = PyType_GenericNew;
  if (PyType_Ready(&pyquat_QuatType) < 0)
    return;

  // Define the pyquat module.
  m = Py_InitModule3("pyquat", pyquat_methods,
		     "Quaternion module with fast unit (right) quaternion math written in C.");

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

static PyObject * pyquat_Quat_mul(PyObject* self, PyObject* arg) {

  // Expects the one argument to be a pyquat_Quat
  if (!PyObject_IsInstance(arg, (PyObject*)&pyquat_QuatType)) {
    Py_DECREF(arg);
    PyErr_SetString(PyExc_IOError, "expected quaternion");
    return NULL;
  }

  pyquat_Quat* rhs    = (pyquat_Quat*)(arg);
  pyquat_Quat* lhs    = (pyquat_Quat*)(self);
  pyquat_Quat* result = (pyquat_Quat *)Py_TYPE(self)->tp_alloc(Py_TYPE(self), 0);
  
  result->s    = lhs->s * rhs->s - (lhs->v[0] * rhs->v[0] + lhs->v[1] * rhs->v[1] + lhs->v[2] * rhs->v[2]);
  result->v[0] = lhs->s * rhs->v[0] + rhs->s * lhs->v[0] - (lhs->v[1] * rhs->v[2] - lhs->v[2] * rhs->v[1]);
  result->v[1] = lhs->s * rhs->v[1] + rhs->s * lhs->v[1] - (lhs->v[2] * rhs->v[0] - lhs->v[0] * rhs->v[2]);
  result->v[2] = lhs->s * rhs->v[2] + rhs->s * lhs->v[2] - (lhs->v[0] * rhs->v[1] - lhs->v[1] * rhs->v[0]);

  return (PyObject*)(result);
}

static PyObject* pyquat_Quat_inplace_normalize(PyObject* self) {

  pyquat_Quat* q = (pyquat_Quat*)(self);

  double q_mag = sqrt(q->s * q->s + q->v[0] * q->v[0] + q->v[1] * q->v[1] + q->v[2] * q->v[2]);
  if (q_mag > PYQUAT_QUAT_SMALL) q_mag = 1.0 / q_mag;
  else                           q_mag = 0.0;

  q->s    *= q_mag;
  q->v[0] *= q_mag;
  q->v[1] *= q_mag;
  q->v[2] *= q_mag;

  return self;
}


static PyObject* pyquat_Quat_inplace_conjugate(PyObject* self) {

  pyquat_Quat* q = (pyquat_Quat*)(self);

  q->v[0] = -q->v[0];
  q->v[1] = -q->v[1];
  q->v[2] = -q->v[2];

  return self;
}



static PyObject* pyquat_Quat_to_angle_vector(PyObject* self) {
  npy_intp dims[2] = {3,1};
  
  pyquat_Quat* q = (pyquat_Quat*)(self);

  double vec_mag = 2.0 * acos(q->s);
  double a       = sin(vec_mag / 2.0);
  PyArrayObject* ary  = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
  double* vec    = (double*)ary->data;
  
  if (a > PYQUAT_QUAT_SMALL) {
    double scale = vec_mag / a;
    vec[0] = q->v[0] * scale;
    vec[1] = q->v[1] * scale;
    vec[2] = q->v[2] * scale;
  } else {
    vec[0] = vec[1] = vec[2] = 0.0;
  }

  return PyArray_Return(ary);
}


static PyObject* pyquat_Quat_to_matrix(PyObject* self) {
  npy_intp dims[2] = {3,3};
  
  pyquat_Quat* q = (pyquat_Quat*)(self);

  PyArrayObject* ary  = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
  double* T = (double*)ary->data;
  
  T[0] = 1.0 - 2.0 * (q->v[2] * q->v[2] + q->v[1] * q->v[1]); 
  T[1] =       2.0 * (q->v[1] * q->v[0] +    q->s * q->v[2]);
  T[2] =       2.0 * (q->v[2] * q->v[0] -    q->s * q->v[1]);
  T[3] =       2.0 * (q->v[1] * q->v[0] -    q->s * q->v[2]);
  T[4] = 1.0 - 2.0 * (q->v[2] * q->v[2] + q->v[0] * q->v[0]);
  T[5] =       2.0 * (q->v[2] * q->v[1] +    q->s * q->v[0]);
  T[6] =       2.0 * (q->v[2] * q->v[0] +    q->s * q->v[1]);
  T[7] =       2.0 * (q->v[2] * q->v[1] -    q->s * q->v[0]);
  T[8] = 1.0 - 2.0 * (q->v[1] * q->v[1] + q->v[0] * q->v[0]);

  return PyArray_Return(ary);
}
