#ifndef PYQUAT_H
#define PYQUAT_H

#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>

#ifndef PyMODINIT_FUNC /* declarations for dylib/dll/so import/export */
#define PyMODINIT_FUNC void
#endif

#define PYQUAT_SMALL 1.0e-12

typedef struct {
  PyObject_HEAD // semicolon intentionally left out, don't add

  /* Type-specific fields go here */
  double s;     // scalar component
  double v[3];  // vector components
} pyquat_Quat;


#endif // PYQUAT_H
