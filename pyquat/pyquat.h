#ifndef PYQUAT_H
#define PYQUAT_H

#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>

#if PY_MAJOR_VERSION >= 3
#define IS_PY3K
#endif

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


#ifndef Py_CheckAlloc
/** @brief Macro checks that a new object was properly allocated, and if not,
 **        raises a NoMemoryError and returns from the calling function.
 *
 * @param[in] obj  PyObject pointer to test
 */
#define Py_CheckAlloc(obj)   \
    if (!obj) {              \
      PyErr_NoMemory();      \
      return NULL;           \
    }
#endif


struct module_state {
    PyObject *error;
};

#ifdef IS_PY3K
#define MOD_DEF(ob, name, doc, methods)                       \
  static struct PyModuleDef moduledef = {                     \
    PyModuleDef_HEAD_INIT, name, doc, -1, methods, };         \
  ob = PyModule_Create(&moduledef);

#else // Python 2.7 defs

#define MOD_DEF(ob, name, doc, methods)         \
  ob = Py_InitModule3(name, methods, doc);

static struct module_state _state;

#endif

#endif // PYQUAT_H
