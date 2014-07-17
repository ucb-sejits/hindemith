#include <cstdio>
#include <cstdlib>
#include <string.h>
#include <vector>
#include <map>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <cstring>

#ifdef __linux__
extern "C" {
#include <execinfo.h>
#include <python2.7/Python.h>
#include <python2.7/pyconfig.h>
#include <python2.7/numpy/arrayobject.h>
#include <python2.7/numpy/arrayscalars.h>
};
#elif __APPLE__
#include <execinfo.h>
#include <python2.7/Python.h>
#include <python2.7/pyconfig.h>
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>
#else
#error Unsupported OS
#endif

#define EXIT_MSG(str) {\
	printf("Error: %s\n", str); \
	Py_INCREF(Py_None); \
	return Py_None; \
}

#define MAX_RETURN_DIMS 10

#include "colorcode.h"

static char run_doc[] = 
"run(a, b)\n\
\n\
Executes a DAG of Add, Subtract, and SPMV operations.";

static PyObject*
run(PyObject *self, PyObject *args, PyObject *kwargs)
{

  /* get the number of arguments */
  Py_ssize_t argc = PyTuple_Size(args);
  if(argc != 3)
  {
    std::cout << "Not the right number of arguments" << std::endl;
    Py_INCREF(Py_None);
    return Py_None;
  }

  /* get the first argument (kwargs) */
  PyObject *dxObj = PyTuple_GetItem(args, 0); 
  PyObject *dyObj = PyTuple_GetItem(args, 1); 
  PyObject *fmObj = PyTuple_GetItem(args, 2); 
  assert(PyArray_Check(dxObj));
  assert(PyArray_Check(dyObj));

  // Found variable, now copy it into the C/OCL data structure
  PyArrayObject * dxArray = (PyArrayObject*)dxObj;
  PyArrayObject * dyArray = (PyArrayObject*)dyObj;

  npy_intp * dims = PyArray_DIMS(dxArray);
  npy_intp out_dims[3];
  out_dims[0] = dims[0];
  out_dims[1] = dims[1];
  out_dims[2] = 3;

  double * dxData = (double *)dxArray->data;
  double * dyData = (double *)dyArray->data;
  unsigned char * out_data = (unsigned char *) calloc(out_dims[0] * out_dims[1] * out_dims[2], sizeof(unsigned char));

  double flowmax = 0.0;

  double input_fm;
  PyArray_ScalarAsCtype(fmObj, &input_fm);

  if(input_fm <= 0)
  {

    for(int i = 0 ; i < out_dims[0] ; i++)
    {
      for(int j = 0 ; j < out_dims[1] ; j++)
      {
        double fx = dxData[j + i * out_dims[1]];
        double fy = dyData[j + i * out_dims[1]];
        double mag = sqrt(fx * fx + fy * fy);
        if(mag > flowmax) flowmax = mag;
      }
    }
  }
  else
  {
    flowmax = input_fm;
  }


  for(int i = 0 ; i < out_dims[0] ; i++)
  {
    for(int j = 0 ; j < out_dims[1] ; j++)
    {
      double fx = dxData[j + i * out_dims[1]];
      double fy = dyData[j + i * out_dims[1]];
      computeColor(fx / flowmax, fy / flowmax , &out_data[j * 3 + i * out_dims[1] * 3]);
    }
  }

  PyObject * retArray = PyArray_New(&PyArray_Type, 3, out_dims, NPY_UINT8, NULL, out_data, 0, NPY_C_CONTIGUOUS, NULL);

  return retArray;

}


static char flow_mod_doc[] = 
"xmlcompile(a, b)\n\
\n\
Executes a DAG of Add, Subtract, and SPMV operations.";

static PyMethodDef flow_mod_methods[] = {
  {"run", (PyCFunction)run, METH_VARARGS | METH_KEYWORDS,
	 run_doc},
	 {NULL, NULL}
};

PyMODINIT_FUNC
initflow_mod(void)
{
  _import_array();
  Py_InitModule3("flow_mod", flow_mod_methods, flow_mod_doc);
}

