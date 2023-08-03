# distutils: language = c++
# cython: c_string_encoding=ascii, language_level=3

import numpy as np
cimport numpy as cnp
from libcpp cimport float
from libc.string cimport memcpy
from cereal.visionipc.visionipc_pyx cimport VisionBuf, CLContext as BaseCLContext
from cereal.visionipc.visionipc cimport cl_device_id, cl_context, cl_mem
from .commonmodel cimport CL_DEVICE_TYPE_DEFAULT, cl_get_device_id, cl_create_context, mat3, ModelFrame as cppModelFrame

cdef class CLContext(BaseCLContext):
  def __cinit__(self):
    self.device_id = cl_get_device_id(CL_DEVICE_TYPE_DEFAULT)
    self.context = cl_create_context(self.device_id)

cdef class CLMem:
  @staticmethod
  cdef create(void * cmem):
    mem = CLMem()
    mem.mem = <cl_mem*> cmem
    return mem

cdef class ModelFrame:
  cdef cppModelFrame * frame

  def __cinit__(self, CLContext context):
    self.frame = new cppModelFrame(context.device_id, context.context)

  def __dealloc__(self):
    del self.frame

  def prepare(self, VisionBuf buf, float[:] projection, CLMem output):
    cdef mat3 cprojection
    memcpy(cprojection.v, &projection[0], 9*sizeof(float))
    cdef float * data = self.frame.prepare(buf.buf.buf_cl, buf.width, buf.height, buf.stride, buf.uv_offset, cprojection, output.mem)
    if not data:
      return None
    return np.asarray(<cnp.float32_t[:self.frame.buf_size]> data)
