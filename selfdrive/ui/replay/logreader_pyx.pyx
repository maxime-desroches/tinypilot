# distutils: language = c++
# cython: language_level = 3
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string

cdef extern from "selfdrive/ui/replay/logreader.h":
  cdef cppclass cpp_LogReader "LogReader":
    cpp_LogReader()
    bool load(string)


cdef class LogReader:
  cdef cpp_LogReader* lr

  def __cinit__(self):
    self.lr = new cpp_LogReader()

  def __dealloc__(self):
    del self.lr

  #def set_allow(self, allow_list):
  #  print([x.encode() for x in allow_list])
  #  self.lr.setAllow([x.encode() for x in allow_list])

  def load(self, logfile):
    return self.lr.load(logfile)
