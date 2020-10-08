from libcpp.string cimport string
from libcpp cimport bool

cdef extern from "selfdrive/common/params.cc":
  pass

cdef extern from "selfdrive/common/util.c":
  pass

# Declare the class with cdef
cdef extern from "selfdrive/common/params.h":
  cdef cppclass Params:
    Params(bool)
    Params(string)
    string get(string, bool) nogil
    void rm(string);
    void put(string, string);
