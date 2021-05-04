# distutils: language = c++
# cython: language_level = 3
from libcpp cimport bool
from libcpp.string cimport string
from common.params_pxd cimport Params as c_Params, ParamKeyType

import os
import threading
from enum import Enum, unique
from common.basedir import BASEDIR


cdef class TxType:
  PERSISTENT = ParamKeyType.PERSISTENT
  CLEAR_ON_MANAGER_START = ParamKeyType.CLEAR_ON_MANAGER_START
  CLEAR_ON_PANDA_DISCONNECT = ParamKeyType.CLEAR_ON_PANDA_DISCONNECT
  ALL = ParamKeyType.ALL

def ensure_bytes(v):
  if isinstance(v, str):
    return v.encode()
  else:
    return v


class UnknownKeyName(Exception):
  pass

cdef class Params:
  cdef c_Params* p

  def __cinit__(self, d=None, bool persistent_params=False):
    if d is None:
      self.p = new c_Params(persistent_params)
    else:
      self.p = new c_Params(<string>d.encode())

  def __dealloc__(self):
    del self.p

  def clear_all(self, tx_type=None):
    if (tx_type == None):
      tx_type = TxType.ALL

    self.p.clear_all(tx_type)

  def manager_start(self):
    self.clear_all(TxType.CLEAR_ON_MANAGER_START)

  def panda_disconnect(self):
    self.clear_all(TxType.CLEAR_ON_PANDA_DISCONNECT)

  def check_key(self, key):
    key = ensure_bytes(key)

    if not self.p.check_key(key):
      raise UnknownKeyName(key)

    return key

  def get(self, key, block=False, encoding=None):
    cdef string k = self.check_key(key)
    cdef bool b = block

    cdef string val
    with nogil:
      val = self.p.get(k, b)

    if val == b"":
      if block:
        # If we got no value while running in blocked mode
        # it means we got an interrupt while waiting
        raise KeyboardInterrupt
      else:
        return None

    if encoding is not None:
      return val.decode(encoding)
    else:
      return val

  def get_bool(self, key):
    cdef string k = self.check_key(key)
    return self.p.getBool(k)

  def put(self, key, dat):
    """
    Warning: This function blocks until the param is written to disk!
    In very rare cases this can take over a second, and your code will hang.
    Use the put_nonblocking helper function in time sensitive code, but
    in general try to avoid writing params as much as possible.
    """
    cdef string k = self.check_key(key)
    dat = ensure_bytes(dat)
    self.p.put(k, dat)

  def put_bool(self, key, val):
    cdef string k = self.check_key(key)
    self.p.putBool(k, val)

  def delete(self, key):
    cdef string k = self.check_key(key)
    self.p.remove(k)


def put_nonblocking(key, val, d=None):
  def f(key, val):
    params = Params(d)
    cdef string k = ensure_bytes(key)
    params.put(k, val)

  t = threading.Thread(target=f, args=(key, val))
  t.start()
  return t
