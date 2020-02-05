#!/usr/bin/env python2
# TODO: why are the keras models saved with python 2?
from __future__ import print_function

import sys
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

def read(sz):
  dd = []
  gt = 0
  while gt < sz*4:
    # TODO: wrong for python 3
    st = sys.stdin.read()
    print(len(st), file=sys.stderr)
    dd.append(st)
    gt += len(st)
  return np.fromstring(b''.join(dd), dtype=np.float32)

def write(d):
  # TODO: wrong for python 3
  sys.stdout.write(np.tostring(d))

def run_loop(m):
  isize = m.inputs[0].shape[1]
  osize = m.outputs[0].shape[1]
  print("ready to run keras model %d -> %d" % (isize, osize), file=sys.stderr)
  while 1:
    idata = read(isize).reshape((1, isize))
    ret = m.predict_on_batch(idata)
    write(ret)

if __name__ == "__main__":
  m = load_model(sys.argv[1])
  print(m, file=sys.stderr)
  bs = [int(np.product(ii.shape[1:])) for ii in m.inputs]
  ri = keras.layers.Input((sum(bs),))

  tii = []
  acc = 0
  for i, ii in enumerate(m.inputs):
    ti = keras.layers.Lambda(lambda x: x[acc:acc+bs[i]])(ri)
    acc += bs[i]
    tii.append(keras.layers.Reshape(ii.shape[1:])(ti))
  no = keras.layers.Concatenate()(m(tii))
  m = Model(inputs=ri, outputs=[no])
  run_loop(m)

