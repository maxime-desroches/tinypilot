#!/usr/bin/env python
from __future__ import print_function

import os
import numpy as np
import tempfile
import zmq
import json

from selfdrive.swaglog import cloudlog
from common.services import service_list
import selfdrive.messaging as messaging
from selfdrive.config import ImageParams
from cereal import car
from common.params import Params
from selfdrive.calibrationd.calibration import ViewCalibrator, CalibStatus

def load_calibration(gctx):
  # calibration initialization
  I = ImageParams()
  vp_guess = None

  if gctx is not None:
    warp_matrix_start = np.array(
        gctx['calibration']["initial_homography"]).reshape(3, 3)
    big_box_size = [560, 304]
  else:
    warp_matrix_start = np.array([[1., 0., I.SX_R],
                                  [0., 1., I.SY_R],
                                  [0., 0., 1.]])
    big_box_size = [640, 480]

  # translate the vanishing point into phone image space
  vp_box = (I.VPX_R-I.SX_R, I.VPY_R-I.SY_R)
  vp_trans = np.dot(warp_matrix_start, vp_box+(1.,))
  vp_img = (vp_trans[0]/vp_trans[2], vp_trans[1]/vp_trans[2])

  # load calibration data
  try:
    # If the calibration file exist, start from the last cal values
    data = json.loads(Params().get("CalibrationParams"))
    return ViewCalibrator(
        (I.X, I.Y),
        big_box_size,
        vp_img,
        warp_matrix_start,
        vp_f=[data[2], data[3]],
        cal_cycle=data[0],
        cal_status=data[1])
  except Exception as e:
    print("Could not load calibration file: {}".format(e))

  return ViewCalibrator(
    (I.X, I.Y), big_box_size, vp_img, warp_matrix_start, vp_f=vp_guess)

def store_calibration(calib):
  cal_string = json.dumps([calib.cal_cycle, calib.cal_status, calib.vp_f[0], calib.vp_f[1]])
  cloudlog.event("store_calibration", cal_string=cal_string)
  Params().put("CalibrationParams", cal_string)

def calibrationd_thread(gctx):
  context = zmq.Context()

  features = messaging.sub_sock(context, service_list['features'].port)
  live100 = messaging.sub_sock(context, service_list['live100'].port)

  livecalibration = messaging.pub_sock(context, service_list['liveCalibration'].port)

  # subscribe to stats about the car
  cloudlog.info("calibrationd is waiting for CarParams")
  CP = car.CarParams.from_bytes(Params().get("CarParams", block=True))
  cloudlog.info("calibrationd got CarParams")

  v_ego = None

  calib = load_calibration(gctx)
  last_write_cycle = None

  while 1:
    # calibration at the end so it does not delay radar processing above
    ft = messaging.recv_sock(features, wait=True)

    # get latest here
    l100 = messaging.recv_sock(live100)
    if l100 is not None:
      v_ego = l100.live100.vEgo
      steer_angle = l100.live100.angleSteers

    if v_ego is None:
      continue

    p0 = ft.features.p0
    p1 = ft.features.p1
    st = ft.features.status

    calib.calibration(p0, p1, st, v_ego, steer_angle, CP)

    # write a new calibration every 100 cal cycle
    if last_write_cycle is None or calib.cal_cycle - last_write_cycle >= 100:
      print("writing cal", calib.cal_cycle)
      store_calibration(calib)
      last_write_cycle = calib.cal_cycle

    warp_matrix = map(float, calib.warp_matrix.reshape(9).tolist())
    dat = messaging.new_message()
    dat.init('liveCalibration')
    dat.liveCalibration.warpMatrix = warp_matrix
    dat.liveCalibration.calStatus = calib.cal_status
    dat.liveCalibration.calCycle = calib.cal_cycle
    dat.liveCalibration.calPerc = calib.cal_perc
    livecalibration.send(dat.to_bytes())

def main(gctx=None):
  calibrationd_thread(gctx)

if __name__ == "__main__":
  main()
