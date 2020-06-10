#!/usr/bin/env python3
import math

import json
import numpy as np

import cereal.messaging as messaging
from cereal import car, log
from common.params import Params, put_nonblocking
from selfdrive.locationd.models.car_kf import CarKalman, ObservationKind, States
from selfdrive.locationd.models.constants import GENERATED_DIR
from selfdrive.swaglog import cloudlog

KalmanStatus = log.LiveLocationKalman.Status

CARSTATE_DECIMATION = 5


class ParamsLearner:
  def __init__(self, CP, steer_ratio, stiffness_factor, angle_offset):
    self.kf = CarKalman(GENERATED_DIR, steer_ratio, stiffness_factor, angle_offset)

    self.kf.filter.set_mass(CP.mass)  # pylint: disable=no-member
    self.kf.filter.set_rotational_inertia(CP.rotationalInertia)  # pylint: disable=no-member
    self.kf.filter.set_center_to_front(CP.centerToFront)  # pylint: disable=no-member
    self.kf.filter.set_center_to_rear(CP.wheelbase - CP.centerToFront)  # pylint: disable=no-member
    self.kf.filter.set_stiffness_front(CP.tireStiffnessFront)  # pylint: disable=no-member
    self.kf.filter.set_stiffness_rear(CP.tireStiffnessRear)  # pylint: disable=no-member

    self.active = False

    self.speed = 0
    self.steering_pressed = False
    self.steering_angle = 0
    self.carstate_counter = 0

    self.valid = True

  def handle_log(self, t, which, msg):
    if which == 'liveLocationKalman':

      yaw_rate = msg.angularVelocityCalibrated.value[2]
      yaw_rate_std = msg.angularVelocityCalibrated.std[2]

      if self.active:
        if msg.inputsOK and msg.posenetOK and msg.status == KalmanStatus.valid:
          self.kf.predict_and_observe(t,
                                      ObservationKind.ROAD_FRAME_YAW_RATE,
                                      np.array([[[-yaw_rate]]]),
                                      np.array([np.atleast_2d(yaw_rate_std**2)]))
        self.kf.predict_and_observe(t, ObservationKind.ANGLE_OFFSET_FAST, np.array([[[0]]]))

    elif which == 'carState':
      self.carstate_counter += 1
      if self.carstate_counter % CARSTATE_DECIMATION == 0:
        self.steering_angle = msg.steeringAngle
        self.steering_pressed = msg.steeringPressed
        self.speed = msg.vEgo

        in_linear_region = abs(self.steering_angle) < 45 or not self.steering_pressed
        self.active = self.speed > 5 and in_linear_region

        if self.active:
          self.kf.predict_and_observe(t, ObservationKind.STEER_ANGLE, np.array([[[math.radians(msg.steeringAngle)]]]))
          self.kf.predict_and_observe(t, ObservationKind.ROAD_FRAME_X_SPEED, np.array([[[self.speed]]]))

    if not self.active:
      # Reset time when stopped so uncertainty doesn't grow
      self.kf.filter.filter_time = t
      self.kf.filter.reset_rewind()


def main(sm=None, pm=None):
  if sm is None:
    sm = messaging.SubMaster(['liveLocationKalman', 'carState'])
  if pm is None:
    pm = messaging.PubMaster(['liveParameters'])

  params_reader = Params()
  # wait for stats about the car to come in from controls
  cloudlog.info("paramsd is waiting for CarParams")
  CP = car.CarParams.from_bytes(params_reader.get("CarParams", block=True))
  cloudlog.info("paramsd got CarParams")

  params = params_reader.get("LiveParameters")

  # Check if car model matches
  if params is not None:
    params = json.loads(params)
    if params.get('carFingerprint', None) != CP.carFingerprint:
      cloudlog.info("Parameter learner found parameters for wrong car.")
      params = None

  if params is None:
    params = {
      'carFingerprint': CP.carFingerprint,
      'steerRatio': CP.steerRatio,
      'stiffnessFactor': 1.0,
      'angleOffsetAverage': 0.0,
    }
    cloudlog.info("Parameter learner resetting to default values")

  learner = ParamsLearner(CP, params['steerRatio'], params['stiffnessFactor'], math.radians(params['angleOffsetAverage']))
  min_sr, max_sr = 0.5 * CP.steerRatio, 2.0 * CP.steerRatio

  while True:
    sm.update()

    for which, updated in sm.updated.items():
      if not updated:
        continue
      t = sm.logMonoTime[which] * 1e-9
      learner.handle_log(t, which, sm[which])

    if sm.updated['carState'] and (learner.carstate_counter % CARSTATE_DECIMATION == 0):
      msg = messaging.new_message('liveParameters')
      msg.logMonoTime = sm.logMonoTime['carState']

      msg.liveParameters.posenetValid = True
      msg.liveParameters.sensorValid = True

      x = learner.kf.x
      msg.liveParameters.steerRatio = float(x[States.STEER_RATIO])
      msg.liveParameters.stiffnessFactor = float(x[States.STIFFNESS])
      msg.liveParameters.angleOffsetAverage = math.degrees(x[States.ANGLE_OFFSET])
      msg.liveParameters.angleOffset = msg.liveParameters.angleOffsetAverage + math.degrees(x[States.ANGLE_OFFSET_FAST])
      msg.liveParameters.valid = all((
        abs(msg.liveParameters.angleOffsetAverage) < 10.0,
        abs(msg.liveParameters.angleOffset) < 10.0,
        0.2 <= msg.liveParameters.stiffnessFactor <= 5.0,
        min_sr <= msg.liveParameters.steerRatio <= max_sr,
      ))

      if learner.carstate_counter % 6000 == 0:   # once a minute
        params = {
          'carFingerprint': CP.carFingerprint,
          'steerRatio': msg.liveParameters.steerRatio,
          'stiffnessFactor': msg.liveParameters.stiffnessFactor,
          'angleOffsetAverage': msg.liveParameters.angleOffsetAverage,
        }
        put_nonblocking("LiveParameters", json.dumps(params))

      pm.send('liveParameters', msg)


if __name__ == "__main__":
  main()
