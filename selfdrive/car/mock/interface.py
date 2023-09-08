#!/usr/bin/env python3
from cereal import car
import cereal.messaging as messaging
from openpilot.selfdrive.car.interfaces import CarInterfaceBase

# mocked car interface for dashcam mode
class CarInterface(CarInterfaceBase):
  def __init__(self, CP, CarController, CarState):
    super().__init__(CP, CarController, CarState)

    self.speed = 0.
    self.sm = messaging.SubMaster(['gpsLocation', 'gpsLocationExternal'])

  @staticmethod
  def _get_params(ret, candidate, fingerprint, car_fw, experimental_long, docs):
    ret.carName = "mock"
    return ret

  def _update(self, c):
    self.sm.update(0)
    gps_sock = 'gpsLocationExternal' if self.sm.rcv_frame['gpsLocationExternal'] > 1 else 'gpsLocation'

    ret = car.CarState.new_message()
    ret.vEgo = self.speedself.sm[gps_sock].speed
    ret.vEgoRaw = self.speedself.sm[gps_sock].speed

    return ret

  def apply(self, c, now_nanos):
    actuators = car.CarControl.Actuators.new_message()
    return actuators, []
