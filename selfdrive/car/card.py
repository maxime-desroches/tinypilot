#!/usr/bin/env python3
import os
import time

import cereal.messaging as messaging

from cereal import car

from panda import ALTERNATIVE_EXPERIENCE

from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process, Priority, Ratekeeper, DT_CTRL
from openpilot.common.swaglog import cloudlog

from openpilot.selfdrive.boardd.boardd import can_list_to_can_capnp
from openpilot.selfdrive.car.car_helpers import get_car, get_one_can
from openpilot.selfdrive.car.interfaces import CarInterfaceBase

REPLAY = "REPLAY" in os.environ


class Car:
  CI: CarInterfaceBase

  def __init__(self, CI=None):
    self.can_sock = messaging.sub_sock('can', timeout=20)
    self.sm = messaging.SubMaster(['pandaStates', 'carControl', 'controlsState'], poll='carControl')
    self.pm = messaging.PubMaster(['sendcan', 'carState', 'carParams', 'carOutput'])

    self.can_rcv_timeout_counter = 0  # consecutive timeout count
    self.can_rcv_cum_timeout_counter = 0  # cumulative timeout count

    self.CC_prev = car.CarControl.new_message()
    self.controlsState_prev = car.CarState.new_message()

    self.last_actuators_output = car.CarControl.Actuators.new_message()

    self.params = Params()

    if CI is None:
      # wait for one pandaState and one CAN packet
      print("Waiting for CAN messages...")
      get_one_can(self.can_sock)

      num_pandas = len(messaging.recv_one_retry(self.sm.sock['pandaStates']).pandaStates)
      experimental_long_allowed = self.params.get_bool("ExperimentalLongitudinalEnabled")
      self.CI, self.CP = get_car(self.can_sock, self.pm.sock['sendcan'], experimental_long_allowed, num_pandas)
    else:
      self.CI, self.CP = CI, CI.CP

    # set alternative experiences from parameters
    disengage_on_accelerator = self.params.get_bool("DisengageOnAccelerator")
    self.CP.alternativeExperience = 0
    if not disengage_on_accelerator:
      self.CP.alternativeExperience |= ALTERNATIVE_EXPERIENCE.DISABLE_DISENGAGE_ON_GAS

    openpilot_enabled_toggle = self.params.get_bool("OpenpilotEnabledToggle")

    controller_available = self.CI.CC is not None and openpilot_enabled_toggle and not self.CP.dashcamOnly

    self.CP.passive = not controller_available or self.CP.dashcamOnly
    if self.CP.passive:
      safety_config = car.CarParams.SafetyConfig.new_message()
      safety_config.safetyModel = car.CarParams.SafetyModel.noOutput
      self.CP.safetyConfigs = [safety_config]

    # Write previous route's CarParams
    prev_cp = self.params.get("CarParamsPersistent")
    if prev_cp is not None:
      self.params.put("CarParamsPrevRoute", prev_cp)

    # Write CarParams for controls and radard
    cp_bytes = self.CP.to_bytes()
    self.params.put("CarParams", cp_bytes)
    self.params.put_nonblocking("CarParamsCache", cp_bytes)
    self.params.put_nonblocking("CarParamsPersistent", cp_bytes)

    # card is driven by can recv, expected at 100Hz
    self.rk = Ratekeeper(100, print_delay_threshold=None)

  def state_update(self):
    """carState update loop, driven by can"""

    # Update carState from CAN
    can_strs = messaging.drain_sock_raw(self.can_sock, wait_for_one=True)
    CS = self.CI.update(self.CC_prev, can_strs)

    can_rcv_valid = len(can_strs) > 0

    # Check for CAN timeout
    if not can_rcv_valid:
      self.can_rcv_timeout_counter += 1
      self.can_rcv_cum_timeout_counter += 1
    else:
      self.can_rcv_timeout_counter = 0

    self.can_rcv_timeout = self.can_rcv_timeout_counter >= 5

    if can_rcv_valid and REPLAY:
      self.can_log_mono_time = messaging.log_from_bytes(can_strs[0]).logMonoTime

    # carState is sent last to ensure controlsd receives above services this cycle
    cs_send = messaging.new_message('carState')
    cs_send.valid = CS.canValid
    cs_send.carState = CS
    cs_send.carState.canRcvTimeout = self.can_rcv_timeout
    cs_send.carState.canErrorCounter = self.can_rcv_cum_timeout_counter
    cs_send.carState.cumLagMs = -self.rk.remaining * 1000.
    self.pm.send('carState', cs_send)
    cloudlog.timestamp('Sent carState')

    # Wait for carControl response from controlsd
    self.sm.update(20)

    return CS

  def state_publish(self, CS: car.CarState):
    """carState and carParams publish loop"""

    # carParams - logged every 50 seconds (> 1 per segment)
    if (self.sm.frame % int(50. / DT_CTRL) == 0):
      cp_send = messaging.new_message('carParams')
      cp_send.valid = True
      cp_send.carParams = self.CP
      self.pm.send('carParams', cp_send)

    # publish new carOutput
    co_send = messaging.new_message('carOutput')
    co_send.valid = self.sm.all_checks(['carControl'])
    co_send.carOutput.actuatorsOutput = self.last_actuators_output
    self.pm.send('carOutput', co_send)

    # # carState is sent last to ensure controlsd receives above services this cycle
    # cs_send = messaging.new_message('carState')
    # cs_send.valid = CS.canValid
    # cs_send.carState = CS
    # cs_send.carState.canRcvTimeout = self.can_rcv_timeout
    # cs_send.carState.canErrorCounter = self.can_rcv_cum_timeout_counter
    # cs_send.carState.cumLagMs = -self.rk.remaining * 1000.
    # self.pm.send('carState', cs_send)

  def controls_update(self, CS: car.CarState, CC: car.CarControl):
    """control update loop, driven by carControl"""

    if self.sm.all_checks(['carControl']):
      # send car controls over can
      now_nanos = self.can_log_mono_time if REPLAY else int(time.monotonic() * 1e9)
      # TODO: CC shouldn't be builder
      self.last_actuators_output, can_sends = self.CI.apply(CC.as_builder(), now_nanos)
      self.pm.send('sendcan', can_list_to_can_capnp(can_sends, msgtype='sendcan', valid=CS.canValid))

      self.CC_prev = CC

  def step(self):
    controlsState = self.sm['controlsState']
    if controlsState.initialized and not self.controlsState_prev.initialized:
      self.CI.init(self.CP, self.can_sock, self.pm.sock['sendcan'])
      cloudlog.timestamp("Initialized")

    CS = self.state_update()
    cloudlog.timestamp("State updated")

    if not self.CP.passive and controlsState.initialized:
      self.controls_update(CS, self.sm['carControl'])
      cloudlog.timestamp("Controls updated")

    # TODO: this should be moved back up, note that carOutput will be delayed a frame
    self.state_publish(CS)
    cloudlog.timestamp("State published")

    self.controlsState_prev = controlsState

  def card_thread(self):
    while True:
      self.step()
      self.rk.monitor_time()


def main():
  config_realtime_process(4, Priority.CTRL_HIGH)
  car = Car()
  car.card_thread()


if __name__ == "__main__":
  main()
