#!/usr/bin/env python3
import os
import time

import cereal.messaging as messaging

from cereal import car, log

from panda import ALTERNATIVE_EXPERIENCE

from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process, Priority, Ratekeeper, DT_CTRL
from openpilot.common.swaglog import cloudlog

from openpilot.selfdrive.boardd.boardd import can_list_to_can_capnp
from openpilot.selfdrive.car.car_helpers import get_car, get_one_can
from openpilot.selfdrive.car.interfaces import CarInterfaceBase
from openpilot.selfdrive.controls.lib.drive_helpers import VCruiseHelper
from openpilot.selfdrive.controls.lib.events import Events

REPLAY = "REPLAY" in os.environ

State = log.ControlsState.OpenpilotState
EventName = car.CarEvent.EventName
ButtonType = car.CarState.ButtonEvent.Type


class Car:
  CI: CarInterfaceBase

  def __init__(self, CI=None):
    self.POLL = False

    self.can_sock = messaging.sub_sock('can', timeout=20)
    self.sm = messaging.SubMaster(['pandaStates', 'carControl', 'controlsState'],
                                  poll='carControl' if self.POLL else None)
    self.pm = messaging.PubMaster(['sendcan', 'carState', 'carParams', 'carOutput'])

    self.can_rcv_timeout_counter = 0  # consecutive timeout count
    self.can_rcv_cum_timeout_counter = 0  # cumulative timeout count

    self.CC_prev = car.CarControl.new_message()
    self.CS_prev = car.CarState.new_message()
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

    # read params
    self.is_metric = self.params.get_bool("IsMetric")

    # set alternative experiences from parameters
    self.disengage_on_accelerator = self.params.get_bool("DisengageOnAccelerator")
    self.CP.alternativeExperience = 0
    if not self.disengage_on_accelerator:
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

    self.events = Events()
    self.v_cruise_helper = VCruiseHelper(self.CP)

    # card is driven by can recv, expected at 100Hz
    self.rk = Ratekeeper(100, print_delay_threshold=None)

  def state_update(self) -> car.CarState:
    """carState update loop, driven by can"""

    # Update carState from CAN
    can_strs = messaging.drain_sock_raw(self.can_sock, wait_for_one=True)
    cloudlog.timestamp('Received can')
    CS = self.CI.update(self.CC_prev, can_strs)

    if not self.POLL:
      self.sm.update(0)

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

    if self.sm['controlsState'].initialized and not self.controlsState_prev.initialized:
      self.CI.init(self.CP, self.can_sock, self.pm.sock['sendcan'])
      cloudlog.timestamp("Initialized")

    return CS

  def update_events(self, CS: car.CarState) -> car.CarState:
    self.events.clear()

    self.events.add_from_msg(CS.events)

    # Block resume if cruise never previously enabled
    resume_pressed = any(be.type in (ButtonType.accelCruise, ButtonType.resumeCruise) for be in CS.buttonEvents)
    if not self.CP.pcmCruise and not self.v_cruise_helper.v_cruise_initialized and resume_pressed:
      self.events.add(EventName.resumeBlocked)

    # Disable on rising edge of accelerator or brake. Also disable on brake when speed > 0
    if (CS.gasPressed and not self.CS_prev.gasPressed and self.disengage_on_accelerator) or \
      (CS.brakePressed and (not self.CS_prev.brakePressed or not CS.standstill)) or \
      (CS.regenBraking and (not self.CS_prev.regenBraking or not CS.standstill)):
      self.events.add(EventName.pedalPressed)

    CS.events = self.events.to_msg()

  def state_transition(self, CS: car.CarState):
    self.v_cruise_helper.update_v_cruise(CS, self.sm['controlsState'].enabled, self.is_metric)

    controlsState = self.sm['controlsState']
    if self.controlsState_prev.state == State.disabled:
      # TODO: use ENABLED_STATES from controlsd? it includes softDisabling which isn't possible here
      if controlsState.state in (State.preEnabled, State.overriding, State.enabled):
       self.v_cruise_helper.initialize_v_cruise(CS, controlsState.experimentalMode)

  def state_publish(self, CS: car.CarState):
    """carState and carParams publish loop"""

    # carParams - logged every 50 seconds (> 1 per segment)
    if self.sm.frame % int(50. / DT_CTRL) == 0:
      cp_send = messaging.new_message('carParams')
      cp_send.valid = True
      cp_send.carParams = self.CP
      self.pm.send('carParams', cp_send)

    # publish new carOutput
    co_send = messaging.new_message('carOutput')
    co_send.valid = self.sm.all_checks(['carControl'])
    co_send.carOutput.actuatorsOutput = self.last_actuators_output
    self.pm.send('carOutput', co_send)

    # kick off controlsd step now while we actuate the latest carControl packet
    cs_send = messaging.new_message('carState')
    cs_send.valid = CS.canValid
    cs_send.carState = CS
    cs_send.carState.canRcvTimeout = self.can_rcv_timeout
    cs_send.carState.canErrorCounter = self.can_rcv_cum_timeout_counter
    cs_send.carState.cumLagMs = -self.rk.remaining * 1000.
    self.pm.send('carState', cs_send)
    cloudlog.timestamp('Sent carState')

    if self.POLL:
      # wait for latest carControl
      self.sm.update(20)

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
    cloudlog.timestamp("Start card")
    CS = self.state_update()
    cloudlog.timestamp("State updated")

    self.update_events(CS)

    if not self.CP.passive and self.sm['controlsState'].initialized:
      self.state_transition(CS)

    self.state_publish(CS)
    cloudlog.timestamp("State published")

    controlsState = self.sm['controlsState']
    if not self.CP.passive and controlsState.initialized:
      self.controls_update(CS, self.sm['carControl'])
      cloudlog.timestamp("Controls updated")

    self.controlsState_prev = controlsState
    self.CS_prev = CS.as_reader()

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
