from collections import namedtuple
from cereal import car
from common.realtime import DT_CTRL
from selfdrive.controls.lib.drive_helpers import rate_limit
from common.numpy_fast import clip
from selfdrive.car import create_gas_command
from selfdrive.car.honda import hondacan
from selfdrive.car.honda.values import CruiseButtons, CAR, VISUAL_HUD
from opendbc.can.packer import CANPacker

VisualAlert = car.CarControl.HUDControl.VisualAlert

def actuator_hystereses(brake, braking, brake_steady, v_ego, car_fingerprint):
  # hyst params
  brake_hyst_on = 0.02     # to activate brakes exceed this value
  brake_hyst_off = 0.005                     # to deactivate brakes below this value
  brake_hyst_gap = 0.01                      # don't change brake command for small oscillations within this value

  #*** hysteresis logic to avoid brake blinking. go above 0.1 to trigger
  if (brake < brake_hyst_on and not braking) or brake < brake_hyst_off:
    brake = 0.
  braking = brake > 0.

  # for small brake oscillations within brake_hyst_gap, don't change the brake command
  if brake == 0.:
    brake_steady = 0.
  elif brake > brake_steady + brake_hyst_gap:
    brake_steady = brake - brake_hyst_gap
  elif brake < brake_steady - brake_hyst_gap:
    brake_steady = brake + brake_hyst_gap
  brake = brake_steady

  if (car_fingerprint in (CAR.ACURA_ILX, CAR.CRV)) and brake > 0.0:
    brake += 0.15

  return brake, braking, brake_steady


def brake_pump_hysteresis(apply_brake, apply_brake_last, last_pump_ts, ts):
  pump_on = False

  # reset pump timer if:
  # - there is an increment in brake request
  # - we are applying steady state brakes and we haven't been running the pump
  #   for more than 20s (to prevent pressure bleeding)
  if apply_brake > apply_brake_last or (ts - last_pump_ts > 20. and apply_brake > 0):
    last_pump_ts = ts

  # once the pump is on, run it for at least 0.2s
  if ts - last_pump_ts < 0.2 and apply_brake > 0:
    pump_on = True

  return pump_on, last_pump_ts


def process_hud_alert(hud_alert):
  # initialize to no alert
  fcw_display = 0
  steer_required = 0
  acc_alert = 0

  # priority is: FCW, steer required, all others
  if hud_alert == VisualAlert.fcw:
    fcw_display = VISUAL_HUD[hud_alert.raw]
  elif hud_alert == VisualAlert.steerRequired:
    steer_required = VISUAL_HUD[hud_alert.raw]
  else:
    acc_alert = VISUAL_HUD[hud_alert.raw]

  return fcw_display, steer_required, acc_alert


HUDData = namedtuple("HUDData",
                     ["pcm_accel", "v_cruise",  "car",
                     "lanes", "fcw", "acc_alert", "steer_required"])


class CarController():
  def __init__(self, dbc_name, CP):
    self.braking = False
    self.brake_steady = 0.
    self.brake_last = 0.
    self.apply_brake_last = 0
    self.last_pump_ts = 0.
    self.packer = CANPacker(dbc_name)
    self.new_radar_config = False
    self.eps_modified = False
    for fw in CP.carFw:
      if fw.ecu == "eps" and b"," in fw.fwVersion:
        print("EPS FW MODIFIED!")
        self.eps_modified = True

  def update(self, enabled, CS, frame, actuators, \
             pcm_speed, pcm_override, pcm_cancel_cmd, pcm_accel, \
             hud_v_cruise, hud_show_lanes, hud_show_car, hud_alert):

    # *** apply brake hysteresis ***
    brake, self.braking, self.brake_steady = actuator_hystereses(actuators.brake, self.braking, self.brake_steady, CS.v_ego, CS.CP.carFingerprint)

    # *** no output if not enabled ***
    if not enabled and CS.pcm_acc_status:
      # send pcm acc cancel cmd if drive is disabled but pcm is still on, or if the system can't be activated
      pcm_cancel_cmd = True

    # *** rate limit after the enable check ***
    self.brake_last = rate_limit(brake, self.brake_last, -2., DT_CTRL)

    # vehicle hud display, wait for one update from 10Hz 0x304 msg
    if hud_show_lanes:
      hud_lanes = 1
    else:
      hud_lanes = 0

    if enabled:
      if hud_show_car:
        hud_car = 2
      else:
        hud_car = 1
    else:
      hud_car = 0

    fcw_display, steer_required, acc_alert = process_hud_alert(hud_alert)

    hud = HUDData(int(pcm_accel), int(round(hud_v_cruise)), hud_car,
                  hud_lanes, fcw_display, acc_alert, steer_required)

    # **** process the car messages ****

    # *** compute control surfaces ***
    BRAKE_MAX = 1024//4
    if CS.CP.carFingerprint in (CAR.ACURA_ILX):
      STEER_MAX = 0xF00
    elif CS.CP.carFingerprint in (CAR.CRV, CAR.ACURA_RDX):
      STEER_MAX = 0x3e8  # CR-V only uses 12-bits and requires a lower value
    elif CS.CP.carFingerprint in (CAR.ODYSSEY_CHN):
      STEER_MAX = 0x7FFF
    elif CS.CP.carFingerprint in (CAR.CIVIC) and self.eps_modified:
      STEER_MAX = 0x1400
    else:
      STEER_MAX = 0x1000

    # steer torque is converted back to CAN reference (positive when steering right)
    apply_gas = clip(actuators.gas, 0., 1.)
    apply_brake = int(clip(self.brake_last * BRAKE_MAX, 0, BRAKE_MAX - 1))
    apply_steer = int(clip(-actuators.steer * STEER_MAX, -STEER_MAX, STEER_MAX))

    if CS.CP.carFingerprint in (CAR.CIVIC) and self.eps_modified:
      if apply_steer > 0xA00:
        apply_steer = (apply_steer - 0xA00) / 2 + 0xA00
      elif apply_steer < -0xA00:
        apply_steer = (apply_steer + 0xA00) / 2 - 0xA00

    lkas_active = enabled and not CS.steer_not_allowed

    # Send CAN commands.
    can_sends = []

    # Send steering command.
    idx = frame % 4
    can_sends.append(hondacan.create_steering_control(self.packer, apply_steer,
      lkas_active, CS.CP.carFingerprint, idx, CS.CP.isPandaBlack))

    # Send dashboard UI commands.
    if (frame % 10) == 0:
      idx = (frame//10) % 4
      can_sends.extend(hondacan.create_ui_commands(self.packer, pcm_speed, hud, CS.CP.carFingerprint, CS.is_metric, idx, CS.CP.isPandaBlack, CS.stock_hud))

    if CS.CP.radarOffCan:
      # If using stock ACC, spam cancel command to kill gas when OP disengages.
      if pcm_cancel_cmd:
        can_sends.append(hondacan.spam_buttons_command(self.packer, CruiseButtons.CANCEL, idx, CS.CP.carFingerprint, CS.CP.isPandaBlack))
      elif CS.stopped:
        if CS.CP.carFingerprint in (CAR.CIVIC_BOSCH, CAR.CRV_HYBRID, CAR.CRV):
          if CS.hud_lead == 1:
            can_sends.append(hondacan.spam_buttons_command(self.packer, CruiseButtons.RES_ACCEL, idx, CS.CP.carFingerprint, CS.CP.isPandaBlack))
        else:
          can_sends.append(hondacan.spam_buttons_command(self.packer, CruiseButtons.RES_ACCEL, idx, CS.CP.carFingerprint, CS.CP.isPandaBlack))

    else:
      # Send gas and brake commands.
      if (frame % 2) == 0:
        idx = frame // 2
        ts = frame * DT_CTRL
        pump_on, self.last_pump_ts = brake_pump_hysteresis(apply_brake, self.apply_brake_last, self.last_pump_ts, ts)
        can_sends.append(hondacan.create_brake_command(self.packer, apply_brake, pump_on,
          pcm_override, pcm_cancel_cmd, hud.fcw, idx, CS.CP.carFingerprint, CS.CP.isPandaBlack, CS.stock_brake))
        self.apply_brake_last = apply_brake

        if CS.CP.enableGasInterceptor:
          # send exactly zero if apply_gas is zero. Interceptor will send the max between read value and apply_gas.
          # This prevents unexpected pedal range rescaling
          can_sends.append(create_gas_command(self.packer, apply_gas, idx))

    return can_sends
