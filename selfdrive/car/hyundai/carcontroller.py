from cereal import car
from selfdrive.car import apply_std_steer_torque_limits
from selfdrive.car.hyundai.hyundaican import create_lkas11, create_clu11
from selfdrive.car.hyundai.values import Buttons, SteerLimitParams, CAR
from opendbc.can.packer import CANPacker

VisualAlert = car.CarControl.HUDControl.VisualAlert


def process_hud_alert(enabled, fingerprint, visual_alert, left_line,
                      right_line, left_lane_depart, right_lane_depart):
  hud_alert = (visual_alert == VisualAlert.steerRequired)

  # initialize to no line visible
  lane_visible = 1
  if left_line and right_line or hud_alert:  #HUD alert only display when LKAS status is active
    if enabled or hud_alert:
      lane_visible = 3
    else:
      lane_visible = 4
  elif left_line:
    lane_visible = 5
  elif right_line:
    lane_visible = 6

  # initialize to no warnings
  left_lane_warning = 0
  right_lane_warning = 0
  if left_lane_depart:
    left_lane_warning = 1 if fingerprint in [CAR.GENESIS_G90, CAR.GENESIS_G80] else 2
  if right_lane_depart:
    right_lane_warning = 1 if fingerprint in [CAR.GENESIS_G90, CAR.GENESIS_G80] else 2

  return hud_alert, lane_visible, left_lane_warning, right_lane_warning


class CarController():
  def __init__(self, dbc_name, CP, VM):
    self.apply_steer_last = 0
    self.car_fingerprint = CP.carFingerprint
    self.packer = CANPacker(dbc_name)
    self.steer_rate_limited = False
    self.resume_cnt = 0
    self.last_resume_frame = 0
    self.last_lead_distance = 0

  def update(self, enabled, CS, frame, actuators, pcm_cancel_cmd, visual_alert,
             left_line, right_line, left_lane_depart, right_lane_depart):
    # Steering Torque
    new_steer = actuators.steer * SteerLimitParams.STEER_MAX
    apply_steer = apply_std_steer_torque_limits(new_steer, self.apply_steer_last, CS.out.steeringTorque, SteerLimitParams)
    self.steer_rate_limited = new_steer != apply_steer

    # disable if steer angle reach 90 deg, otherwise mdps fault in some models
    lkas_active = enabled and abs(CS.out.steeringAngle) < 90.

    # fix for Genesis hard fault at low speed
    if CS.out.vEgo < 16.7 and self.car_fingerprint == CAR.HYUNDAI_GENESIS:
      lkas_active = 0

    if not lkas_active:
      apply_steer = 0

    self.apply_steer_last = apply_steer

    hud_alert, lane_visible, left_lane_warning, right_lane_warning =\
      process_hud_alert(enabled, self.car_fingerprint, visual_alert,
                        left_line, right_line, left_lane_depart, right_lane_depart)

    can_sends = []
    can_sends.append(create_lkas11(self.packer, frame, self.car_fingerprint, apply_steer, lkas_active,
                                   CS.lkas11, hud_alert, lane_visible, left_lane_warning, right_lane_warning))

    if pcm_cancel_cmd:
      can_sends.append(create_clu11(self.packer, frame, CS.clu11, Buttons.CANCEL))

    elif CS.out.cruiseState.standstill:
      # run only first time when the car stopped
      if self.last_lead_distance == 0:
        # get the lead distance from the Radar
        self.last_lead_distance = CS.lead_distance
        self.resume_cnt = 0
      # when lead car starts moving, create 6 RES msgs
      elif CS.lead_distance != self.last_lead_distance and (frame - self.last_resume_frame) > 5:
        can_sends.append(create_clu11(self.packer, CS.clu11, Buttons.RES_ACCEL))
        self.resume_cnt += 1
        # interval after 6 msgs
        if self.resume_cnt > 5:
          self.last_resume_frame = frame
          self.clu11_cnt = 0
    # reset lead distnce after the car starts moving
    elif self.last_lead_distance != 0:
      self.last_lead_distance = 0


    return can_sends
