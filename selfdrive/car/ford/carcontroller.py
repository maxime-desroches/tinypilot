import math
from cereal import car
from common.numpy_fast import clip, interp
from opendbc.can.packer import CANPacker
from selfdrive.car.ford import fordcan
from selfdrive.car.ford.values import CarControllerParams

VisualAlert = car.CarControl.HUDControl.VisualAlert


def apply_ford_steer_angle_limits(apply_angle, apply_angle_last, vEgo, VM):
  # rate limit
  steer_up = apply_angle_last * apply_angle > 0. and abs(apply_angle) > abs(apply_angle_last)
  rate_limit = CarControllerParams.RATE_LIMIT_UP if steer_up else CarControllerParams.RATE_LIMIT_DOWN
  max_angle_diff = interp(vEgo, rate_limit.speed_points, rate_limit.max_angle_diff_points)
  apply_angle = clip(apply_angle, (apply_angle_last - max_angle_diff), (apply_angle_last + max_angle_diff))

  # absolute limit (LatCtlPath_An_Actl)
  apply_path_angle = math.radians(apply_angle) / CarControllerParams.STEER_RATIO
  apply_path_angle = clip(apply_path_angle, -0.4995, 0.5240)
  apply_angle = math.degrees(apply_path_angle) * CarControllerParams.STEER_RATIO

  return apply_angle


class CarController:
  def __init__(self, dbc_name, CP, VM):
    self.CP = CP
    self.VM = VM
    self.packer = CANPacker(dbc_name)
    self.frame = 0

    self.apply_angle_last = 0
    self.main_on_last = False
    self.lkas_enabled_last = False
    self.steer_alert_last = False

  def update(self, CC, CS):
    can_sends = []

    actuators = CC.actuators
    hud_control = CC.hudControl

    main_on = CS.out.cruiseState.available
    steer_alert = hud_control.visualAlert in (VisualAlert.steerRequired, VisualAlert.ldw)

    if CC.cruiseControl.cancel:
      # cancel stock ACC
      can_sends.append(fordcan.create_button_command(self.packer, cancel=True))


    ### lateral control ###
    if CC.latActive:
      apply_angle = apply_ford_steer_angle_limits(actuators.steeringAngleDeg, self.apply_angle_last, CS.out.vEgo, self.VM)
    else:
      apply_angle = CS.out.steeringAngleDeg

    # send steering commands at 20Hz
    if (self.frame % CarControllerParams.LKAS_STEER_STEP) == 0:
      lca_rq = 1 if CC.latActive else 0

      # use LatCtlPath_An_Actl to actuate steering
      # path angle is the car wheel angle, not the steering wheel angle
      path_angle = math.radians(apply_angle) / CarControllerParams.STEER_RATIO

      # ramp rate: 0=Slow, 1=Medium, 2=Fast, 3=Immediately
      # TODO: slower ramp speed when driver torque detected
      ramp_type = 3  # 0=Slow, 1=Medium, 2=Fast, 3=Immediately
      precision = 1  # 0=Comfortable, 1=Precise

      self.apply_angle_last = apply_angle
      can_sends.append(fordcan.create_lka_command(self.packer, apply_angle, 0))
      can_sends.append(fordcan.create_tja_command(self.packer, lca_rq, ramp_type, precision,
                                                  0, path_angle, 0, 0))


    ### ui ###
    send_ui = (self.main_on_last != main_on) or (self.lkas_enabled_last != CC.latActive) or (self.steer_alert_last != steer_alert)

    # send lkas ui command at 1Hz or if ui state changes
    if (self.frame % CarControllerParams.LKAS_UI_STEP) == 0 or send_ui:
      can_sends.append(fordcan.create_lkas_ui_command(self.packer, main_on, CC.latActive, steer_alert, CS.lkas_status_stock_values))

    # send acc ui command at 20Hz or if ui state changes
    if (self.frame % CarControllerParams.ACC_UI_STEP) == 0 or send_ui:
      can_sends.append(fordcan.create_acc_ui_command(self.packer, main_on, CC.latActive, CS.acc_tja_status_stock_values))

    self.main_on_last = main_on
    self.lkas_enabled_last = CC.latActive
    self.steer_alert_last = steer_alert

    new_actuators = actuators.copy()
    new_actuators.steeringAngleDeg = apply_angle

    self.frame += 1
    return new_actuators, can_sends
