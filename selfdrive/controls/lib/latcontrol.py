from selfdrive.controls.lib.pid import PIController
from common.numpy_fast import interp
from cereal import car

_DT = 0.01    # 100Hz
_DT_MPC = 0.05  # 20Hz
_NOISE_THRESHOLD = 1.2


def get_steer_max(CP, v_ego):
  return interp(v_ego, CP.steerMaxBP, CP.steerMaxV)


class LatControl(object):
  def __init__(self, CP):
    self.pid = PIController((CP.steerKpBP, CP.steerKpV),
                            (CP.steerKiBP, CP.steerKiV),
                            k_f=CP.steerKf, pos_limit=1.0)
    self.last_cloudlog_t = 0.0
    self.angle_steers_des = 0.
    self.angle_ff_ratio = 0.0
    self.angle_ff_gain = 1.0
    self.rate_ff_gain = 0.01
    self.average_angle_steers = 0.
    self.angle_steers_noise = _NOISE_THRESHOLD
    self.angle_ff_bp = [[0.5, 5.0],[0.0, 1.0]]

    # TODO: add the feedforward parameters to LiveParameters
    self.angle_ff_gain = 1.0
    self.rate_ff_gain = 0.02
    self.angle_ff_bp = [[0.5, 5.0],[0.0, 1.0]]

  def reset(self):
    self.pid.reset()

  def adjust_angle_gain(self):
    if self.pid.i > self.previous_integral:
      if self.pid.f > 0 and self.pid.i > 0:
        self.angle_ff_gain *= 1.0001
      else:
        self.angle_ff_gain *= 0.9999
    elif self.pid.i < self.previous_integral:
      if self.pid.f < 0 and self.pid.i < 0:
        self.angle_ff_gain *= 1.0001
      else:
        self.angle_ff_gain *= 0.9999
    self.previous_integral = self.pid.i

  def adjust_rate_gain(self, angle_steers):
    self.angle_steers_noise += 0.0001 * ((angle_steers - self.average_angle_steers)**2 - self.angle_steers_noise)
    if self.angle_steers_noise > _NOISE_THRESHOLD:
      self.rate_ff_gain *= 0.9999
    else:
      self.rate_ff_gain *= 1.0001

  def update(self, active, v_ego, angle_steers, steer_override, CP, VM, path_plan):
    if v_ego < 0.3 or not active:
      output_steer = 0.0
      self.pid.reset()
      self.previous_integral = 0.0
    else:
      # TODO: ideally we should interp, but for tuning reasons we keep the mpc solution
      # constant for 0.05s.
      #dt = min(cur_time - self.angle_steers_des_time, _DT_MPC + _DT) + _DT  # no greater than dt mpc + dt, to prevent too high extraps
      #self.angle_steers_des = self.angle_steers_des_prev + (dt / _DT_MPC) * (self.angle_steers_des_mpc - self.angle_steers_des_prev)
      self.angle_steers_des = path_plan.angleSteers  # get from MPC/PathPlanner

      steers_max = get_steer_max(CP, v_ego)
      self.pid.pos_limit = steers_max
      self.pid.neg_limit = -steers_max
      steer_feedforward = self.angle_steers_des   # feedforward desired angle
      if CP.steerControlType == car.CarParams.SteerControlType.torque:
        angle_feedforward = steer_feedforward - path_plan.angleOffset
        self.angle_ff_ratio = interp(abs(angle_feedforward), self.angle_ff_bp[0], self.angle_ff_bp[1])
        angle_feedforward *= self.angle_ff_ratio * self.angle_ff_gain
        rate_feedforward = (1.0 - self.angle_ff_ratio) * self.rate_ff_gain * path_plan.rateSteers
        steer_feedforward = v_ego**2 * (rate_feedforward + angle_feedforward)

        if not steer_override and v_ego > 10.0:
          if abs(angle_steers) > (self.angle_ff_bp[0][1] / 2.0):
            self.adjust_angle_gain()
          else:
            self.previous_integral = self.pid.i
            self.adjust_rate_gain(angle_steers)

      deadzone = 0.0
      output_steer = self.pid.update(self.angle_steers_des, angle_steers, check_saturation=(v_ego > 10), override=steer_override,
                                     feedforward=steer_feedforward, speed=v_ego, deadzone=deadzone)

    self.sat_flag = self.pid.saturated
    self.average_angle_steers += 0.01 * (angle_steers - self.average_angle_steers)
    return output_steer, float(self.angle_steers_des)
