import zmq
from selfdrive.services import service_list
import selfdrive.messaging as messaging
from selfdrive.controls.lib.pid import PIController
from common.numpy_fast import interp
from cereal import car

_DT = 0.01    # 100Hz
_DT_MPC = 0.05  # 20Hz


def get_steer_max(CP, v_ego):
  return interp(v_ego, CP.steerMaxBP, CP.steerMaxV)

def apply_deadzone(angle, deadzone):
  if angle > deadzone:
    angle -= deadzone
  elif angle < -deadzone:
    angle += deadzone
  else:
    angle = 0.
  return angle


class LatControl(object):
  def __init__(self, CP):
    self.pid = PIController((CP.steerKpBP, CP.steerKpV),
                            (CP.steerKiBP, CP.steerKiV),
                            k_f=CP.steerKf, pos_limit=1.0)
    self.last_cloudlog_t = 0.0

    context = zmq.Context()
    self.latControl_sock = messaging.pub_sock(context, service_list['latControl'].port)
    self.blindspot_blink_counter_right_check = 0
    self.blindspot_blink_counter_left_check = 0
    self.angle_steers_des = 0.

  def reset(self):
    self.pid.reset()

  def update(self, active, v_ego, angle_steers, steer_override, CP, VM, path_plan,blindspot,leftBlinker,rightBlinker):
    if v_ego < 0.3 or not active:
      output_steer = 0.0
      self.feed_forward = 0.0
      self.pid.reset()
      self.angle_steers_des = angle_steers
      self.avg_angle_steers = angle_steers
      self.cur_state[0].delta = math.radians(angle_steers - angle_offset) / CP.steerRatio
    else:
      # TODO: ideally we should interp, but for tuning reasons we keep the mpc solution
      # constant for 0.05s.
      #dt = min(cur_time - self.angle_steers_des_time, _DT_MPC + _DT) + _DT  # no greater than dt mpc + dt, to prevent too high extraps
      #self.angle_steers_des = self.angle_steers_des_prev + (dt / _DT_MPC) * (self.angle_steers_des_mpc - self.angle_steers_des_prev)
      self.angle_steers_des = path_plan.angleSteers  # get from MPC/PathPlanner

      steers_max = get_steer_max(CP, v_ego)
      self.pid.pos_limit = steers_max
      self.pid.neg_limit = -steers_max

      if CP.steerControlType == car.CarParams.SteerControlType.torque:
        # Decide which feed forward mode should be used (angle or rate).  Use more dominant mode, and only if conditions are met
        # Spread feed forward out over a period of time to make it more inductive (for resonance)
        if abs(self.ff_rate_factor * float(restricted_steer_rate)) > abs(self.ff_angle_factor * float(self.angle_steers_des) - float(angle_offset)) - 0.5 \
            and (abs(float(restricted_steer_rate)) > abs(angle_rate) or (float(restricted_steer_rate) < 0) != (angle_rate < 0)) \
            and (float(restricted_steer_rate) < 0) == (float(self.angle_steers_des) - float(angle_offset) - 0.5 < 0):
          self.feed_forward = (((self.smooth_factor - 1.) * self.feed_forward) + self.ff_rate_factor * v_ego**2 * float(restricted_steer_rate)) / self.smooth_factor
        elif abs(self.angle_steers_des - float(angle_offset)) > 0.5:
          self.feed_forward = (((self.smooth_factor - 1.) * self.feed_forward) + self.ff_angle_factor * v_ego**2 * float(apply_deadzone(float(self.angle_steers_des) - float(angle_offset), 0.5))) / self.smooth_factor
        else:
          self.feed_forward = (((self.smooth_factor - 1.) * self.feed_forward) + 0.0) / self.smooth_factor

      # Use projected desired and actual angles instead of "current" values, in order to make PI more reactive (for resonance)
      output_steer = self.pid.update(projected_angle_steers_des, self.projected_angle_steers, check_saturation=(v_ego > 10), override=steer_override,
                                     feedforward=self.feed_forward, speed=v_ego, deadzone=deadzone)

      if rightBlinker:
        if blindspot:
          self.blindspot_blink_counter_right_check = 0
          print "debug: blindspot detected"
        self.blindspot_blink_counter_right_check += 1
        if self.blindspot_blink_counter_right_check > 150:
          self.angle_steers_des -= 0#15

      else:
        self.blindspot_blink_counter_right_check = 0

      if leftBlinker:
        if blindspot:
          self.blindspot_blink_counter_left_check = 0
          print "debug: blindspot detected"
        self.blindspot_blink_counter_left_check += 1
        if self.blindspot_blink_counter_left_check > 150:
          self.angle_steers_des += 0#15
      else:
        self.blindspot_blink_counter_left_check = 0
      # Hide angle error if being overriden
      if steer_override:
        self.projected_angle_steers = self.mpc_angles[1]
        self.avg_angle_steers = self.mpc_angles[1]
        
    self.sat_flag = self.pid.saturated
    self.prev_angle_rate = angle_rate
    dat = messaging.new_message()
    dat.init('latControl')
    dat.latControl.anglelater = math.degrees(list(self.mpc_solution[0].delta)[-1])
    self.latControl_sock.send(dat.to_bytes())

    # ALCA works better with the non-interpolated angle
    if CP.steerControlType == car.CarParams.SteerControlType.torque:
      return output_steer, float(self.angle_steers_des_mpc)
    else:
      return float(self.angle_steers_des_mpc), float(self.angle_steers_des)
