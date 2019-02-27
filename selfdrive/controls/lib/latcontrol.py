import zmq
import math
from selfdrive.services import service_list
import selfdrive.messaging as messaging
from selfdrive.controls.lib.pid import PIController
from common.numpy_fast import interp
from cereal import car

_DT = 0.01    # 100Hz
_DT_MPC = 0.05  # 20Hz


def get_steer_max(CP, v_ego):
  return interp(v_ego, CP.steerMaxBP, CP.steerMaxV)


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
      steer_feedforward = self.angle_steers_des   # feedforward desired angle

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

    dat = messaging.new_message()
    dat.init('latControl')
    dat.latControl.anglelater = math.degrees(list(self.mpc_solution[0].delta)[-1])
    self.latControl_sock.send(dat.to_bytes())

    # ALCA works better with the non-interpolated angle
    if CP.steerControlType == car.CarParams.SteerControlType.torque:
      return output_steer, float(self.angle_steers_des_mpc)
    else:
      return float(self.angle_steers_des_mpc), float(self.angle_steers_des)
