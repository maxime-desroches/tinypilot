import zmq
import math
import numpy as np
import time
import json
from selfdrive.controls.lib.pid import PIController
from selfdrive.controls.lib.drive_helpers import MPC_COST_LAT
from selfdrive.controls.lib.lateral_mpc import libmpc_py
from common.numpy_fast import interp
from common.realtime import sec_since_boot
from selfdrive.swaglog import cloudlog
from cereal import car

_DT = 0.01    # 100Hz
_DT_MPC = 0.05  # 20Hz


def calc_states_after_delay(states, v_ego, steer_angle, curvature_factor, steer_ratio, delay):
  states[0].x = v_ego * delay
  states[0].psi = v_ego * curvature_factor * math.radians(steer_angle) / steer_ratio * delay
  return states


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
  def __init__(self, VM):
    self.pid = PIController((VM.CP.steerKpBP, VM.CP.steerKpV),
                            (VM.CP.steerKiBP, VM.CP.steerKiV),
                            k_f=VM.CP.steerKf, pos_limit=1.0)
    self.last_cloudlog_t = 0.0
    self.setup_mpc(VM.CP.steerRateCost)

  def setup_mpc(self, steer_rate_cost):
    self.libmpc = libmpc_py.libmpc
    self.libmpc.init(MPC_COST_LAT.PATH, MPC_COST_LAT.LANE, MPC_COST_LAT.HEADING, steer_rate_cost)

    self.mpc_solution = libmpc_py.ffi.new("log_t *")
    self.cur_state = libmpc_py.ffi.new("state_t *")
    self.mpc_updated = False
    self.mpc_nans = False
    self.cur_state[0].x = 0.0
    self.cur_state[0].y = 0.0
    self.cur_state[0].psi = 0.0
    self.cur_state[0].delta = 0.0

    self.last_mpc_ts = 0.0
    self.angle_steers_des = 0.0
    self.angle_steers_des_mpc = 0.0
    self.angle_steers_des_prev = 0.0
    self.angle_steers_des_time = 0.0
    self.context = zmq.Context()
    self.steerpub = self.context.socket(zmq.PUB)
    self.steerpub.bind("tcp://*:8594")
    self.steerdata = ""
    self.steerpub2 = self.context.socket(zmq.PUB)
    self.steerpub2.bind("tcp://*:8596")
    self.steerdata2 = ""
    self.ratioExp = 2.6
    self.ratioScale = 10.
    self.steer_steps = [0., 0., 0., 0., 0.]
    self.probFactor = 0.
    self.prev_output_steer = 0.
    self.rough_angle_array = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
    self.steer_speed_array = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
    self.tiny_angle_array = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    self.steer_torque_array = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
    self.steer_torque_count = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    self.tiny_torque_array = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    self.tiny_torque_count = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    self.center_angle = 0.
    self.center_count = 0
    self.save_steering = False
    self.steer_zero_crossing = 0.0
    self.steer_initialized = False
    self.smooth_angle_steers = 0.0

  def reset(self):
    self.pid.reset()

  def update(self, active, v_ego, angle_steers, steer_override, d_poly, angle_offset, VM, PL):
    cur_time = sec_since_boot()
    self.mpc_updated = False

    smoothing = 0.0
    self.smooth_angle_steers = (smoothing * self.smooth_angle_steers + angle_steers) / (smoothing + 1.0)
    ratioFactor = max(0.1, 1. - self.ratioScale * abs(self.smooth_angle_steers / 100.) ** self.ratioExp)
    cur_Steer_Ratio = VM.CP.steerRatio * ratioFactor

    # TODO: this creates issues in replay when rewinding time: mpc won't run
    if self.last_mpc_ts < PL.last_md_ts:
      self.last_mpc_ts = PL.last_md_ts
      self.angle_steers_des_prev = self.angle_steers_des_mpc

      curvature_factor = VM.curvature_factor(v_ego)

      self.l_poly = libmpc_py.ffi.new("double[4]", list(PL.PP.l_poly))
      self.r_poly = libmpc_py.ffi.new("double[4]", list(PL.PP.r_poly))
      self.p_poly = libmpc_py.ffi.new("double[4]", list(PL.PP.p_poly))

      # account for actuation delay
      self.cur_state = calc_states_after_delay(self.cur_state, v_ego, self.smooth_angle_steers, curvature_factor, cur_Steer_Ratio, VM.CP.steerActuatorDelay)

      v_ego_mpc = max(v_ego, 5.0)  # avoid mpc roughness due to low speed
      self.libmpc.run_mpc(self.cur_state, self.mpc_solution,
                          self.l_poly, self.r_poly, self.p_poly,
                          PL.PP.l_prob, PL.PP.r_prob, PL.PP.p_prob, curvature_factor, v_ego_mpc, PL.PP.lane_width)

      # reset to current steer angle if not active or overriding
      if active:
        self.isActive = 1
        delta_desired = self.mpc_solution[0].delta[1]
      else:
        self.isActive = 0
        delta_desired = math.radians(self.smooth_angle_steers - angle_offset) / cur_Steer_Ratio

      self.cur_state[0].delta = delta_desired

      self.angle_steers_des_mpc = float(math.degrees(delta_desired * cur_Steer_Ratio) + angle_offset)
      self.angle_steers_des_time = cur_time
      self.mpc_updated = True

      #  Check for infeasable MPC solution
      self.mpc_nans = np.any(np.isnan(list(self.mpc_solution[0].delta)))
      t = sec_since_boot()
      if self.mpc_nans:
        self.libmpc.init(MPC_COST_LAT.PATH, MPC_COST_LAT.LANE, MPC_COST_LAT.HEADING, VM.CP.steerRateCost)
        self.cur_state[0].delta = math.radians(self.smooth_angle_steers) / cur_Steer_Ratio

        if t > self.last_cloudlog_t + 5.0:
          self.last_cloudlog_t = t
          cloudlog.warning("Lateral mpc - nan: True")

    if self.steerdata != "" and int(cur_time * 100) % 50 == 4:
      self.steerpub.send(self.steerdata)
      self.steerdata = ""

    if (int(cur_time * 100) % 200) == 0:  
      for i in range(21):
        if self.steer_torque_count[i] > 0:
          self.steerdata2 += 'steerTune,type=%s,angleTag=%d angle=%d,value=%f,speed=%d,count=%d\n' % ('rough', self.rough_angle_array[i], self.rough_angle_array[i], self.steer_torque_array[i], self.steer_speed_array[i], self.steer_torque_count[i])
      if len(self.steerdata2) > 0:
        self.steerpub2.send(self.steerdata2)
        self.steerdata2 = ""

    if v_ego < 0.3 or not active:
      output_steer = 0.0
      self.prev_output_steer = 0
      self.pid.reset()
      if self.save_steering:
        file = open("/sdcard/realdata/steering/gernby.dat","w")
        file.write(json.dumps([self.steer_torque_array, self.steer_speed_array, self.steer_torque_count]))
        file.close()
        self.save_steering = False
      elif not self.steer_initialized:
        self.steer_initialized = True
        file = open("/sdcard/realdata/steering/gernby.dat","r")
        self.steer_torque_array, self.steer_speed_array, self.steer_torque_count = json.loads(file.read())
        for i in range(21):
          self.rough_angle_array[i] = i - 11
        print (self.steer_torque_array)
        print (self.steer_speed_array)
        print (self.steer_torque_count)
    else:
      # TODO: ideally we should interp, but for tuning reasons we keep the mpc solution
      # constant for 0.05s.
      #dt = min(cur_time - self.angle_steers_des_time, _DT_MPC + _DT) + _DT  # no greater than dt mpc + dt, to prevent too high extraps
      #self.angle_steers_des = self.angle_steers_des_prev + (dt / _DT_MPC) * (self.angle_steers_des_mpc - self.angle_steers_des_prev)
      self.angle_steers_des = (smoothing * self.angle_steers_des + self.angle_steers_des_mpc) / (smoothing + 1.0)
      steers_max = get_steer_max(VM.CP, v_ego)
      self.pid.pos_limit = steers_max
      self.pid.neg_limit = -steers_max

      if VM.CP.steerControlType == car.CarParams.SteerControlType.torque:
        steer_feedforward = self.angle_steers_des - self.steer_zero_crossing
        steer_feedforward *= v_ego**2  # proportional to realigning tire momentum (~ lateral accel)
      else:
        steer_feedforward = self.angle_steers_des   # feedforward desired angle
  
      deadzone = 0.0

      prev_i_f = self.pid.i + self.pid.f

      output_steer =  self.pid.update(self.angle_steers_des, self.smooth_angle_steers, check_saturation=False, override=steer_override,
                                     feedforward=steer_feedforward, speed=v_ego, deadzone=deadzone)

      if not steer_override and v_ego > 10. and abs(self.smooth_angle_steers) <= 10:
        #take torque samples for external characterization

        angle_index = int(self.smooth_angle_steers) + 10
        self.rough_angle_array[angle_index] = int(self.smooth_angle_steers)
        if int(self.smooth_angle_steers) == int(self.angle_steers_des):
          self.save_steering = True
          self.steer_torque_array[angle_index] = int((self.steer_torque_count[angle_index] * self.steer_torque_array[angle_index] + (int((output_steer / v_ego**2) * 100000000.))) / (self.steer_torque_count[angle_index] + 1))    
          self.steer_speed_array[angle_index] = int((self.steer_torque_count[angle_index] * self.steer_speed_array[angle_index] + 10 * v_ego) / (self.steer_torque_count[angle_index] + 1))
          self.steer_torque_count[angle_index] = min(1000, self.steer_torque_count[angle_index] + 1)

        if self.prev_output_steer != 0 and int(self.smooth_angle_steers * 10) == int(self.angle_steers_des * 10) and self.prev_output_steer * output_steer <= 0:
          self.center_angle = (self.center_count * self.center_angle + self.smooth_angle_steers) / (self.center_count + 1)
          self.center_count = min(1000, self.center_count + 1)
  
        if (int(cur_time * 100) % 1) == 0:
          self.steerdata += ("%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%d|" % (self.isActive, self.steer_zero_crossing, self.center_angle, self.smooth_angle_steers, self.angle_steers_des, angle_offset, \
          self.angle_steers_des_mpc, cur_Steer_Ratio, VM.CP.steerKf / ratioFactor, VM.CP.steerKpV[0] / ratioFactor, VM.CP.steerKiV[0] / ratioFactor, VM.CP.steerRateCost, PL.PP.l_prob, \
          PL.PP.r_prob, PL.PP.c_prob, PL.PP.p_prob, self.l_poly[0], self.l_poly[1], self.l_poly[2], self.l_poly[3], self.r_poly[0], self.r_poly[1], self.r_poly[2], self.r_poly[3], \
          self.p_poly[0], self.p_poly[1], self.p_poly[2], self.p_poly[3], PL.PP.c_poly[0], PL.PP.c_poly[1], PL.PP.c_poly[2], PL.PP.c_poly[3], PL.PP.d_poly[0], PL.PP.d_poly[1], \
          PL.PP.d_poly[2], PL.PP.lane_width, PL.PP.lane_width_estimate, PL.PP.lane_width_certainty, v_ego, self.pid.p, self.pid.i, self.pid.f, int(time.time() * 100) * 10000000))

        if (prev_i_f * (self.pid.i + self.pid.f)) < 0.0:
          self.steer_zero_crossing = self.smooth_angle_steers

      #else:
        #self.steer_zero_crossing = 0.0
    
    self.sat_flag = self.pid.saturated
    self.prev_output_steer = output_steer
    return output_steer, float(self.angle_steers_des)
