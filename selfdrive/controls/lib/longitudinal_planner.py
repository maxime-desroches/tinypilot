#!/usr/bin/env python3
import math
import numpy as np
from common.params import Params
from common.numpy_fast import interp

import cereal.messaging as messaging
from common.realtime import sec_since_boot
from selfdrive.swaglog import cloudlog
from selfdrive.config import Conversions as CV
from selfdrive.controls.lib.speed_smoother import speed_smoother
from selfdrive.controls.lib.longcontrol import LongCtrlState
from selfdrive.controls.lib.events import Events
from selfdrive.controls.lib.fcw import FCWChecker
from selfdrive.controls.lib.long_mpc import LongitudinalMpc
from selfdrive.controls.lib.drive_helpers import V_CRUISE_MAX
from selfdrive.controls.lib.turn_controller import TurnController
from selfdrive.controls.lib.speed_limit_controller import SpeedLimitController, SpeedLimitResolver


LON_MPC_STEP = 0.2  # first step is 0.2s
AWARENESS_DECEL = -0.2     # car smoothly decel at .2m/s^2 when user is distracted

# lookup tables VS speed to determine min and max accels in cruise
# make sure these accelerations are smaller than mpc limits
_A_CRUISE_MIN_V = [-1.0, -.8, -.67, -.5, -.30]
_A_CRUISE_MIN_BP = [  0.,  5.,  10., 20.,  40.]

# need fast accel at very low speed for stop and go
# make sure these accelerations are smaller than mpc limits
_A_CRUISE_MAX_V = [1.2, 1.2, 0.65, .4]
_A_CRUISE_MAX_V_FOLLOWING = [1.6, 1.6, 0.65, .4]
_A_CRUISE_MAX_BP = [0.,  6.4, 22.5, 40.]

# Lookup table for turns
_A_TOTAL_MAX_V = [1.7, 3.2]
_A_TOTAL_MAX_BP = [20., 40.]

# dp
DP_FOLLOWING_DIST = {
  0: 1.2,
  1: 1.5,
  2: 1.8,
}

DP_ACCEL_ECO = 0
DP_ACCEL_NORMAL = 1
DP_ACCEL_SPORT = 2

# accel profile by @arne182
_DP_CRUISE_MIN_V = [-2.0, -1.5, -1.0, -0.7, -0.5]
_DP_CRUISE_MIN_V_ECO = [-1.0, -0.7, -0.6, -0.5, -0.3]
_DP_CRUISE_MIN_V_SPORT = [-3.0, -2.6, -2.3, -2.0, -1.0]
_DP_CRUISE_MIN_V_FOLLOWING = [-4.0, -4.0, -3.5, -2.5, -2.0]
_DP_CRUISE_MIN_BP = [0.0, 5.0, 10.0, 20.0, 55.0]

_DP_CRUISE_MAX_V = [2.0, 2.0, 1.5, .5, .3]
_DP_CRUISE_MAX_V_ECO = [0.8, 0.9, 1.0, 0.4, 0.2]
_DP_CRUISE_MAX_V_SPORT = [3.0, 3.5, 3.0, 2.0, 2.0]
_DP_CRUISE_MAX_V_FOLLOWING = [1.6, 1.4, 1.4, .7, .3]
_DP_CRUISE_MAX_BP = [0., 5., 10., 20., 55.]

# Lookup table for turns
_DP_TOTAL_MAX_V = [3.3, 3.0, 3.9]
_DP_TOTAL_MAX_BP = [0., 25., 55.]

def dp_calc_cruise_accel_limits(v_ego, following, dp_profile):
  if following:
    a_cruise_min = interp(v_ego, _DP_CRUISE_MIN_BP, _DP_CRUISE_MIN_V_FOLLOWING)
    a_cruise_max = interp(v_ego, _DP_CRUISE_MAX_BP, _DP_CRUISE_MAX_V_FOLLOWING)
  else:
    if dp_profile == DP_ACCEL_ECO:
      a_cruise_min = interp(v_ego, _DP_CRUISE_MIN_BP, _DP_CRUISE_MIN_V_ECO)
      a_cruise_max = interp(v_ego, _DP_CRUISE_MAX_BP, _DP_CRUISE_MAX_V_ECO)
    elif dp_profile == DP_ACCEL_SPORT:
      a_cruise_min = interp(v_ego, _DP_CRUISE_MIN_BP, _DP_CRUISE_MIN_V_SPORT)
      a_cruise_max = interp(v_ego, _DP_CRUISE_MAX_BP, _DP_CRUISE_MAX_V_SPORT)
    else:
      a_cruise_min = interp(v_ego, _DP_CRUISE_MIN_BP, _DP_CRUISE_MIN_V)
      a_cruise_max = interp(v_ego, _DP_CRUISE_MAX_BP, _DP_CRUISE_MAX_V)
  return np.vstack([a_cruise_min, a_cruise_max])

def calc_cruise_accel_limits(v_ego, following):
  a_cruise_min = interp(v_ego, _A_CRUISE_MIN_BP, _A_CRUISE_MIN_V)

  if following:
    a_cruise_max = interp(v_ego, _A_CRUISE_MAX_BP, _A_CRUISE_MAX_V_FOLLOWING)
  else:
    a_cruise_max = interp(v_ego, _A_CRUISE_MAX_BP, _A_CRUISE_MAX_V)
  return np.vstack([a_cruise_min, a_cruise_max])


def limit_accel_in_turns(v_ego, angle_steers, a_target, CP):
  """
  This function returns a limited long acceleration allowed, depending on the existing lateral acceleration
  this should avoid accelerating when losing the target in turns
  """

  a_total_max = interp(v_ego, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V)
  a_y = v_ego**2 * angle_steers * CV.DEG_TO_RAD / (CP.steerRatio * CP.wheelbase)
  a_x_allowed = math.sqrt(max(a_total_max**2 - a_y**2, 0.))

  return [a_target[0], min(a_target[1], a_x_allowed)]


class Planner():
  def __init__(self, CP):
    self.CP = CP

    self.mpc1 = LongitudinalMpc(1)
    self.mpc2 = LongitudinalMpc(2)
    self.turn_controller = TurnController(CP)
    self.speed_limit_controller = SpeedLimitController()

    self.v_acc_start = 0.0
    self.a_acc_start = 0.0
    self.v_acc_next = 0.0
    self.a_acc_next = 0.0

    self.v_acc = 0.0
    self.v_acc_future = 0.0
    self.a_acc = 0.0
    self.v_cruise = 0.0
    self.a_cruise = 0.0

    self.longitudinalPlanSource = 'cruise'
    self.fcw_checker = FCWChecker()
    self.path_x = np.arange(192)

    self.fcw = False

    self.params = Params()
    self.first_loop = True

    # dp
    self.dp_accel_profile_ctrl = False
    self.dp_accel_profile = DP_ACCEL_ECO
    self.dp_following_profile_ctrl = False
    self.dp_following_profile = 3
    self.dp_following_dist = 1.8 # default val
    self.events = Events()

  def choose_solution(self, v_cruise_setpoint, enabled, lead_1, lead_2, steeringAngleDeg):
    center_x = -2.5 # Wheel base 2.5m
    lead1_check = True
    lead2_check = True
    if steeringAngleDeg > 100: # only at high angles
      center_y = -1+2.5/math.tan(steeringAngleDeg/1800.*math.pi) # Car Width 2m. Left side considered in left hand turn
      lead1_check = math.sqrt((lead_1.dRel-center_x)**2+(lead_1.yRel-center_y)**2) < abs(2.5/math.sin(steeringAngleDeg/1800.*math.pi))+1. # extra meter clearance to car
      lead2_check = math.sqrt((lead_2.dRel-center_x)**2+(lead_2.yRel-center_y)**2) < abs(2.5/math.sin(steeringAngleDeg/1800.*math.pi))+1.
    elif steeringAngleDeg < -100: # only at high angles
      center_y = +1+2.5/math.tan(steeringAngleDeg/1800.*math.pi) # Car Width 2m. Right side considered in right hand turn
      lead1_check = math.sqrt((lead_1.dRel-center_x)**2+(lead_1.yRel-center_y)**2) < abs(2.5/math.sin(steeringAngleDeg/1800.*math.pi))+1.
      lead2_check = math.sqrt((lead_2.dRel-center_x)**2+(lead_2.yRel-center_y)**2) < abs(2.5/math.sin(steeringAngleDeg/1800.*math.pi))+1.
    if enabled:
      solutions = {'cruise': self.v_cruise}
      if self.mpc1.prev_lead_status:
        solutions['mpc1'] = self.mpc1.v_mpc
      if self.mpc2.prev_lead_status:
        solutions['mpc2'] = self.mpc2.v_mpc
      if self.turn_controller.is_active:
        solutions['turn'] = self.turn_controller.v_turn
      if self.speed_limit_controller.is_active:
        solutions['limit'] = self.speed_limit_controller.v_limit

      slowest = min(solutions, key=solutions.get)

      self.longitudinalPlanSource = slowest
      # Choose lowest of MPC and cruise
      if slowest == 'mpc1':
        self.v_acc = self.mpc1.v_mpc
        self.a_acc = self.mpc1.a_mpc
      elif slowest == 'mpc2':
        self.v_acc = self.mpc2.v_mpc
        self.a_acc = self.mpc2.a_mpc
      elif slowest == 'cruise':
        self.v_acc = self.v_cruise
        self.a_acc = self.a_cruise
      elif slowest == 'turn':
        self.v_acc = self.turn_controller.v_turn
        self.a_acc = self.turn_controller.a_turn
      elif slowest == 'limit':
        self.v_acc = self.speed_limit_controller.v_limit
        self.a_acc = self.speed_limit_controller.a_limit

    self.v_acc_future = min([self.mpc1.v_mpc_future, self.mpc2.v_mpc_future, v_cruise_setpoint])
    if self.turn_controller.is_active:
      self.v_acc_future = min(self.v_acc_future, self.turn_controller.v_turn_future)
    if self.speed_limit_controller.is_active:
      self.v_acc_future = min(self.v_acc_future, self.speed_limit_controller.v_limit_future)

  def update(self, sm, CP):
    """Gets called when new radarState is available"""
    cur_time = sec_since_boot()
    v_ego = sm['carState'].vEgo

    long_control_state = sm['controlsState'].longControlState
    v_cruise_kph = sm['controlsState'].vCruise
    force_slow_decel = sm['controlsState'].forceDecel

    v_cruise_kph = min(v_cruise_kph, V_CRUISE_MAX)
    v_cruise_setpoint = v_cruise_kph * CV.KPH_TO_MS

    lead_1 = sm['radarState'].leadOne
    lead_2 = sm['radarState'].leadTwo
    self.events = Events()

    enabled = (long_control_state == LongCtrlState.pid) or (long_control_state == LongCtrlState.stopping)
    following = lead_1.status and lead_1.dRel < 45.0 and lead_1.vLeadK > v_ego and lead_1.aLeadK > 0.0

    self.v_acc_start = self.v_acc_next
    self.a_acc_start = self.a_acc_next

    # dp
    self.dp_accel_profile_ctrl = sm['dragonConf'].dpAccelProfileCtrl
    self.dp_accel_profile = sm['dragonConf'].dpAccelProfile
    self.dp_following_profile_ctrl = sm['dragonConf'].dpAccelProfileCtrl
    self.dp_following_profile = sm['dragonConf'].dpFollowingProfile
    self.dp_following_dist = DP_FOLLOWING_DIST[0 if not self.dp_following_profile_ctrl else self.dp_following_profile]

    # Calculate speed for normal cruise control
    pedal_pressed = sm['carState'].gasPressed or sm['carState'].brakePressed
    if enabled and not self.first_loop and not pedal_pressed:
      if not self.dp_accel_profile_ctrl:
        accel_limits = [float(x) for x in calc_cruise_accel_limits(v_ego, following)]
      else:
        accel_limits = [float(x) for x in dp_calc_cruise_accel_limits(v_ego, following, self.dp_accel_profile)]
      jerk_limits = [min(-0.1, accel_limits[0]), max(0.1, accel_limits[1])]  # TODO: make a separate lookup for jerk tuning
      accel_limits_turns = limit_accel_in_turns(v_ego, sm['carState'].steeringAngleDeg, accel_limits, self.CP)

      if force_slow_decel:
        # if required so, force a smooth deceleration
        accel_limits_turns[1] = min(accel_limits_turns[1], AWARENESS_DECEL)
        accel_limits_turns[0] = min(accel_limits_turns[0], accel_limits_turns[1])

      self.v_cruise, self.a_cruise = speed_smoother(self.v_acc_start, self.a_acc_start,
                                                    v_cruise_setpoint,
                                                    accel_limits_turns[1], accel_limits_turns[0],
                                                    jerk_limits[1], jerk_limits[0],
                                                    LON_MPC_STEP)

      # cruise speed can't be negative even is user is distracted
      self.v_cruise = max(self.v_cruise, 0.)
      # update speed limit solution calculation.
      self.speed_limit_controller.update(enabled, self.v_acc_start, self.a_acc_start, sm,
                                         v_cruise_setpoint, accel_limits_turns, jerk_limits, self.events)
    else:
      starting = long_control_state == LongCtrlState.starting
      a_ego = min(sm['carState'].aEgo, 0.0)
      reset_speed = self.CP.minSpeedCan if starting else v_ego
      reset_accel = self.CP.startAccel if starting else a_ego
      self.v_acc = reset_speed
      self.a_acc = reset_accel
      self.v_acc_start = reset_speed
      self.a_acc_start = reset_accel
      self.v_cruise = reset_speed
      self.a_cruise = reset_accel
      # dp reset
      self.v_model = reset_speed
      self.a_model = reset_accel
      self.speed_limit_controller.deactivate()  # Deactivate speed limit controller to provide no solution.

    self.mpc1.set_cur_state(self.v_acc_start, self.a_acc_start)
    self.mpc2.set_cur_state(self.v_acc_start, self.a_acc_start)

    self.mpc1.update(sm['carState'], lead_1, self.dp_following_dist)
    self.mpc2.update(sm['carState'], lead_2, self.dp_following_dist)
    self.turn_controller.update(enabled, self.v_acc_start, self.a_acc_start, v_cruise_setpoint, sm)

    self.choose_solution(v_cruise_setpoint, enabled, lead_1, lead_2, sm['carState'].steeringAngleDeg)

    # determine fcw
    if self.mpc1.new_lead:
      self.fcw_checker.reset_lead(cur_time)

    blinkers = sm['carState'].leftBlinker or sm['carState'].rightBlinker
    self.fcw = self.fcw_checker.update(self.mpc1.mpc_solution, cur_time,
                                       sm['controlsState'].active,
                                       v_ego, sm['carState'].aEgo,
                                       lead_1.dRel, lead_1.vLead, lead_1.aLeadK,
                                       lead_1.yRel, lead_1.vLat,
                                       lead_1.fcw, blinkers) and not sm['carState'].brakePressed
    if self.fcw:
      cloudlog.info("FCW triggered %s", self.fcw_checker.counters)

    # Interpolate 0.05 seconds and save as starting point for next iteration
    a_acc_sol = self.a_acc_start + (CP.radarTimeStep / LON_MPC_STEP) * (self.a_acc - self.a_acc_start)
    v_acc_sol = self.v_acc_start + CP.radarTimeStep * (a_acc_sol + self.a_acc_start) / 2.0
    self.v_acc_next = v_acc_sol
    self.a_acc_next = a_acc_sol

    self.first_loop = False

  def publish(self, sm, pm):
    self.mpc1.publish(pm)
    self.mpc2.publish(pm)

    plan_send = messaging.new_message('longitudinalPlan')

    plan_send.valid = sm.all_alive_and_valid(service_list=['carState', 'controlsState', 'radarState'])

    longitudinalPlan = plan_send.longitudinalPlan
    longitudinalPlan.mdMonoTime = sm.logMonoTime['modelV2']
    longitudinalPlan.radarStateMonoTime = sm.logMonoTime['radarState']
    longitudinalPlan.vCruise = float(self.v_cruise)
    longitudinalPlan.aCruise = float(self.a_cruise)
    longitudinalPlan.vStart = float(self.v_acc_start)
    longitudinalPlan.aStart = float(self.a_acc_start)
    longitudinalPlan.vTarget = float(self.v_acc)
    longitudinalPlan.aTarget = float(self.a_acc)
    longitudinalPlan.vTargetFuture = float(self.v_acc_future)
    longitudinalPlan.hasLead = self.mpc1.prev_lead_status
    longitudinalPlan.longitudinalPlanSource = self.longitudinalPlanSource
    longitudinalPlan.fcw = self.fcw

    longitudinalPlan.turnControllerState = self.turn_controller.state
    longitudinalPlan.turnAcc = float(self.turn_controller.a_turn)
    longitudinalPlan.speedLimitControlState = self.speed_limit_controller.state
    longitudinalPlan.speedLimit = float(self.speed_limit_controller.speed_limit)
    longitudinalPlan.speedLimitOffset = float(self.speed_limit_controller.speed_limit_offset)
    longitudinalPlan.distToSpeedLimit = float(self.speed_limit_controller.distance)
    longitudinalPlan.isMapSpeedLimit = bool(self.speed_limit_controller.source == SpeedLimitResolver.Source.map_data)
    longitudinalPlan.eventsDEPRECATED = self.events.to_msg()

    longitudinalPlan.processingDelay = (plan_send.logMonoTime / 1e9) - sm.rcv_time['radarState']

    pm.send('longitudinalPlan', plan_send)
