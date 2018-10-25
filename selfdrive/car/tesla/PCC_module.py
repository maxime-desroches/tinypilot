from selfdrive.car.tesla import teslacan
#from selfdrive.controls.lib.pid import PIController
from selfdrive.car.tesla.longcontrol_tesla import LongControl, LongCtrlState, STARTING_TARGET_SPEED
#from selfdrive.car.tesla.interface import tesla_compute_gb
from selfdrive.car.tesla import teslacan
from common.numpy_fast import clip, interp
from selfdrive.services import service_list
from selfdrive.car.tesla.values import AH,CruiseState, CruiseButtons, CAR
from selfdrive.boardd.boardd import can_list_to_can_capnp
from selfdrive.config import Conversions as CV
from selfdrive.controls.lib.speed_smoother import speed_smoother
from common.realtime import sec_since_boot
from common.numpy_fast import clip
import selfdrive.messaging as messaging
import os
import subprocess
import time
import zmq
import math
import numpy as np


MPC_BRAKE_MULTIPLIER = 6.
DEBUG = False

PCC_SPEED_FACTOR = 5.
PCC_X_SAFE = 2.

# TODO: these should end up in values.py at some point, probably variable by trim
# Accel limits
ACCEL_HYST_GAP = 0.5  # don't change accel command for small oscilalitons within this value

PEDAL_MAX_UP = 4.
PEDAL_MAX_DOWN = 150.
#BB
MIN_SAFE_DIST_M = 4. # min safe distance in meters
FRAMES_PER_SEC = 20.

SPEED_UP = 3. / FRAMES_PER_SEC   # 2 m/s = 7.5 mph = 12 kph 
SPEED_DOWN = 3. / FRAMES_PER_SEC

MAX_PEDAL_VALUE = 112.

#BBTODO: move the vehicle variables; maybe make them speed variable
TORQUE_LEVEL_ACC = 0.
TORQUE_LEVEL_DECEL = -30.
FOLLOW_UP_TIME = 1.5 #time in seconds to follow car in front
MIN_PCC_V = 0. #
MAX_PCC_V = 170.

MIN_CAN_SPEED = 0.3  #TODO: parametrize this in car interface


AWARENESS_DECEL = -0.2     # car smoothly decel at .2m/s^2 when user is distracted

# lookup tables VS speed to determine min and max accels in cruise
# make sure these accelerations are smaller than mpc limits
_A_CRUISE_MIN_V  = [-6.0, -5.0, -4.2, -3.0, -1.90]
_A_CRUISE_MIN_BP = [   0., 5.,  10., 20.,  40.]

# need fast accel at very low speed for stop and go
# make sure these accelerations are smaller than mpc limits
_A_CRUISE_MAX_V = [1.0, 1.0, .8, .6, .4]
_A_CRUISE_MAX_V_FOLLOWING = [1.0, 1.0, .8, .6, .4]
_A_CRUISE_MAX_BP = [0.,  5., 10., 20., 40.]

# Lookup table for turns
_A_TOTAL_MAX_V = [1.5, 1.5, 1.5]
_A_TOTAL_MAX_BP = [0., 20., 40.]

_FCW_A_ACT_V = [-3., -2.]
_FCW_A_ACT_BP = [0., 30.]

# max acceleration allowed in acc, which happens in restart
A_ACC_MAX = max(_A_CRUISE_MAX_V_FOLLOWING)

_DT = 0.05    # 20Hz in our case, since we don't want to process more than once the same live20 message
_DT_MPC = 0.05  # 20Hz 

def tesla_compute_gb(accel, speed):
  return float(accel)  / 3.

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

  a_target[1] = min(a_target[1], a_x_allowed)
  return a_target


class PCCState(object):
  # Possible state of the ACC system, following the DI_cruiseState naming
  # scheme.
  OFF = 0         # Disabled by UI.
  STANDBY = 1     # Ready to be enaged.
  ENABLED = 2     # Engaged.
  NOT_READY = 9   # Not ready to be engaged due to the state of the car.



def _current_time_millis():
  return int(round(time.time() * 1000))

def accel_hysteresis(accel, accel_steady, enabled):

  # for small accel oscillations within ACCEL_HYST_GAP, don't change the accel command
  if not enabled:
    # send 0 when disabled, otherwise acc faults
    accel_steady = 0.
  elif accel > accel_steady + ACCEL_HYST_GAP:
    accel_steady = accel - ACCEL_HYST_GAP
  elif accel < accel_steady - ACCEL_HYST_GAP:
    accel_steady = accel + ACCEL_HYST_GAP
  accel = accel_steady
  return accel, accel_steady


#this is for the pedal cruise control
class PCCController(object):
  def __init__(self,carcontroller):
    self.CC = carcontroller
    self.human_cruise_action_time = 0
    self.automated_cruise_action_time = 0
    self.last_angle = 0.
    context = zmq.Context()
    self.poller = zmq.Poller()
    self.live20 = messaging.sub_sock(context, service_list['live20'].port, conflate=True, poller=self.poller)
    self.lead_1 = None
    self.last_update_time = 0
    self.enable_pedal_cruise = False
    self.last_cruise_stalk_pull_time = 0
    self.prev_pcm_acc_status = 0
    self.prev_cruise_buttons = CruiseButtons.IDLE
    self.prev_cruise_setting = CruiseButtons.IDLE
    self.pedal_hardware_present = False
    self.pedal_hardware_first_check = True
    self.pedal_speed_kph = 0.
    self.pedal_idx = 0
    self.accel_steady = 0.
    self.prev_tesla_accel = 0.
    self.prev_tesla_pedal = 0.
    self.user_pedal_state = 0
    self.torqueLevel_last = 0.
    self.prev_v_ego = 0.
    self.PedalForZeroTorque = 18. #starting number, works on my S85
    self.lastTorqueForPedalForZeroTorque = TORQUE_LEVEL_DECEL
    self.follow_time = FOLLOW_UP_TIME # in seconds
    self.v_pid = 0.
    self.a_pid = 0.
    self.b_pid = 0.
    self.last_output_gb = 0.
    self.last_speed = 0.
    self.last_brake = 0.
    #for smoothing the changes in speed
    self.v_acc_start = 0.0
    self.a_acc_start = 0.0
    self.acc_start_time = sec_since_boot()
    self.v_acc = 0.0
    self.v_acc_sol = 0.0
    self.v_acc_future = 0.0
    self.a_acc = 0.0
    self.a_acc_sol = 0.0
    self.v_cruise = 0.0
    self.a_cruise = 0.0
    #Long Control
    self.LoC = None
    #when was radar data last updated?
    self.last_md_ts = None
    self.last_l100_ts = None
    self.md_ts = None
    self.l100_ts = None


  def reset(self, v_pid):
    if self.LoC:
      self.LoC.reset(v_pid)

  def update_stat(self,CS, enabled, sendcan):
    if self.LoC == None:
      self.LoC = LongControl(CS.CP, tesla_compute_gb)


    can_sends = []
    #BBTODO: a better way to engage the pedal early and reset its CAN
    # on first brake press check if hardware present; being on CAN2 values are not included in fingerprinting
    self.pedal_hardware_present = CS.pedal_hardware_present
    
    if not self.pedal_hardware_present:
      if (CS.cstm_btns.get_button_status("pedal")>0):
        #no pedal hardware, disable button
        CS.cstm_btns.set_button_status("pedal",0)
        print "disabling pedal"
      print "no pedal hardware"
      return
    if self.pedal_hardware_present:
      if (CS.cstm_btns.get_button_status("pedal")==0):
        #pedal hardware, enable button
        CS.cstm_btns.set_button_status("pedal",1)
        print "enabling pedal"
    # check if we had error before
    if self.user_pedal_state != CS.user_pedal_state:
      self.user_pedal_state = CS.user_pedal_state
      CS.cstm_btns.set_button_status("pedal", 1 if self.user_pedal_state > 0 else 0)
      if self.user_pedal_state > 0:
        CS.UE.custom_alert_message(3,"Pedal Interceptor Error (" + `self.user_pedal_state` + ")",150,4)
        # send reset command
        idx = self.pedal_idx
        self.pedal_idx = (self.pedal_idx + 1) % 16
        can_sends.append(teslacan.create_pedal_command_msg(0,0,idx))
        sendcan.send(can_list_to_can_capnp(can_sends, msgtype='sendcan').to_bytes())
    # disable on brake
    if CS.brake_pressed and self.enable_pedal_cruise:
      self.enable_pedal_cruise = False
      self.reset(0.)
      CS.UE.custom_alert_message(3,"PDL Disabled",150,4)
      CS.cstm_btns.set_button_status("pedal",1)
      print "brake pressed"

    prev_enable_pedal_cruise = self.enable_pedal_cruise
    # process any stalk movement
    curr_time_ms = _current_time_millis()
    speed_uom_kph = 1.
    if CS.imperial_speed_units:
      speed_uom_kph = CV.MPH_TO_KPH
    if (CS.cruise_buttons == CruiseButtons.MAIN and
        self.prev_cruise_buttons != CruiseButtons.MAIN):
      double_pull = curr_time_ms - self.last_cruise_stalk_pull_time < 750
      self.last_cruise_stalk_pull_time = curr_time_ms
      ready = (CS.cstm_btns.get_button_status("pedal") > PCCState.OFF and
               enabled and
               CruiseState.is_off(CS.pcm_acc_status))
      if ready and double_pull:
        # A double pull enables ACC. updating the max ACC speed if necessary.
        self.enable_pedal_cruise = True
        self.LoC.reset(CS.v_ego)
        # Increase PCC speed to match current, if applicable.
        self.pedal_speed_kph = max(CS.v_ego * CV.MS_TO_KPH, self.pedal_speed_kph)
      else:
        # A single pull disables ACC (falling back to just steering).
        self.enable_pedal_cruise = False
    # Handle pressing the cancel button.
    elif CS.cruise_buttons == CruiseButtons.CANCEL:
      self.enable_pedal_cruise = False
      self.pedal_speed_kph = 0. 
      self.last_cruise_stalk_pull_time = 0
    # Handle pressing up and down buttons.
    elif (self.enable_pedal_cruise and 
          CS.cruise_buttons !=self.prev_cruise_buttons):
      # Real stalk command while PCC is already enabled. Adjust the max PCC
      # speed if necessary. 
      actual_speed_kph = CS.v_ego * CV.MS_TO_KPH
      if CS.cruise_buttons == CruiseButtons.RES_ACCEL:
        self.pedal_speed_kph = max(self.pedal_speed_kph, actual_speed_kph) + speed_uom_kph
      elif CS.cruise_buttons == CruiseButtons.RES_ACCEL_2ND:
        self.pedal_speed_kph = max(self.pedal_speed_kph, actual_speed_kph) + 5 * speed_uom_kph
      elif CS.cruise_buttons == CruiseButtons.DECEL_SET:
        self.pedal_speed_kph = min(self.pedal_speed_kph, actual_speed_kph) - speed_uom_kph
      elif CS.cruise_buttons == CruiseButtons.DECEL_2ND:
        self.pedal_speed_kph = min(self.pedal_speed_kph, actual_speed_kph) - 5 * speed_uom_kph
      # Clip ACC speed between 0 and 170 KPH.
      self.pedal_speed_kph = clip(self.pedal_speed_kph,MIN_PCC_V ,MAX_PCC_V)
    # If something disabled cruise control, disable PCC too
    elif (self.enable_pedal_cruise == True and
          CS.pcm_acc_status != 0):
      self.enable_pedal_cruise = False
    
    # Notify if PCC was toggled
    if prev_enable_pedal_cruise and not self.enable_pedal_cruise:
      CS.UE.custom_alert_message(3, "PCC Disabled", 150, 4)
      CS.cstm_btns.set_button_status("pedal", PCCState.STANDBY)
    elif self.enable_pedal_cruise and not prev_enable_pedal_cruise:
      CS.UE.custom_alert_message(2, "PCC Enabled", 150)
      CS.cstm_btns.set_button_status("pedal", PCCState.ENABLED)

    # Update the UI to show whether the current car state allows PCC.
    if CS.cstm_btns.get_button_status("pedal") in [PCCState.STANDBY, PCCState.NOT_READY]:
      if (enabled
          and CruiseState.is_off(CS.pcm_acc_status)):
        CS.cstm_btns.set_button_status("pedal", PCCState.STANDBY)
      else:
        CS.cstm_btns.set_button_status("pedal", PCCState.NOT_READY)
          
    # Update prev state after all other actions.
    self.prev_cruise_buttons = CS.cruise_buttons
    self.prev_pcm_acc_status = CS.pcm_acc_status
    

  def update_pdl(self,enabled,CS,frame,actuators,pcm_speed):
    cur_time = sec_since_boot()
    idx = self.pedal_idx
    self.pedal_idx = (self.pedal_idx + 1) % 16
    if not self.pedal_hardware_present or not enabled:
      return 0.,0,idx
    following = False
    # Alternative speed decision logic that uses the lead car's distance
    # and speed more directly.
    # Bring in the lead car distance from the Live20 feed
    l20 = None
    if enabled:
      for socket, _ in self.poller.poll(0):
        if socket is self.live20:
          l20 = messaging.recv_one(socket)
          break
    if l20 is not None:
      self.lead_1 = l20.live20.leadOne
      self.md_ts = l20.live20.mdMonoTime
      self.l100_ts = l20.live20.l100MonoTime

    prevent_overshoot = False #not CS.CP.stoppingControl and CS.v_ego < 1.5 and v_target_future < 0.7
    accel_max = interp(CS.v_ego, CS.CP.gasMaxBP, CS.CP.gasMaxV)
    brake_max = interp(CS.v_ego, CS.CP.brakeMaxBP, CS.CP.brakeMaxV)

    output_gb = 0
    ####################################################################
    # this mode (Follow) uses the Follow logic created by JJ for ACC
    #
    # once the speed is detected we have to use our own PID to determine
    # how much accel and break we have to do
    ####################################################################
    if (CS.cstm_btns.get_button_label2_index("pedal") == 1):
      self.v_pid, self.b_pid = self.calc_follow_speed(CS)
      # cruise speed can't be negative even is user is distracted
      self.v_pid = max(self.v_pid, 0.)
      #self.b_pid *= MPC_BRAKE_MULTIPLIER


      enabled = ((self.LoC.long_control_state == LongCtrlState.pid) or (self.LoC.long_control_state == LongCtrlState.stopping)) and self.enable_pedal_cruise

      if self.enable_pedal_cruise: # enabled:
        accel_limits = map(float, calc_cruise_accel_limits(CS.v_ego, following))
        # TODO: make a separate lookup for jerk tuning
        jerk_limits = [min(-0.1, accel_limits[0]), max(0.1, accel_limits[1])]
        accel_limits = limit_accel_in_turns(CS.v_ego, CS.angle_steers, accel_limits, CS.CP)
        self.v_cruise, self.a_cruise = speed_smoother(self.v_acc_start, self.a_acc_start,
                                                      self.v_pid,
                                                      accel_limits[1], accel_limits[0],
                                                      jerk_limits[1], jerk_limits[0],
                                                      _DT_MPC)
        
        # cruise speed can't be negative even is user is distracted
        self.v_cruise = max(self.v_cruise, 0.)
        self.v_acc = self.v_cruise
        self.a_acc = self.a_cruise

        self.v_acc_future = self.v_pid
        vTargetFuture = self.v_acc_future

        self.v_acc_start = self.v_acc_sol
        self.a_acc_start = self.a_acc_sol
        self.acc_start_time = cur_time

        # Interpolation of trajectory
        dt = min(cur_time - self.acc_start_time, _DT_MPC + _DT) + _DT  # no greater than dt mpc + dt, to prevent too high extraps
        self.a_acc_sol = self.a_acc_start + (dt / _DT_MPC) * (self.a_acc - self.a_acc_start)
        self.v_acc_sol = self.v_acc_start + dt * (self.a_acc_sol + self.a_acc_start) / 2.0


        # we will try to feed forward the pedal position.... we might want to feed the last output_gb....
        # it's all about testing now.
        aTarget = self.a_acc_sol
        vTarget = clip(self.v_acc_sol,0,self.v_pid)
        vTargetFuture = clip(vTargetFuture, 0, self.v_pid)

        t_go, t_brake = self.LoC.update(self.enable_pedal_cruise, CS.v_ego, CS.brake_pressed != 0, CS.standstill, False, 
                  self.v_pid , vTarget, vTargetFuture, aTarget, CS.CP, None)
        output_gb = t_go - t_brake
        print "Output GB Follow:",output_gb
      else:
        self.LoC.reset(CS.v_ego)
        print "PID reset"
        output_gb = 0.
        starting = self.LoC.long_control_state == LongCtrlState.starting
        a_ego = min(CS.a_ego, 0.0)
        reset_speed = MIN_CAN_SPEED if starting else CS.v_ego
        reset_accel = CS.CP.startAccel if starting else a_ego
        self.v_acc = reset_speed
        self.a_acc = reset_accel
        self.v_acc_start = reset_speed
        self.a_acc_start = reset_accel
        self.v_cruise = reset_speed
        self.a_cruise = reset_accel
        self.v_acc_sol = reset_speed
        self.a_acc_sol = reset_accel
        self.v_pid = reset_speed
        

      
    ##############################################################
    # this mode (Lng MPC) uses the longitudinal MPC built in OP
    #
    # we use the values from actuator.accel and actuator.brake
    ##############################################################
    elif (CS.cstm_btns.get_button_label2("pedal") == "Lng MPC"):
      self.b_pid = MPC_BRAKE_MULTIPLIER
      output_gb = actuators.gas -  actuators.brake


    ######################################################################################
    # Determine pedal "zero"
    #
    #save position for cruising (zero acc, zero brake, no torque) when we are above 10 MPH
    ######################################################################################
    if (CS.torqueLevel < TORQUE_LEVEL_ACC) and (CS.torqueLevel > TORQUE_LEVEL_DECEL) and (CS.v_ego >= 10.* CV.MPH_TO_MS) and (abs(CS.torqueLevel) < abs(self.lastTorqueForPedalForZeroTorque)):
      self.PedalForZeroTorque = self.prev_tesla_accel
      self.lastTorqueForPedalForZeroTorque = CS.torqueLevel
      #print "Detected new Pedal For Zero Torque at %s" % (self.PedalForZeroTorque)
      #print "Torque level at detection %s" % (CS.torqueLevel)
      #print "Speed level at detection %s" % (CS.v_ego * CV.MS_TO_MPH)

    self.last_output_gb = output_gb
    # accel and brake
    apply_accel = clip(output_gb, 0., accel_max)
    #by adding b_pid we can enforce not to apply brake in small situations
    apply_brake = -clip(output_gb * self.b_pid, -brake_max, 0.)

    #if the speed if over 5mpg, the "zero" is at PedalForZeroTorque otherwise it at zero
    pedal_zero = 0.
    if (CS.v_ego >= 5.* CV.MPH_TO_MS):
      pedal_zero = self.PedalForZeroTorque
    tesla_brake = clip((1. - apply_brake) * pedal_zero,0,pedal_zero)
    tesla_accel = clip(apply_accel * MAX_PEDAL_VALUE,0,MAX_PEDAL_VALUE - tesla_brake)
    tesla_pedal = tesla_brake + tesla_accel



    tesla_pedal, self.accel_steady = accel_hysteresis(tesla_pedal, self.accel_steady, enabled)
    
    tesla_pedal = clip(tesla_pedal, self.prev_tesla_pedal - PEDAL_MAX_DOWN, self.prev_tesla_pedal + PEDAL_MAX_UP)
    tesla_pedal = clip(tesla_pedal, 0., MAX_PEDAL_VALUE) if self.enable_pedal_cruise else 0.
    enable_pedal = 1. if self.enable_pedal_cruise else 0.
    
    self.torqueLevel_last = CS.torqueLevel
    self.prev_tesla_pedal = tesla_pedal * enable_pedal
    self.prev_tesla_accel = apply_accel * enable_pedal
    self.prev_v_ego = CS.v_ego

    self.last_md_ts = self.md_ts
    self.last_l100_ts = self.l100_ts

    return self.prev_tesla_pedal,enable_pedal,idx


  # function to calculate the cruise speed based on a safe follow distance
  def calc_follow_speed(self, CS):
    
    current_time_ms = _current_time_millis()
     # Make sure we were able to populate lead_1.
    if self.lead_1 is None:
      return None
    # dRel is in meters.
    lead_dist = self.lead_1.dRel
    # Grab the relative speed.
    rel_speed = self.lead_1.vRel * CV.MS_TO_KPH
    # v_ego is in m/s, so safe_distance is in meters.
    safe_dist_m = max(CS.v_ego * self.follow_time, MIN_SAFE_DIST_M)
    # Current speed in kph
    actual_speed = CS.v_ego * CV.MS_TO_KPH
    available_speed = self.pedal_speed_kph - actual_speed
    # speed and brake to issue
    new_speed = max(actual_speed,self.last_speed)
    new_brake = 1.
    # debug msg
    msg = None
    msg2 = " *** UNKNOWN *** "

    if ((self.last_md_ts == self.md_ts) or (self.last_l100_ts == self.l100_ts)):
      #no radar update, so just keep doing what we're doing
      new_speed = self.last_speed
      new_brake = self.last_brake
      msg2 = "No change - No Data"
      if DEBUG:
        if msg:
          print msg
        CS.UE.custom_alert_message(2,msg2+ " ["+ `int(new_speed*CV.KPH_TO_MPH)` + "]",6,0)
      else:
        print msg2
      return new_speed * CV.KPH_TO_MS , new_brake
    #print "dRel: ", self.lead_1.dRel," yRel: ", self.lead_1.yRel, " vRel: ", self.lead_1.vRel, " aRel: ", self.lead_1.aRel, " vLead: ", self.lead_1.vLead, " vLeadK: ", self.lead_1.vLeadK, " aLeadK: ",     self.lead_1.aLeadK

    ###   Logic to determine best cruise speed ###

    if self.enable_pedal_cruise:
      # If cruise is set to faster than the max speed, slow down
      if lead_dist == 0 and new_speed != self.pedal_speed_kph:
        msg =  "Set to max"
        msg2 = "LD = 0, V = max"
        new_speed = self.pedal_speed_kph
        new_brake = 2.
      # If we have a populated lead_distance
      # TODO: make angle dependent on speed
      elif (lead_dist == 0 or lead_dist >= safe_dist_m) and CS.angle_steers >= 5.0:
        new_speed = self.last_speed
        msg = "Safe distance & turning: steady speed"
        msg2 = "LD = 0 or safe, A > 5, V= cnst"
      elif (lead_dist > 0):
        ### We have lead ###
        if lead_dist >= PCC_X_SAFE * safe_dist_m:
          msg =  "More than 2x safe distance... use lead speed..."
          msg2 = "LD > 2x safe, V=max" 
          if new_speed <= (self.pedal_speed_kph + SPEED_UP):
            #new_speed = self.last_speed + SPEED_UP
            new_speed = new_speed + SPEED_UP
          else:
            new_speed = self.pedal_speed_kph
          #new_speed = new_speed + rel_speed
          new_brake = 1.
        # if between safe dist and 2x safe dist adjust speed
        elif lead_dist >= safe_dist_m and lead_dist < PCC_X_SAFE * safe_dist_m:
          nb_d = (PCC_X_SAFE * safe_dist_m - lead_dist ) / ((PCC_X_SAFE -1) * safe_dist_m)
          nb_v = - rel_speed / PCC_SPEED_FACTOR
          msg2= "LD > safe, LD < 2x safe"
          new_brake = clip (nb_d, 0.1, 1) * clip(nb_v, 0.1, 20)
          if rel_speed > SPEED_UP:
            #new_speed = self.last_speed + SPEED_UP
            new_speed = new_speed + SPEED_UP
          elif rel_speed >= 0:
            new_speed = new_speed + rel_speed 
          else:
            new_speed = new_speed + rel_speed * clip(nb_v, 1, 2)
        #Reduce speed if rel_speed < -15kph so you don't rush up to lead car
        elif rel_speed < -15  and lead_dist >= safe_dist_m:
          msg =  "Approaching fast (-15), still more than the safe distance, slow down 3x"
          msg2 = "LD > safe, RV < -15, V=V-3xRV"
          new_speed = new_speed + rel_speed * 3
          new_brake = 2. 
        #Reduce speed if rel_speed < -5kph so you don't rush up to lead car
        elif rel_speed < -5  and lead_dist >=  safe_dist_m:
          msg =  "Approaching moderate (-5), still more than the safe distance, slow down 2x"
          msg2 = "LD > safe, RV < -5, V=V-2xRV"
          new_speed = new_speed + rel_speed * 2
          new_brake =  .5
        #Reduce speed if rel_speed < 0kph so you don't rush up to lead car
        elif rel_speed < 0  and lead_dist >=  safe_dist_m:
          msg =  "Approaching (-0), still more than the safe distance, slow down 1.5x"
          msg2 = "LD > safe, RV < 0, V=V-1.5xRV"
          new_speed = new_speed + rel_speed * 1.5
          new_brake = .3 
        #Reduce speed if rel_speed < -5kph so you don't rush up to lead car
        elif rel_speed >= 0  and lead_dist >=  safe_dist_m:
          msg =  "Following, still more than the safe distance... continue to follow"
          msg2 = "LD > safe, RV > 0, V=V+RV"
          new_speed = new_speed + rel_speed *.25
          new_brake = .1
        #Reduce speed significantly if lead_dist < safe dist
        elif lead_dist < safe_dist_m and rel_speed < 2:
          if rel_speed > 0:
            msg =  "Less than safe distance and moving away... 1.5 rel speed down"
            msg2 = "LD < safe, RV > 0, V=V-1.5xRV"
            new_speed = new_speed - rel_speed * 1.5
            new_brake = 6.
          else:
            msg =  "Less than safe distance and approaching... 4x rel speed down"
            msg2 = "LD < safe, RV < 0, V=V-4xRV"
            new_speed = new_speed + rel_speed * 4
            new_brake = 6.
        #Reduce speed significantly if lead_dist < safe dist
        elif lead_dist < safe_dist_m and rel_speed >= 2:
          msg =  "Less than safe distance and moving away... 1.5 rel speed down"
          msg2 = "LD < safe, RV > 2, V=V"
          #new_speed = new_speed - rel_speed * 1.5
          new_speed = self.last_speed
          new_brake = .1
        else:
          msg = "Have lead and do nothing"
          msg2 = msg
          new_brake = 1.
      else:
        msg = "No lead and do nothing"
        new_brake = 1.
        msg2 = msg
      #new_speed = clip(new_speed, 0, self.pedal_speed_kph)
      new_speed = clip(new_speed,MIN_PCC_V,MAX_PCC_V)
      new_speed = clip(new_speed, 0, self.pedal_speed_kph)
      self.last_speed = new_speed
      self.last_brake = new_brake
      if DEBUG:
        if msg:
          print msg
        CS.UE.custom_alert_message(2,msg2+ " ["+ `int(new_speed*CV.KPH_TO_MPH)` + "]",6,0)
      else:
        print msg2

    return new_speed * CV.KPH_TO_MS , new_brake
