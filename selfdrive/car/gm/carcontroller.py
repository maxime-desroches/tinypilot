from cereal import car
from common.conversions import Conversions as CV
from common.numpy_fast import interp, clip
from common.realtime import DT_CTRL
from opendbc.can.packer import CANPacker
from selfdrive.car import apply_std_steer_torque_limits, create_gas_interceptor_command2
from selfdrive.car.gm import gmcan
from selfdrive.car.gm.values import DBC, NO_ASCM, CanBus, CarControllerParams
import math

VisualAlert = car.CarControl.HUDControl.VisualAlert
GearShifter = car.CarState.GearShifter


def actuator_hystereses(final_pedal, pedal_steady):
  # hyst params... TODO: move these to VehicleParams
  pedal_hyst_gap = 0.01    # don't change pedal command for small oscillations within this value

  # for small pedal oscillations within pedal_hyst_gap, don't change the pedal command
  if math.isclose(final_pedal,0.0):
    pedal_steady = 0.
  elif final_pedal > pedal_steady + pedal_hyst_gap:
    pedal_steady = final_pedal - pedal_hyst_gap
  elif final_pedal < pedal_steady - pedal_hyst_gap:
    pedal_steady = final_pedal + pedal_hyst_gap
  final_pedal = pedal_steady

  return final_pedal, pedal_steady

NetworkLocation = car.CarParams.NetworkLocation
TransmissionType = car.CarParams.TransmissionType


class CarController:
  def __init__(self, dbc_name, CP, VM):
    self.pedal_steady = 0.
    self.CP = CP
    self.start_time = 0.
    self.apply_steer_last = 0
    self.apply_gas = 0
    self.apply_brake = 0
    self.frame = 0

    self.lka_steering_cmd_counter_last = -1
    self.lka_icon_status_last = (False, False)
    self.steer_rate_limited = False

    self.params = CarControllerParams()

    self.packer_pt = CANPacker(DBC[self.CP.carFingerprint]['pt'])
    self.packer_obj = CANPacker(DBC[self.CP.carFingerprint]['radar'])
    self.packer_ch = CANPacker(DBC[self.CP.carFingerprint]['chassis'])
    self.packer_body = CANPacker(DBC[self.CP.carFingerprint]['body'])

  def update(self, CC, CS):
    actuators = CC.actuators
    hud_control = CC.hudControl
    hud_alert = hud_control.visualAlert
    hud_v_cruise = hud_control.setSpeed
    if hud_v_cruise > 70:
      hud_v_cruise = 0

    # Send CAN commands.
    can_sends = []

    # Steering (50Hz)
    # Avoid GM EPS faults when transmitting messages too close together: skip this transmit if we just received the
    # next Panda loopback confirmation in the current CS frame.
    if CS.lka_steering_cmd_counter != self.lka_steering_cmd_counter_last:
      self.lka_steering_cmd_counter_last = CS.lka_steering_cmd_counter
    elif (self.frame % self.params.STEER_STEP) == 0:
      lkas_enabled = CC.latActive and CS.out.vEgo > self.params.MIN_STEER_SPEED
      if lkas_enabled:
        new_steer = int(round(actuators.steer * self.params.STEER_MAX))
        apply_steer = apply_std_steer_torque_limits(new_steer, self.apply_steer_last, CS.out.steeringTorque, self.params)
        self.steer_rate_limited = new_steer != apply_steer
      else:
        apply_steer = 0

      self.apply_steer_last = apply_steer
      # GM EPS faults on any gap in received message counters. To handle transient OP/Panda safety sync issues at the
      # moment of disengaging, increment the counter based on the last message known to pass Panda safety checks.
      idx = (CS.lka_steering_cmd_counter + 1) % 4
      
      can_sends.append(gmcan.create_steering_control(self.packer_pt, CanBus.POWERTRAIN, apply_steer, idx, lkas_enabled))

    # TODO: All three conditions should not be required - really only last two?
    if self.CP.carFingerprint not in NO_ASCM and self.CP.openpilotLongitudinalControl and not self.CP.pcmCruise:
      # Gas/regen and brakes - all at 25Hz
      if (self.frame % 4) == 0:
        if not CC.longActive:
          # Stock ECU sends max regen when not enabled.
          self.apply_gas = self.params.MAX_ACC_REGEN
          self.apply_brake = 0
        else:
          self.apply_gas = int(round(interp(actuators.accel, self.params.GAS_LOOKUP_BP, self.params.GAS_LOOKUP_V)))
          self.apply_brake = int(round(interp(actuators.accel, self.params.BRAKE_LOOKUP_BP, self.params.BRAKE_LOOKUP_V)))

        idx = (self.frame // 4) % 4
        at_full_stop = CC.longActive and CS.out.standstill
        near_stop = CC.longActive and (CS.out.vEgo < self.params.NEAR_STOP_BRAKE_PHASE)
        # GasRegenCmdActive needs to be 1 to avoid cruise faults. It describes the ACC state, not actuation
        can_sends.append(gmcan.create_gas_regen_command(self.packer_pt, CanBus.POWERTRAIN, self.apply_gas, idx, CC.enabled, at_full_stop))
        can_sends.append(gmcan.create_friction_brake_command(self.packer_ch, CanBus.CHASSIS, self.apply_brake, idx, near_stop, at_full_stop))

      # Send dashboard UI commands (ACC status), 25hz
      if (self.frame % 4) == 0:
        send_fcw = hud_alert == VisualAlert.fcw
        can_sends.append(gmcan.create_acc_dashboard_command(self.packer_pt, CanBus.POWERTRAIN, CC.enabled,
                                                            hud_v_cruise * CV.MS_TO_KPH, hud_control.leadVisible, send_fcw))

      # Radar needs to know current speed and yaw rate (50hz),
      # and that ADAS is alive (10hz)
      time_and_headlights_step = 10
      tt = self.frame * DT_CTRL

      if self.frame % time_and_headlights_step == 0:
        idx = (self.frame // time_and_headlights_step) % 4
        can_sends.append(gmcan.create_adas_time_status(CanBus.OBSTACLE, int((tt - self.start_time) * 60), idx))
        can_sends.append(gmcan.create_adas_headlights_status(self.packer_obj, CanBus.OBSTACLE))

      speed_and_accelerometer_step = 2
      if self.frame % speed_and_accelerometer_step == 0:
        idx = (self.frame // speed_and_accelerometer_step) % 4
        can_sends.append(gmcan.create_adas_steering_status(CanBus.OBSTACLE, idx))
        can_sends.append(gmcan.create_adas_accelerometer_speed_status(CanBus.OBSTACLE, CS.out.vEgo, idx))

      if self.frame % self.params.ADAS_KEEPALIVE_STEP == 0:
        can_sends += gmcan.create_adas_keepalive(CanBus.POWERTRAIN)
    elif CS.CP.openpilotLongitudinalControl:
      # Gas/regen and brakes - all at 25Hz
      if (self.frame % 4) == 0:
        if not CC.longActive:
          # Stock ECU sends max regen when not enabled.
          self.apply_gas = self.params.MAX_ACC_REGEN
          self.apply_brake = 0
        else:
          self.apply_gas = int(round(interp(actuators.accel, self.params.GAS_LOOKUP_BP, self.params.GAS_LOOKUP_V)))
          self.apply_brake = int(round(interp(actuators.accel, self.params.BRAKE_LOOKUP_BP, self.params.BRAKE_LOOKUP_V)))

        idx = (self.frame // 4) % 4

        at_full_stop = CC.longActive and CS.out.standstill
        # near_stop = enabled and (CS.out.vEgo < P.NEAR_STOP_BRAKE_PHASE)
        # VOACC based cars have brakes on PT bus - OP won't be doing VOACC for a while
        # can_sends.append(gmcan.create_friction_brake_command(self.packer_pt, CanBus.POWERTRAIN, self.apply_brake, idx, near_stop, at_full_stop))
        
        if CS.CP.enableGasInterceptor:
          # #TODO: Add alert when not in L mode re: limited braking
          singlePedalMode = CS.out.gearShifter == GearShifter.low and self.CP.transmissionType == TransmissionType.automatic
          # TODO: JJS Detect saturated battery?
          if singlePedalMode:
            # In L Mode, Pedal applies regen at a fixed coast-point (TODO: max regen in L mode may be different per car)
            # This will apply to EVs in L mode.
            # accel values below zero down to a cutoff point 
            #  that approximates the percentage of braking regen can handle should be scaled between 0 and the coast-point
            # accell values below this point will need to be add-on future hijacked AEB
            # TODO: Determine (or guess) at regen precentage

            # From Felger's Bolt Fort
            #It seems in L mode, accel / decel point is around 1/5
            #-1-------AEB------0----regen---0.15-------accel----------+1
            # Shrink gas request to 0.85, have it start at 0.2
            # Shrink brake request to 0.85, first 0.15 gives regen, rest gives AEB
            
            zero = 0.15625 # 40/256
            
            if (actuators.accel > 0.):
              # Scales the accel from 0-1 to 0.156-1
              pedal_gas = clip(((1-zero) * actuators.accel + zero), 0., 1.)
            else:
              # if accel is negative, -0.1 -> 0.015625
              pedal_gas = clip(zero + actuators.accel, 0., zero) # Make brake the same size as gas, but clip to regen
              # aeb = actuators.brake*(1-zero)-regen # For use later, braking more than regen
          else:
            pedal_gas = clip(actuators.accel, 0., 1.)
            
          
          # apply pedal hysteresis and clip the final output to valid values.
          pedal_final, self.pedal_steady = actuator_hystereses(pedal_gas, self.pedal_steady)
          pedal_gas = clip(pedal_final, 0., 1.)
          
          if not CC.longActive:
            pedal_gas = 0.0 # May not be needed with the enable param
               
          can_sends.append(create_gas_interceptor_command2(self.packer_pt, CC.longActive, pedal_gas, idx))
        else:
          can_sends.append(gmcan.create_gas_regen_command(self.packer_pt, CanBus.POWERTRAIN, self.apply_gas, idx, CC.enabled, at_full_stop))

              
    # Show green icon when LKA torque is applied, and
    # alarming orange icon when approaching torque limit.
    # If not sent again, LKA icon disappears in about 5 seconds.
    # Conveniently, sending camera message periodically also works as a keepalive.
    lka_active = CS.lkas_status == 1
    lka_critical = lka_active and abs(actuators.steer) > 0.9
    lka_icon_status = (lka_active, lka_critical)
    if self.CP.networkLocation != NetworkLocation.fwdCamera and (self.frame % self.params.CAMERA_KEEPALIVE_STEP == 0 or lka_icon_status != self.lka_icon_status_last):
      steer_alert = hud_alert in (VisualAlert.steerRequired, VisualAlert.ldw)
      can_sends.append(gmcan.create_lka_icon_command(CanBus.SW_GMLAN, lka_active, lka_critical, steer_alert))
      self.lka_icon_status_last = lka_icon_status

    new_actuators = actuators.copy()
    new_actuators.steer = self.apply_steer_last / self.params.STEER_MAX
    new_actuators.gas = self.apply_gas
    new_actuators.brake = self.apply_brake

    self.frame += 1
    return new_actuators, can_sends
  