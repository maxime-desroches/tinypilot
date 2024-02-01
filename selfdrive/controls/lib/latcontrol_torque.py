import math

from cereal import log
from openpilot.common.numpy_fast import interp
from openpilot.selfdrive.car.interfaces import ControllerState, ControllerStateHistory
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.selfdrive.controls.lib.pid import PIDController
from openpilot.selfdrive.controls.lib.vehicle_model import ACCELERATION_DUE_TO_GRAVITY

# At higher speeds (25+mph) we can assume:
# Lateral acceleration achieved by a specific car correlates to
# torque applied to the steering rack. It does not correlate to
# wheel slip, or to speed.

# This controller applies torque to achieve desired lateral
# accelerations. To compensate for the low speed effects we
# use a LOW_SPEED_FACTOR in the error. Additionally, there is
# friction in the steering wheel that needs to be overcome to
# move it at all, this is compensated for too.

LOW_SPEED_X = [0, 10, 20, 30]
LOW_SPEED_Y = [15, 13, 10, 5]
HISTORY_LEN = 3
CONTROLSD_FPS = 100
HISTORY_FPS = 10


class LatControlTorque(LatControl):
  def __init__(self, CP, CI):
    super().__init__(CP, CI)
    self.torque_params = CP.lateralTuning.torque
    self.pid = PIDController(self.torque_params.kp, self.torque_params.ki,
                             k_f=self.torque_params.kf, pos_limit=self.steer_max, neg_limit=-self.steer_max)
    self.torque_from_lateral_accel = CI.torque_from_lateral_accel()
    self.use_steering_angle = self.torque_params.useSteeringAngle
    self.steering_angle_deadzone_deg = self.torque_params.steeringAngleDeadzoneDeg
    self.history = ControllerStateHistory(max_len=HISTORY_LEN)
    self.frame_counter = 0
    self.record_frame_id = CONTROLSD_FPS // HISTORY_FPS

  def update_live_torque_params(self, latAccelFactor, latAccelOffset, friction):
    self.torque_params.latAccelFactor = latAccelFactor
    self.torque_params.latAccelOffset = latAccelOffset
    self.torque_params.friction = friction

  def update(self, active, CS, VM, params, steer_limited, desired_curvature, llk):
    pid_log = log.ControlsState.LateralTorqueState.new_message()
    actual_curvature_vm = -VM.calc_curvature(math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), CS.vEgo, params.roll)
    roll_compensation = params.roll * ACCELERATION_DUE_TO_GRAVITY

    if not active:
      output_torque = 0.0
      pid_log.active = False
    else:
      if self.use_steering_angle:
        actual_curvature = actual_curvature_vm
        curvature_deadzone = abs(VM.calc_curvature(math.radians(self.steering_angle_deadzone_deg), CS.vEgo, 0.0))
      else:
        actual_curvature_llk = llk.angularVelocityCalibrated.value[2] / CS.vEgo
        actual_curvature = interp(CS.vEgo, [2.0, 5.0], [actual_curvature_vm, actual_curvature_llk])
        curvature_deadzone = 0.0
      desired_lateral_accel = desired_curvature * CS.vEgo ** 2

      # desired rate is the desired rate of change in the setpoint, not the absolute desired curvature
      # desired_lateral_jerk = desired_curvature_rate * CS.vEgo ** 2
      actual_lateral_accel = actual_curvature * CS.vEgo ** 2
      lateral_accel_deadzone = curvature_deadzone * CS.vEgo ** 2

      low_speed_factor = interp(CS.vEgo, LOW_SPEED_X, LOW_SPEED_Y)**2
      setpoint = desired_lateral_accel + low_speed_factor * desired_curvature
      measurement = actual_lateral_accel + low_speed_factor * actual_curvature
      gravity_adjusted_lateral_accel = desired_lateral_accel - roll_compensation
      torque_from_setpoint = self.torque_from_lateral_accel(ControllerState(setpoint, roll_compensation, CS.vEgo, CS.aEgo), self.history,
                                                            self.torque_params, setpoint, lateral_accel_deadzone, friction_compensation=False,
                                                            gravity_adjusted=False)
      torque_from_measurement = self.torque_from_lateral_accel(ControllerState(measurement, roll_compensation, CS.vEgo, CS.aEgo), self.history,
                                                               self.torque_params, measurement, lateral_accel_deadzone, friction_compensation=False,
                                                               gravity_adjusted=False)
      pid_log.error = torque_from_setpoint - torque_from_measurement
      ff = self.torque_from_lateral_accel(ControllerState(gravity_adjusted_lateral_accel, roll_compensation, CS.vEgo, CS.aEgo), self.history,
                                          self.torque_params, desired_lateral_accel - actual_lateral_accel, lateral_accel_deadzone,
                                          friction_compensation=True, gravity_adjusted=True)

      freeze_integrator = steer_limited or CS.steeringPressed or CS.vEgo < 5
      output_torque = self.pid.update(pid_log.error,
                                      feedforward=ff,
                                      speed=CS.vEgo,
                                      freeze_integrator=freeze_integrator)

      pid_log.active = True
      pid_log.p = self.pid.p
      pid_log.i = self.pid.i
      pid_log.d = self.pid.d
      pid_log.f = self.pid.f
      pid_log.output = -output_torque
      pid_log.actualLateralAccel = actual_lateral_accel
      pid_log.desiredLateralAccel = desired_lateral_accel
      pid_log.saturated = self._check_saturation(self.steer_max - abs(output_torque) < 1e-3, CS, steer_limited)

    if self.frame_counter % self.record_frame_id == 0:
      self.history.append(ControllerState(actual_curvature_vm * CS.vEgo ** 2, roll_compensation, CS.vEgo, CS.aEgo))
    self.frame_counter = (self.frame_counter + 1) % self.record_frame_id
    # TODO left is positive in this convention
    return -output_torque, 0.0, pid_log
