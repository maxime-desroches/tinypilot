from math import atan2

from cereal import car
import cereal.messaging as messaging
from openpilot.selfdrive.controls.lib.events import Events
from openpilot.common.numpy_fast import interp
from openpilot.common.realtime import DT_DMON
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.stat_live import RunningStatFilter
from openpilot.common.transformations.camera import DEVICE_CAMERAS

EventName = car.CarEvent.EventName

# ******************************************************************************************
#  NOTE: To fork maintainers.
#  Disabling or nerfing safety features will get you and your users banned from our servers.
#  We recommend that you do not change these numbers from the defaults.
# ******************************************************************************************

class DRIVER_MONITOR_SETTINGS:
  def __init__(self):
    self._DT_DMON = DT_DMON
    # ref (page15-16): https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:42018X1947&rid=2
    self._AWARENESS_TIME = 30. # passive wheeltouch total timeout
    self._AWARENESS_PRE_TIME_TILL_TERMINAL = 15.
    self._AWARENESS_PROMPT_TIME_TILL_TERMINAL = 6.
    self._DISTRACTED_TIME = 11. # active monitoring total timeout
    self._DISTRACTED_PRE_TIME_TILL_TERMINAL = 8.
    self._DISTRACTED_PROMPT_TIME_TILL_TERMINAL = 6.

_FACE_THRESHOLD = 0.5
_PARTIAL_FACE_THRESHOLD = 0.75 if TICI else 0.5
_EYE_THRESHOLD = 0.5
_SG_THRESHOLD = 0.5
_BLINK_THRESHOLD = 0.5
_BLINK_THRESHOLD_SLACK = 0.65
_BLINK_THRESHOLD_STRICT = 0.5
_PITCH_WEIGHT = 1.35 # 1.5  # pitch matters a lot more
_METRIC_THRESHOLD = 0.4
_METRIC_THRESHOLD_SLACK = 0.55
_METRIC_THRESHOLD_STRICT = 0.4
_PITCH_POS_ALLOWANCE = 0.12  # rad, to not be too sensitive on positive pitch
_PITCH_NATURAL_OFFSET = 0.02  # people don't seem to look straight when they drive relaxed, rather a bit up
_YAW_NATURAL_OFFSET = 0.08  # people don't seem to look straight when they drive relaxed, rather a bit to the right (center of car)

_DISTRACTED_FILTER_TS = 0.25  # 0.6Hz

    self._HI_STD_FALLBACK_TIME = int(10  / self._DT_DMON)  # fall back to wheel touch if model is uncertain for 10s
    self._DISTRACTED_FILTER_TS = 0.25  # 0.6Hz
    self._ALWAYS_ON_ALERT_MIN_SPEED = 7

    self._POSE_CALIB_MIN_SPEED = 13  # 30 mph
    self._POSE_OFFSET_MIN_COUNT = int(60 / self._DT_DMON)  # valid data counts before calibration completes, 1min cumulative
    self._POSE_OFFSET_MAX_COUNT = int(360 / self._DT_DMON)  # stop deweighting new data after 6 min, aka "short term memory"

    self._WHEELPOS_CALIB_MIN_SPEED = 11
    self._WHEELPOS_THRESHOLD = 0.5
    self._WHEELPOS_FILTER_MIN_COUNT = int(15 / self._DT_DMON) # allow 15 seconds to converge wheel side

    self._RECOVERY_FACTOR_MAX = 5.  # relative to minus step change
    self._RECOVERY_FACTOR_MIN = 1.25  # relative to minus step change

    self._MAX_TERMINAL_ALERTS = 3  # not allowed to engage after 3 terminal alerts
    self._MAX_TERMINAL_DURATION = int(30 / self._DT_DMON)  # not allowed to engage after 30s of terminal alerts

class DistractedType:
  NOT_DISTRACTED = 0
  DISTRACTED_POSE = 1 << 0
  DISTRACTED_BLINK = 1 << 1
  DISTRACTED_E2E = 1 << 2

class DriverPose:
  def __init__(self, max_trackable):
    self.yaw = 0.
    self.pitch = 0.
    self.roll = 0.
    self.yaw_std = 0.
    self.pitch_std = 0.
    self.roll_std = 0.
    self.pitch_offseter = RunningStatFilter(max_trackable=max_trackable)
    self.yaw_offseter = RunningStatFilter(max_trackable=max_trackable)
    self.calibrated = False
    self.low_std = True
    self.cfactor_pitch = 1.
    self.cfactor_yaw = 1.

class DriverBlink:
  def __init__(self):
    self.left = 0.
    self.right = 0.


# model output refers to center of undistorted+leveled image
EFL = 598.0 # focal length in K
cam = DEVICE_CAMERAS[("tici", "ar0231")] # corrected image has same size as raw
W, H = (cam.dcam.width, cam.dcam.height)  # corrected image has same size as raw

def face_orientation_from_net(angles_desc, pos_desc, rpy_calib):
  # the output of these angles are in device frame
  # so from driver's perspective, pitch is up and yaw is right

  pitch_net, yaw_net, roll_net = angles_desc

  face_pixel_position = ((pos_desc[0]+0.5)*W, (pos_desc[1]+0.5)*H)
  yaw_focal_angle = atan2(face_pixel_position[0] - W//2, EFL)
  pitch_focal_angle = atan2(face_pixel_position[1] - H//2, EFL)

  pitch = pitch_net + pitch_focal_angle
  yaw = -yaw_net + yaw_focal_angle

  # no calib for roll
  pitch -= rpy_calib[1]
  yaw -= rpy_calib[2]
  return roll_net, pitch, yaw

class DriverPose:
  def __init__(self, max_trackable):
    self.yaw = 0.
    self.pitch = 0.
    self.roll = 0.
    self.pitch_offseter = RunningStatFilter(max_trackable=_POSE_OFFSET_MAX_COUNT)
    self.yaw_offseter = RunningStatFilter(max_trackable=_POSE_OFFSET_MAX_COUNT)
    self.cfactor = 1.

class DriverMonitoring:
  def __init__(self, rhd_saved=False, settings=None, always_on=False):
    if settings is None:
      settings = DRIVER_MONITOR_SETTINGS()
    # init policy settings
    self.settings = settings

    # init driver status
    self.wheelpos_learner = RunningStatFilter()
    self.pose = DriverPose(self.settings._POSE_OFFSET_MAX_COUNT)
    self.blink = DriverBlink()
    self.eev1 = 0.
    self.eev2 = 1.
    self.ee1_offseter = RunningStatFilter(max_trackable=self.settings._POSE_OFFSET_MAX_COUNT)
    self.ee2_offseter = RunningStatFilter(max_trackable=self.settings._POSE_OFFSET_MAX_COUNT)
    self.ee1_calibrated = False
    self.ee2_calibrated = False

    self.always_on = always_on
    self.distracted_types = []
    self.driver_distracted = False
    self.driver_distraction_filter = FirstOrderFilter(0., self.settings._DISTRACTED_FILTER_TS, self.settings._DT_DMON)
    self.wheel_on_right = False
    self.wheel_on_right_last = None
    self.wheel_on_right_default = rhd_saved
    self.face_detected = False
    self.terminal_alert_cnt = 0
    self.terminal_time = 0
    self.step_change = 0.
    self.active_monitoring_mode = True
    self.threshold_prompt = _DISTRACTED_PROMPT_TIME_TILL_TERMINAL / _DISTRACTED_TIME

    self._reset_awareness()
    self._set_timers(active_monitoring=True)
    self._reset_events()

  def _reset_awareness(self):
    self.awareness = 1.
    self.awareness_active = 1.
    self.awareness_passive = 1.

  def _reset_events(self):
    self.current_events = Events()

  def _set_timers(self, active_monitoring):
    if self.active_monitoring_mode and self.awareness <= self.threshold_prompt:
      if active_monitoring:
        self.step_change = self.settings._DT_DMON / self.settings._DISTRACTED_TIME
      else:
        self.step_change = 0.
      return  # no exploit after orange alert
    elif self.awareness <= 0.:
      return

    if active_monitoring:
      # when falling back from passive mode to active mode, reset awareness to avoid false alert
      if not self.active_monitoring_mode:
        self.awareness_passive = self.awareness
        self.awareness = self.awareness_active

      self.threshold_pre = self.settings._DISTRACTED_PRE_TIME_TILL_TERMINAL / self.settings._DISTRACTED_TIME
      self.threshold_prompt = self.settings._DISTRACTED_PROMPT_TIME_TILL_TERMINAL / self.settings._DISTRACTED_TIME
      self.step_change = self.settings._DT_DMON / self.settings._DISTRACTED_TIME
      self.active_monitoring_mode = True
    else:
      if self.active_monitoring_mode:
        self.awareness_active = self.awareness
        self.awareness = self.awareness_passive

      self.threshold_pre = self.settings._AWARENESS_PRE_TIME_TILL_TERMINAL / self.settings._AWARENESS_TIME
      self.threshold_prompt = self.settings._AWARENESS_PROMPT_TIME_TILL_TERMINAL / self.settings._AWARENESS_TIME
      self.step_change = self.settings._DT_DMON / self.settings._AWARENESS_TIME
      self.active_monitoring_mode = False

  def _set_policy(self, model_data, car_speed):
    bp = model_data.meta.disengagePredictions.brakeDisengageProbs[0] # brake disengage prob in next 2s
    k1 = max(-0.00156*((car_speed-16)**2)+0.6, 0.2)
    bp_normal = max(min(bp / k1, 0.5),0)
    self.pose.cfactor_pitch = interp(bp_normal, [0, 0.5],
                                           [self.settings._POSE_PITCH_THRESHOLD_SLACK,
                                            self.settings._POSE_PITCH_THRESHOLD_STRICT]) / self.settings._POSE_PITCH_THRESHOLD
    self.pose.cfactor_yaw = interp(bp_normal, [0, 0.5],
                                           [self.settings._POSE_YAW_THRESHOLD_SLACK,
                                            self.settings._POSE_YAW_THRESHOLD_STRICT]) / self.settings._POSE_YAW_THRESHOLD

  def get_pose(self, driver_state, cal_rpy, car_speed, op_engaged):
    # 10 Hz
    if len(driver_monitoring.faceOrientation) == 0 or len(driver_monitoring.facePosition) == 0:
      return

    self.pose.roll, self.pose.pitch, self.pose.yaw = face_orientation_from_net(driver_monitoring.faceOrientation, driver_monitoring.facePosition, cal_rpy)
    self.blink.left_blink = driver_monitoring.leftBlinkProb * (driver_monitoring.leftEyeProb>_EYE_THRESHOLD)
    self.blink.right_blink = driver_monitoring.rightBlinkProb * (driver_monitoring.rightEyeProb>_EYE_THRESHOLD)
    self.face_detected = driver_monitoring.faceProb > _FACE_THRESHOLD and \
                          abs(driver_monitoring.facePosition[0]) <= 0.4 and abs(driver_monitoring.facePosition[1]) <= 0.45 and \
                          not self.is_rhd_region

    self.driver_distracted = self._is_driver_distracted(self.pose, self.blink)>0
    # first order filters
    self.driver_distraction_filter.update(self.driver_distracted)

    # update offseter
    # only update when driver is actively driving the car above a certain speed
    if self.face_detected and car_speed>_POSE_CALIB_MIN_SPEED and (not op_engaged or not self.driver_distracted):
      self.pose.pitch_offseter.push_and_update(self.pose.pitch)
      self.pose.yaw_offseter.push_and_update(self.pose.yaw)
      self.ee1_offseter.push_and_update(self.eev1)
      self.ee2_offseter.push_and_update(self.eev2)

    self.pose.calibrated = self.pose.pitch_offseter.filtered_stat.n > self.settings._POSE_OFFSET_MIN_COUNT and \
                                       self.pose.yaw_offseter.filtered_stat.n > self.settings._POSE_OFFSET_MIN_COUNT
    self.ee1_calibrated = self.ee1_offseter.filtered_stat.n > self.settings._POSE_OFFSET_MIN_COUNT
    self.ee2_calibrated = self.ee2_offseter.filtered_stat.n > self.settings._POSE_OFFSET_MIN_COUNT

    self._set_timers(self.face_detected)

  def _update_events(self, driver_engaged, op_engaged, standstill, wrong_gear, car_speed):
    self._reset_events()
    # Block engaging after max number of distrations or when alert active
    if self.terminal_alert_cnt >= self.settings._MAX_TERMINAL_ALERTS or \
       self.terminal_time >= self.settings._MAX_TERMINAL_DURATION or \
       self.always_on and self.awareness <= self.threshold_prompt:
      self.current_events.add(EventName.tooDistracted)

    always_on_valid = self.always_on and not wrong_gear
    if (driver_engaged and self.awareness > 0 and not self.active_monitoring_mode) or \
       (not always_on_valid and not op_engaged) or \
       (always_on_valid and not op_engaged and self.awareness <= 0):
      # always reset on disengage with normal mode; disengage resets only on red if always on
      self._reset_awareness()
      return

    driver_attentive = self.driver_distraction_filter.x < 0.37
    awareness_prev = self.awareness

    if (driver_attentive and self.face_detected and self.awareness > 0):
      # only restore awareness when paying attention and alert is not red
      self.awareness = min(self.awareness + ((self.settings._RECOVERY_FACTOR_MAX-self.settings._RECOVERY_FACTOR_MIN)*
                                             (1.-self.awareness)+self.settings._RECOVERY_FACTOR_MIN)*self.step_change, 1.)
      if self.awareness == 1.:
        self.awareness_passive = min(self.awareness_passive + self.step_change, 1.)
      # don't display alert banner when awareness is recovering and has cleared orange
      if self.awareness > self.threshold_prompt:
        return

    _reaching_audible = self.awareness - self.step_change <= self.threshold_prompt
    _reaching_terminal = self.awareness - self.step_change <= 0
    standstill_exemption = standstill and _reaching_audible
    always_on_red_exemption = always_on_valid and not op_engaged and _reaching_terminal
    always_on_lowspeed_exemption = always_on_valid and not op_engaged and car_speed < self.settings._ALWAYS_ON_ALERT_MIN_SPEED and _reaching_audible

    certainly_distracted = self.driver_distraction_filter.x > 0.63 and self.driver_distracted and self.face_detected
    maybe_distracted = self.hi_stds > self.settings._HI_STD_FALLBACK_TIME or not self.face_detected

    if certainly_distracted or maybe_distracted:
      # should always be counting if distracted unless at standstill (lowspeed for always-on) and reaching orange
      # also will not be reaching 0 if DM is active when not engaged
      if not (standstill_exemption or always_on_red_exemption or always_on_lowspeed_exemption):
        self.awareness = max(self.awareness - self.step_change, -0.1)

    alert = None
    if self.awareness <= 0.:
      # terminal red alert: disengagement required
      alert = EventName.driverDistracted if self.active_monitoring_mode else EventName.driverUnresponsive
      self.terminal_time += 1
      if awareness_prev > 0.:
        self.terminal_alert_cnt += 1
    elif self.awareness <= self.threshold_prompt:
      # prompt orange alert
      alert = EventName.promptDriverDistracted if self.active_monitoring_mode else EventName.promptDriverUnresponsive
    elif self.awareness <= self.threshold_pre:
      # pre green alert
      alert = EventName.preDriverDistracted if self.active_monitoring_mode else EventName.preDriverUnresponsive

    if alert is not None:
      self.current_events.add(alert)


  def get_state_packet(self, valid=True):
    # build driverMonitoringState packet
    dat = messaging.new_message('driverMonitoringState', valid=valid)
    dat.driverMonitoringState = {
      "events": self.current_events.to_msg(),
      "faceDetected": self.face_detected,
      "isDistracted": self.driver_distracted,
      "distractedType": sum(self.distracted_types),
      "awarenessStatus": self.awareness,
      "posePitchOffset": self.pose.pitch_offseter.filtered_stat.mean(),
      "posePitchValidCount": self.pose.pitch_offseter.filtered_stat.n,
      "poseYawOffset": self.pose.yaw_offseter.filtered_stat.mean(),
      "poseYawValidCount": self.pose.yaw_offseter.filtered_stat.n,
      "stepChange": self.step_change,
      "awarenessActive": self.awareness_active,
      "awarenessPassive": self.awareness_passive,
      "isLowStd": self.pose.low_std,
      "hiStdCount": self.hi_stds,
      "isActiveMode": self.active_monitoring_mode,
      "isRHD": self.wheel_on_right,
    }
    return dat

  def run_step(self, sm):
    # Set strictness
    self._set_policy(
      model_data=sm['modelV2'],
      car_speed=sm['carState'].vEgo
    )

    # Parse data from dmonitoringmodeld
    self._update_states(
      driver_state=sm['driverStateV2'],
      cal_rpy=sm['liveCalibration'].rpyCalib,
      car_speed=sm['carState'].vEgo,
      op_engaged=sm['controlsState'].enabled
    )

    # Update distraction events
    self._update_events(
      driver_engaged=sm['carState'].steeringPressed or sm['carState'].gasPressed,
      op_engaged=sm['controlsState'].enabled,
      standstill=sm['carState'].standstill,
      wrong_gear=sm['carState'].gearShifter in [car.CarState.GearShifter.reverse, car.CarState.GearShifter.park],
      car_speed=sm['carState'].vEgo
    )
