from abc import abstractmethod, ABC

from common.numpy_fast import clip
from common.realtime import DT_CTRL

MIN_STEER_SPEED = 0.3


class LatControl(ABC):
  # we define the steer torque scale as [-1.0...1.0]
  steer_max = 1.0
  sat_check_min_speed = 10.

  def __init__(self, CP, CI):
    self.sat_limit = CP.steerLimitTimer
    self.sat_count = 0.

  @abstractmethod
  def update(self, active, CS, VM, params, last_actuators, steer_limited, desired_curvature, desired_curvature_rate, llk):
    pass

  def reset(self):
    self.sat_count = 0.

  def _check_saturation(self, saturated, CS, steer_limited):
    if saturated and CS.vEgo > self.sat_check_min_speed and not steer_limited and not CS.steeringPressed:
      self.sat_count += DT_CTRL
    else:
      self.sat_count -= DT_CTRL
    self.sat_count = clip(self.sat_count, 0.0, self.sat_limit)
    return self.sat_count > (self.sat_limit - 1e-3)
