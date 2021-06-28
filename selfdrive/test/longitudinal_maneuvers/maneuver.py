import numpy as np
from selfdrive.test.longitudinal_maneuvers.plant import Plant


class Maneuver():
  def __init__(self, title, duration, **kwargs):
    # Was tempted to make a builder class
    self.distance_lead = kwargs.get("initial_distance_lead", 200.0)
    self.speed = kwargs.get("initial_speed", 0.0)
    self.lead_relevancy = kwargs.get("lead_relevancy", 0)

    self.speed_lead_values = kwargs.get("speed_lead_values", [0.0, 0.0])
    self.speed_lead_breakpoints = kwargs.get("speed_lead_breakpoints", [0.0, duration])

    self.duration = duration
    self.title = title

  def evaluate(self):
    plant = Plant(
      lead_relevancy=self.lead_relevancy,
      speed=self.speed,
      distance_lead=self.distance_lead
    )

    valid = True
    while plant.current_time() < self.duration:
      speeds_lead = np.interp(plant.current_time() + np.arange(0.,12.,2.), self.speed_lead_breakpoints, self.speed_lead_values)
      log = plant.step(speeds_lead)

      d_rel = log['distance_lead'] - log['distance'] if self.lead_relevancy else 200.
      v_rel = speeds_lead[0] - log['speed'] if self.lead_relevancy else 0.
      log['d_rel'] = d_rel
      log['v_rel'] = v_rel

      if d_rel < 1.0:
        print("Crashed!!!!")
        valid = False

    print("maneuver end", valid)
    return valid
