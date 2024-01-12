#!/usr/bin/env python3
import json
import random
import unittest
import numpy as np

import cereal.messaging as messaging
from openpilot.common.params import Params
from openpilot.selfdrive.manager.process_config import managed_processes

"""
TODO:
- test language settings for all supported languages
"""

class TestNavd(unittest.TestCase):
  def setUp(self):
    self.params = Params()
    self.sm = messaging.SubMaster(['navRoute', 'navInstruction'])

  def tearDown(self):
    managed_processes['navd'].stop()

  def _check_route(self, start, end, check_coords=True):
    self.params.put("NavDestination", json.dumps(end))
    self.params.put("LastGPSPosition", json.dumps(start))

    managed_processes['navd'].start()
    for _ in range(30):
      self.sm.update(1000)
      if all(f > 0 for f in self.sm.rcv_frame.values()):
        break
    else:
      raise Exception("didn't get a route")

    assert managed_processes['navd'].proc.is_alive()
    managed_processes['navd'].stop()

    # ensure start and end match up
    coords = self.sm['navRoute'].coordinates
    if check_coords or len(coords) > 2:
      assert np.allclose([start['latitude'], start['longitude'], end['latitude'], end['longitude']],
                         [coords[0].latitude, coords[0].longitude, coords[-1].latitude, coords[-1].longitude],
                         rtol=1e-3)

  def test_simple(self):
    start = {
      "latitude": 32.7427228,
      "longitude": -117.2321177,
    }
    end = {
      "latitude": 32.7557004,
      "longitude": -117.268002,
    }
    self._check_route(start, end)

  def test_random(self):
    for i in range(10):
      start = {"latitude": random.uniform(-90, 90), "longitude": random.uniform(-180, 180)}
      end = {"latitude": random.uniform(-90, 90), "longitude": random.uniform(-180, 180)}
      self._check_route(start, end, check_coords=False)


if __name__ == "__main__":
  unittest.main()
