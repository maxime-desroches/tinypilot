#!/usr/bin/env python3
import importlib
import unittest

from cereal import car
from selfdrive.car.car_helpers import interfaces
from selfdrive.car.fingerprints import _FINGERPRINTS as FINGERPRINTS
from selfdrive.car.fingerprints import all_known_cars


class TestCarInterfaces(unittest.TestCase):
  def test_car_interfaces(self):
    all_cars = all_known_cars()

    for car_name in all_cars:
      fingerprint = FINGERPRINTS[car_name][0]

      CarInterface, CarController, CarState = interfaces[car_name]
      fingerprints = {
        0: fingerprint,
        1: fingerprint,
        2: fingerprint,
      }

      car_fw = []

      for has_relay in [True, False]:
        car_params = CarInterface.get_params(car_name, fingerprints, has_relay, car_fw)
        car_interface = CarInterface(car_params, CarController, CarState)
        assert car_params
        assert car_interface

        # Run car interface
        CC = car.CarControl.new_message()
        for _ in range(10):
          car_interface.update(CC, [])
          car_interface.apply(CC)
          car_interface.apply(CC)

        CC = car.CarControl.new_message()
        CC.enabled = True
        for _ in range(10):
          car_interface.update(CC, [])
          car_interface.apply(CC)
          car_interface.apply(CC)

      # Test radar interface
      RadarInterface = importlib.import_module('selfdrive.car.%s.radar_interface' % car_params.carName).RadarInterface
      radar_interface = RadarInterface(car_params)
      assert radar_interface

      # Run radar interface once
      radar_interface.update([])
      if hasattr(radar_interface, '_update') and hasattr(radar_interface, 'trigger_msg'):
        radar_interface._update([radar_interface.trigger_msg])

if __name__ == "__main__":
  unittest.main()
