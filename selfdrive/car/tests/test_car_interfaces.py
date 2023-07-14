#!/usr/bin/env python3
import math
import unittest
import hypothesis.strategies as st
from hypothesis import given, settings
import importlib
from parameterized import parameterized

from cereal import car
from selfdrive.car import gen_empty_fingerprint
from selfdrive.car.car_helpers import interfaces
from selfdrive.car.fingerprints import all_known_cars


def get_fuzzy_car_interface_args(draw) -> dict:
  # Fuzzy CAN fingerprints and FW versions to test more states of the CarInterface
  fingerprint_strategy = st.fixed_dictionaries({key: st.dictionaries(st.integers(min_value=0, max_value=0x800),
                                                                     st.integers(min_value=0, max_value=64)) for key in
                                                gen_empty_fingerprint()})

  # just the most important fields
  car_fw_strategy = st.lists(st.fixed_dictionaries({
    'ecu': st.sampled_from(list(car.CarParams.Ecu.schema.enumerants.keys())),
    # TODO: only use reasonable addrs for the paired ecu and brand/platform
    'address': st.integers(min_value=0, max_value=0x800),
  }))

  params_strategy = st.fixed_dictionaries({
    'fingerprints': fingerprint_strategy,
    'car_fw': car_fw_strategy,
    'experimental_long': st.booleans(),
  })

  params = draw(params_strategy)
  params['car_fw'] = [car.CarParams.CarFw(**fw) for fw in params['car_fw']]
  return params


class TestCarInterfaces(unittest.TestCase):

  @parameterized.expand([(car,) for car in sorted(all_known_cars())])
  @settings(max_examples=5)
  @given(data=st.data())
  def test_car_interfaces(self, car_name, data):
    CarInterface, CarController, CarState = interfaces[car_name]

    args = get_fuzzy_car_interface_args(data.draw)

    car_params = CarInterface.get_params(car_name, args['fingerprints'], args['car_fw'],
                                         experimental_long=args['experimental_long'], docs=False)
    car_interface = CarInterface(car_params, CarController, CarState)
    assert car_params
    assert car_interface

    self.assertGreater(car_params.mass, 1)
    self.assertGreater(car_params.wheelbase, 0)
    # centerToFront is center of gravity to front wheels, assert a reasonable range
    self.assertTrue(car_params.wheelbase * 0.3 < car_params.centerToFront < car_params.wheelbase * 0.7)
    self.assertGreater(car_params.maxLateralAccel, 0)

    # Longitudinal sanity checks
    self.assertEqual(len(car_params.longitudinalTuning.kpV), len(car_params.longitudinalTuning.kpBP))
    self.assertEqual(len(car_params.longitudinalTuning.kiV), len(car_params.longitudinalTuning.kiBP))
    self.assertEqual(len(car_params.longitudinalTuning.deadzoneV), len(car_params.longitudinalTuning.deadzoneBP))

    # Lateral sanity checks
    if car_params.steerControlType != car.CarParams.SteerControlType.angle:
      tune = car_params.lateralTuning
      if tune.which() == 'pid':
        self.assertTrue(not math.isnan(tune.pid.kf) and tune.pid.kf > 0)
        self.assertTrue(len(tune.pid.kpV) > 0 and len(tune.pid.kpV) == len(tune.pid.kpBP))
        self.assertTrue(len(tune.pid.kiV) > 0 and len(tune.pid.kiV) == len(tune.pid.kiBP))

      elif tune.which() == 'torque':
        self.assertTrue(not math.isnan(tune.torque.kf) and tune.torque.kf > 0)
        self.assertTrue(not math.isnan(tune.torque.friction) and tune.torque.friction > 0)

      elif tune.which() == 'indi':
        self.assertTrue(len(tune.indi.outerLoopGainV))

    cc_msg=FuzzyGenerator.get_random_msg(data.draw, car.CarControl, real_floats=True)
    # Run car interface
    CC = car.CarControl.new_message(**cc_msg)
    for _ in range(10):
      car_interface.update(CC, [])
      car_interface.apply(CC, 0)
      car_interface.apply(CC, 0)

    CC = car.CarControl.new_message(**cc_msg)
    CC.enabled = True
    for _ in range(10):
      car_interface.update(CC, [])
      car_interface.apply(CC, 0)
      car_interface.apply(CC, 0)

    # Test radar interface
    RadarInterface = importlib.import_module(f'selfdrive.car.{car_params.carName}.radar_interface').RadarInterface
    radar_interface = RadarInterface(car_params)
    assert radar_interface

    # Run radar interface once
    radar_interface.update([])
    if not car_params.radarUnavailable and radar_interface.rcp is not None and \
       hasattr(radar_interface, '_update') and hasattr(radar_interface, 'trigger_msg'):
      radar_interface._update([radar_interface.trigger_msg])

if __name__ == "__main__":
  unittest.main()
