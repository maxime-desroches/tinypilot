#!/usr/bin/env python3
import argparse
import sys
from typing import List, Tuple
import unittest

from selfdrive.car.tests.routes import TestRoute
from selfdrive.car.tests.test_models import TestCarModel


def create_test_models_tests(routes: List[Tuple[str, TestRoute]], ci=False) -> unittest.TestSuite:
  """
    Creates a test suite for the TestCarModel class based on routes passed in
  """

  test_suite = unittest.TestSuite()
  for car_model, _test_route in routes:
    test_case_args = {"car_model": car_model, "test_route": _test_route, "ci": ci}

    # create new test case and discover tests
    CarModelTestCase = type("CarModelTestCase", (TestCarModel,), test_case_args)
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(CarModelTestCase))
  return test_suite


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Test any route against common issues with a new car port. " +
                                               "Uses selfdrive/car/tests/test_models.py")
  parser.add_argument("route", nargs="?", help="Specify route to run tests on")
  parser.add_argument("--car", help="Specify car model for test route")
  parser.add_argument("--segment", type=int, nargs="?", help="Specify segment of route to test")
  args = parser.parse_args()
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit()

  assert args.car is not None, "Specify car fingerprint with --car"

  test_route = TestRoute(args.route, args.car, segment=args.segment)
  test_cases = create_test_models_tests([(args.car, test_route)])

  unittest.TextTestRunner().run(test_cases)
