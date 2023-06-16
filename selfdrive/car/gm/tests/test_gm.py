#!/usr/bin/env python3
from parameterized import parameterized
import unittest

from selfdrive.car.gm.values import CAMERA_ACC_CAR, CAR, FINGERPRINTS, GM_RX_OFFSET

CAMERA_DIAGNOSTIC_ADDRESS = 0x24b


class TestGMFingerprint(unittest.TestCase):
  @parameterized.expand(FINGERPRINTS.items())
  def test_can_fingerprints(self, car_model, fingerprints):
    self.assertGreater(len(fingerprints), 0)

    # Trailblazer is in dashcam
    if car_model != CAR.TRAILBLAZER:
      self.assertTrue(all(len(finger) for finger in fingerprints))

    # The camera can sometimes be communicating on startup
    if car_model in CAMERA_ACC_CAR - {CAR.TRAILBLAZER}:
      for finger in fingerprints:
        self.assertIn(CAMERA_DIAGNOSTIC_ADDRESS + GM_RX_OFFSET, finger)
        self.assertEqual(finger[CAMERA_DIAGNOSTIC_ADDRESS + GM_RX_OFFSET], 8)


if __name__ == "__main__":
  unittest.main()
