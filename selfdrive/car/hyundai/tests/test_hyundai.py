#!/usr/bin/env python3
import unittest

from cereal import car
from selfdrive.car.hyundai.values import CANFD_CAR, FW_QUERY_CONFIG, FW_VERSIONS, CAN_GEARS, LEGACY_SAFETY_MODE_CAR, CHECKSUM, CAMERA_SCC_CAR

Ecu = car.CarParams.Ecu
ECU_NAME = {v: k for k, v in Ecu.schema.enumerants.items()}


class TestHyundaiFingerprint(unittest.TestCase):
  def test_canfd_not_in_can_features(self):
    can_specific_feature_list = set.union(*CAN_GEARS.values(), *CHECKSUM.values(), LEGACY_SAFETY_MODE_CAR, CAMERA_SCC_CAR)
    for car_model in CANFD_CAR:
      self.assertNotIn(car_model, can_specific_feature_list, "CAN FD car unexpectedly found in a CAN feature list")

  def test_auxiliary_request_ecu_whitelist(self):
    # Asserts only auxiliary Ecus can exist in database for CAN-FD cars
    whitelisted_ecus = {ecu for r in FW_QUERY_CONFIG.requests for ecu in r.whitelist_ecus if r.auxiliary}

    for car_model in CANFD_CAR:
      ecus = {fw[0] for fw in FW_VERSIONS[car_model].keys()}
      ecus_not_in_whitelist = ecus - whitelisted_ecus
      ecu_strings = ", ".join([f'Ecu.{ECU_NAME[ecu]}' for ecu in ecus_not_in_whitelist])
      self.assertEqual(len(ecus_not_in_whitelist), 0, f'{car_model}: Car model has ECUs not in auxiliary request whitelists: {ecu_strings}')

  def test_blacklisted_fws(self):
    blacklisted_fw = {(Ecu.fwdCamera, 0x7c4, None): [b'\xf1\x00NX4 FR_CMR AT USA LHD 1.00 1.00 99211-CW010 14X']}
    for car_model in FW_VERSIONS.keys():
      for ecu, fw in FW_VERSIONS[car_model].items():
        if ecu in blacklisted_fw:
          common_fw = set(fw).intersection(blacklisted_fw[ecu])
          self.assertTrue(len(common_fw) == 0, f'{car_model}: Blacklisted fw version found in database: {common_fw}')


if __name__ == "__main__":
  unittest.main()
