#!/usr/bin/env python3
import unittest
from parameterized import parameterized
from typing import Dict, Iterable, Optional, Tuple

import capnp

from cereal import car
from selfdrive.car.ford.values import CAR, FW_QUERY_CONFIG, FW_VERSIONS

Ecu = car.CarParams.Ecu


ECU_ADDRESSES = {
  Ecu.eps: 0x730,          # Power Steering Control Module (PSCM)
  Ecu.abs: 0x760,          # Anti-Lock Brake System (ABS)
  Ecu.fwdRadar: 0x764,     # Cruise Control Module (CCM)
  Ecu.fwdCamera: 0x706,    # Image Processing Module A (IPMA)
  Ecu.engine: 0x7E0,       # Powertrain Control Module (PCM)
  Ecu.shiftByWire: 0x732,  # Gear Shift Module (GSM)
}


ECU_FW_CORE = {
  Ecu.eps: [
    "14D003",
  ],
  Ecu.abs: [
    "2D053",
  ],
  Ecu.fwdRadar: [
    "14D049",
  ],
  Ecu.fwdCamera: [
    "14F397",  # Ford Q3
    "14H102",  # Ford Q4
  ],
  Ecu.engine: [
    "14C204",
  ],
}


class TestFordFW(unittest.TestCase):
  def test_fw_query_config(self):
    for (ecu, addr, subaddr) in FW_QUERY_CONFIG.extra_ecus:
      self.assertIn(ecu, ECU_ADDRESSES, "Unknown ECU")
      self.assertEqual(addr, ECU_ADDRESSES[ecu], "ECU address mismatch")
      self.assertIsNone(subaddr, "Unexpected ECU subaddress")

  @parameterized.expand(FW_VERSIONS.items())
  def test_fw_versions(self, car_model: str, fw_versions: Dict[Tuple[capnp.lib.capnp._EnumModule, int, Optional[int]], Iterable[bytes]]):
    self.assertIn(car_model, CAR.__dict__.values())

    for (ecu, addr, subaddr), ecu_rxs in fw_versions.items():
      self.assertIn(ecu, ECU_ADDRESSES, "Unknown ECU")
      self.assertEqual(addr, ECU_ADDRESSES[ecu], "ECU address mismatch")
      self.assertIsNone(subaddr, "Unexpected ECU subaddress")

      # Software part number takes the form: PREFIX-CORE-SUFFIX
      # Prefix changes based on the family of part. It includes the model year
      #   and likely the platform.
      # Core identifies the type of the item (e.g. 14D003 = PSCM, 14C204 = PCM).
      # Suffix specifies the version of the part. -AA would be followed by -AB.
      #   Small increments in the suffix are usually be compatible.
      # Details: https://forscan.org/forum/viewtopic.php?p=70008#p70008
      for rx in ecu_rxs:
        self.assertEqual(len(rx), 24, "Expected ECU response to be 24 bytes")

        rx_parts = rx.rstrip(b'\x00').decode().split('-')
        self.assertEqual(len(rx_parts), 3, "Expected FW to be in format: prefix-core-suffix")

        prefix, core, suffix = rx_parts
        self.assertEqual(len(prefix), 4, "Expected FW prefix to be 4 characters")
        self.assertIn(len(core), (5, 6), "Expected FW core to be 5-6 characters")
        self.assertIn(core, ECU_FW_CORE[ecu], f"Unexpected FW core for {ecu}")
        self.assertIn(len(suffix), (2, 3), "Expected FW suffix to be 2-3 characters")


if __name__ == "__main__":
  unittest.main()
