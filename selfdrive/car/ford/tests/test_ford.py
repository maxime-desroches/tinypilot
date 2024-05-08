#!/usr/bin/env python3
import unittest
from collections.abc import Iterable

import capnp
from hypothesis import settings, given, strategies as st
from parameterized import parameterized

from cereal import car
from openpilot.selfdrive.car.ford.values import CAR, FW_QUERY_CONFIG, FW_PATTERN, get_platform_codes
from openpilot.selfdrive.car.ford.fingerprints import FW_VERSIONS

Ecu = car.CarParams.Ecu


ECU_ADDRESSES = {
  Ecu.eps: 0x730,          # Power Steering Control Module (PSCM)
  Ecu.abs: 0x760,          # Anti-Lock Brake System (ABS)
  Ecu.fwdRadar: 0x764,     # Cruise Control Module (CCM)
  Ecu.fwdCamera: 0x706,    # Image Processing Module A (IPMA)
  Ecu.engine: 0x7E0,       # Powertrain Control Module (PCM)
  Ecu.shiftByWire: 0x732,  # Gear Shift Module (GSM)
  Ecu.debug: 0x7D0,        # Accessory Protocol Interface Module (APIM)
}


ECU_PART_NUMBER = {
  Ecu.eps: [
    b"14D003",
  ],
  Ecu.abs: [
    b"2D053",
  ],
  Ecu.fwdRadar: [
    b"14D049",
  ],
  Ecu.fwdCamera: [
    b"14F397",  # Ford Q3
    b"14H102",  # Ford Q4
  ],
}


class TestFordFW(unittest.TestCase):
  def test_fw_query_config(self):
    for (ecu, addr, subaddr) in FW_QUERY_CONFIG.extra_ecus:
      self.assertIn(ecu, ECU_ADDRESSES, "Unknown ECU")
      self.assertEqual(addr, ECU_ADDRESSES[ecu], "ECU address mismatch")
      self.assertIsNone(subaddr, "Unexpected ECU subaddress")

  @parameterized.expand(FW_VERSIONS.items())
  def test_fw_versions(self, car_model: str, fw_versions: dict[tuple[capnp.lib.capnp._EnumModule, int, int | None], Iterable[bytes]]):
    for (ecu, addr, subaddr), fws in fw_versions.items():
      self.assertIn(ecu, ECU_PART_NUMBER, "Unexpected ECU")
      self.assertEqual(addr, ECU_ADDRESSES[ecu], "ECU address mismatch")
      self.assertIsNone(subaddr, "Unexpected ECU subaddress")

      for fw in fws:
        self.assertEqual(len(fw), 24, "Expected ECU response to be 24 bytes")

        match = FW_PATTERN.match(fw)
        self.assertIsNotNone(match, f"Unable to parse FW: {fw!r}")
        if match:
          part_number = match.group("part_number")
          self.assertIn(part_number, ECU_PART_NUMBER[ecu], f"Unexpected part number for {fw!r}")

        codes = get_platform_codes([fw])
        self.assertEqual(1, len(codes), f"Unable to parse FW: {fw!r}")

  @settings(max_examples=100)
  @given(data=st.data())
  def test_platform_codes_fuzzy_fw(self, data):
    """Ensure function doesn't raise an exception"""
    fw_strategy = st.lists(st.binary())
    fws = data.draw(fw_strategy)
    get_platform_codes(fws)

  def test_platform_codes_spot_check(self):
    # Asserts basic platform code parsing behavior for a few cases
    results = get_platform_codes([
      b"JX6A-14C204-BPL\x00\x00\x00\x00\x00\x00\x00\x00\x00",
      b"NZ6T-14F397-AC\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
      b"PJ6T-14H102-ABJ\x00\x00\x00\x00\x00\x00\x00\x00\x00",
      b"LB5A-14C204-EAC\x00\x00\x00\x00\x00\x00\x00\x00\x00",
    ])
    self.assertEqual(results, {(b"X6A", b"J"), (b"Z6T", b"N"), (b"J6T", b"P"), (b"B5A", b"L")})

  def test_match_fw_fuzzy(self):
    offline_fw = {
      (Ecu.eps, 0x730, None): [
        b"L1MC-14D003-AJ\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        b"L1MC-14D003-AL\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
      ],
      (Ecu.abs, 0x760, None): [
        b"L1MC-2D053-BA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        b"L1MC-2D053-BD\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
      ],
      (Ecu.fwdRadar, 0x764, None): [
        b"LB5T-14D049-AB\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        b"LB5T-14D049-AD\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
      ],
      # We consider all model year hints for ECU, even with different platform codes
      (Ecu.fwdCamera, 0x706, None): [
        b"LB5T-14F397-AD\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        b"NC5T-14F397-AF\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
      ],
    }
    expected_fingerprint = CAR.FORD_EXPLORER_MK6

    # ensure that we fuzzy match on all non-exact FW with changed revisions
    live_fw = {
      (0x730, None): {b"L1MC-14D003-XX\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"},
      (0x760, None): {b"L1MC-2D053-XX\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"},
      (0x764, None): {b"LB5T-14D049-XX\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"},
      (0x706, None): {b"LB5T-14F397-XX\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"},
    }
    candidates = FW_QUERY_CONFIG.match_fw_to_car_fuzzy(live_fw, '', {expected_fingerprint: offline_fw})
    self.assertEqual(candidates, {expected_fingerprint})

    # model year hint in between the range should match
    live_fw[(0x706, None)] = {b"MB5T-14F397-XX\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"}
    candidates = FW_QUERY_CONFIG.match_fw_to_car_fuzzy(live_fw, '', {expected_fingerprint: offline_fw,})
    self.assertEqual(candidates, {expected_fingerprint})

    # unseen model year hint should not match
    live_fw[(0x760, None)] = {b"M1MC-2D053-XX\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"}
    candidates = FW_QUERY_CONFIG.match_fw_to_car_fuzzy(live_fw, '', {expected_fingerprint: offline_fw})
    self.assertEqual(len(candidates), 0, "Should not match new model year hint")


if __name__ == "__main__":
  unittest.main()
