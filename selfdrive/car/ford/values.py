import re
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import Dict, List, Union

from cereal import car
from openpilot.selfdrive.car import AngleRateLimit, dbc_dict
from openpilot.selfdrive.car.docs_definitions import CarFootnote, CarHarness, CarInfo, CarParts, Column, \
                                                     Device
from openpilot.selfdrive.car.fw_query_definitions import FwQueryConfig, Request, StdQueries

Ecu = car.CarParams.Ecu


class CarControllerParams:
  STEER_STEP = 5        # LateralMotionControl, 20Hz
  LKA_STEP = 3          # Lane_Assist_Data1, 33Hz
  ACC_CONTROL_STEP = 2  # ACCDATA, 50Hz
  LKAS_UI_STEP = 100    # IPMA_Data, 1Hz
  ACC_UI_STEP = 20      # ACCDATA_3, 5Hz
  BUTTONS_STEP = 5      # Steering_Data_FD1, 10Hz, but send twice as fast

  CURVATURE_MAX = 0.02  # Max curvature for steering command, m^-1
  STEER_DRIVER_ALLOWANCE = 1.0  # Driver intervention threshold, Nm

  # Curvature rate limits
  # The curvature signal is limited to 0.003 to 0.009 m^-1/sec by the EPS depending on speed and direction
  # Limit to ~2 m/s^3 up, ~3 m/s^3 down at 75 mph
  # Worst case, the low speed limits will allow 4.3 m/s^3 up, 4.9 m/s^3 down at 75 mph
  ANGLE_RATE_LIMIT_UP = AngleRateLimit(speed_bp=[5, 25], angle_v=[0.0002, 0.0001])
  ANGLE_RATE_LIMIT_DOWN = AngleRateLimit(speed_bp=[5, 25], angle_v=[0.000225, 0.00015])
  CURVATURE_ERROR = 0.002  # ~6 degrees at 10 m/s, ~10 degrees at 35 m/s

  ACCEL_MAX = 2.0               # m/s^2 max acceleration
  ACCEL_MIN = -3.5              # m/s^2 max deceleration
  MIN_GAS = -0.5
  INACTIVE_GAS = -5.0

  def __init__(self, CP):
    pass


class CAR(StrEnum):
  BRONCO_SPORT_MK1 = "FORD BRONCO SPORT 1ST GEN"
  ESCAPE_MK4 = "FORD ESCAPE 4TH GEN"
  EXPLORER_MK6 = "FORD EXPLORER 6TH GEN"
  F_150_MK14 = "FORD F-150 14TH GEN"
  FOCUS_MK4 = "FORD FOCUS 4TH GEN"
  MAVERICK_MK1 = "FORD MAVERICK 1ST GEN"
  F_150_LIGHTNING_MK1 = "FORD F-150 LIGHTNING 1ST GEN"
  MUSTANG_MACH_E_MK1 = "FORD MUSTANG MACH-E 1ST GEN"


CANFD_CAR = {CAR.F_150_MK14, CAR.F_150_LIGHTNING_MK1, CAR.MUSTANG_MACH_E_MK1}


class RADAR:
  DELPHI_ESR = 'ford_fusion_2018_adas'
  DELPHI_MRR = 'FORD_CADS'


DBC: Dict[str, Dict[str, str]] = defaultdict(lambda: dbc_dict("ford_lincoln_base_pt", RADAR.DELPHI_MRR))

# F-150 radar is not yet supported
DBC[CAR.F_150_MK14] = dbc_dict("ford_lincoln_base_pt", None)
DBC[CAR.F_150_LIGHTNING_MK1] = dbc_dict("ford_lincoln_base_pt", None)
DBC[CAR.MUSTANG_MACH_E_MK1] = dbc_dict("ford_lincoln_base_pt", None)


class Footnote(Enum):
  FOCUS = CarFootnote(
    "Refers only to the Focus Mk4 (C519) available in Europe/China/Taiwan/Australasia, not the Focus Mk3 (C346) in " +
    "North and South America/Southeast Asia.",
    Column.MODEL,
  )


@dataclass
class FordCarInfo(CarInfo):
  package: str = "Co-Pilot360 Assist+"

  def init_make(self, CP: car.CarParams):
    harness = CarHarness.ford_q4 if CP.carFingerprint in CANFD_CAR else CarHarness.ford_q3
    if CP.carFingerprint in (CAR.BRONCO_SPORT_MK1, CAR.MAVERICK_MK1, CAR.F_150_MK14):
      self.car_parts = CarParts([Device.threex_angled_mount, harness])
    else:
      self.car_parts = CarParts([Device.threex, harness])


CAR_INFO: Dict[str, Union[CarInfo, List[CarInfo]]] = {
  CAR.BRONCO_SPORT_MK1: FordCarInfo("Ford Bronco Sport 2021-22"),
  CAR.ESCAPE_MK4: [
    FordCarInfo("Ford Escape 2020-22"),
    FordCarInfo("Ford Kuga 2020-22", "Adaptive Cruise Control with Lane Centering"),
  ],
  CAR.EXPLORER_MK6: [
    FordCarInfo("Ford Explorer 2020-23"),
    FordCarInfo("Lincoln Aviator 2020-21", "Co-Pilot360 Plus"),
  ],
  CAR.F_150_MK14: FordCarInfo("Ford F-150 2023", "Co-Pilot360 Active 2.0"),
  CAR.F_150_LIGHTNING_MK1: FordCarInfo("Ford F-150 Lightning 2021-23", "Co-Pilot360 Active 2.0"),
  CAR.MUSTANG_MACH_E_MK1: FordCarInfo("Ford Mustang Mach-E 2021-23", "Co-Pilot360 Active 2.0"),
  CAR.FOCUS_MK4: FordCarInfo("Ford Focus 2018", "Adaptive Cruise Control with Lane Centering", footnotes=[Footnote.FOCUS]),
  CAR.MAVERICK_MK1: [
    FordCarInfo("Ford Maverick 2022", "LARIAT Luxury"),
    FordCarInfo("Ford Maverick 2023", "Co-Pilot360 Assist"),
  ],
}


# FW response contains a combined software and part number
# A-Z except no I, O or W
# e.g. NZ6A-14C204-AAA
#      1222-333333-444
# 1 = Model year (incremented for each model year)
# 2 = Platform hint
# 3 = Part number (effectively maps to ECU)
# 4 = Software version (reset to AA for each model year)
# https://regexr.com/7qu7h
FW_ALPHABET = b'A-HJ-NP-VX-Z'
FW_RE = re.compile(b'^(?P<model_year>[' + FW_ALPHABET + b'])' +
                   b'(?P<platform_hint>[0-9' + FW_ALPHABET + b']{3})-' +
                   b'(?P<part_number>[0-9' + FW_ALPHABET + b']{5,6})-' +
                   b'(?P<software_version>[' + FW_ALPHABET + b']{2,})$')


def get_platform_codes(fw_versions: list[bytes]) -> set[tuple[bytes, bytes]]:
  codes = set()  # (platform-part, year-version)

  for firmware in fw_versions:
    m = FW_RE.match(firmware.rstrip(b'\0'))
    if m is None:
      continue

    # since "AAA" is higher than "ZZ", prepend "A" to two-letter versions (i.e. "ZZ" -> "AZZ")
    software_version = (b'A' + m.group('software_version'))[-3:]

    code = b'-'.join([m.group('platform_hint'), m.group('part_number')])
    version = b'-'.join([m.group('model_year'), software_version])
    codes.add((code, version))

  return codes


def match_fw_to_car_fuzzy(live_fw_versions, offline_fw_versions) -> set[str]:
  candidates: set[str] = set()

  for candidate, fws in offline_fw_versions.items():
    # Keep track of ECUs which pass all checks (platform codes, within version range)
    valid_found_ecus = set()
    valid_expected_ecus = {ecu[1:] for ecu in fws}
    for ecu, ecu_fws in fws.items():
      addr = ecu[1:]

      # Expected platform codes and versions
      codes = get_platform_codes(ecu_fws)
      expected_platform_codes = {code for code, _ in codes}
      expected_versions = {version for _, version in codes}

      # Live platform codes and versions
      live_codes = get_platform_codes(live_fw_versions.get(addr, set()))
      found_platform_codes = {code for code, _ in live_codes}
      found_versions = {version for _, version in live_codes}

      # Check platform code + version matches for any found versions
      if not any(found_code in expected_platform_codes for found_code in found_platform_codes):
        break

      # Check version is within expected range
      if not any(min(expected_versions) <= found_version <= max(expected_versions) for found_version in found_versions):
        break

      valid_found_ecus.add(addr)

    # If all live ECUs pass all checks for candidate, add it as a match
    if valid_expected_ecus.issubset(valid_found_ecus):
      candidates.add(candidate)

  return candidates


FW_QUERY_CONFIG = FwQueryConfig(
  requests=[
    # CAN and CAN FD queries are combined.
    # FIXME: For CAN FD, ECUs respond with frames larger than 8 bytes on the powertrain bus
    # TODO: properly handle auxiliary requests to separate queries and add back whitelists
    Request(
      [StdQueries.TESTER_PRESENT_REQUEST, StdQueries.MANUFACTURER_SOFTWARE_VERSION_REQUEST],
      [StdQueries.TESTER_PRESENT_RESPONSE, StdQueries.MANUFACTURER_SOFTWARE_VERSION_RESPONSE],
      # whitelist_ecus=[Ecu.engine],
    ),
    Request(
      [StdQueries.TESTER_PRESENT_REQUEST, StdQueries.MANUFACTURER_SOFTWARE_VERSION_REQUEST],
      [StdQueries.TESTER_PRESENT_RESPONSE, StdQueries.MANUFACTURER_SOFTWARE_VERSION_RESPONSE],
      # whitelist_ecus=[Ecu.eps, Ecu.abs, Ecu.fwdRadar, Ecu.fwdCamera, Ecu.shiftByWire],
      bus=0,
      auxiliary=True,
    ),
  ],
  extra_ecus=[
    (Ecu.shiftByWire, 0x732, None),
  ],
  # Custom fuzzy fingerprinting function using platform codes, part numbers and software versions
  match_fw_to_car_fuzzy=match_fw_to_car_fuzzy,
)
