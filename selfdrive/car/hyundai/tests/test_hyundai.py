#!/usr/bin/env python3
import unittest

from cereal import car
from selfdrive.car.tests.test_fw_fingerprint import TestFwFingerprintBase
from selfdrive.car.fw_versions import match_fw_to_car
from selfdrive.car.hyundai.values import CAMERA_SCC_CAR, CANFD_CAR, CAN_GEARS, CAR, CHECKSUM, FW_QUERY_CONFIG, \
                                         FW_VERSIONS, LEGACY_SAFETY_MODE_CAR, PART_NUMBER_FW_PATTERN, \
                                         get_platform_codes_new

Ecu = car.CarParams.Ecu
ECU_NAME = {v: k for k, v in Ecu.schema.enumerants.items()}


class TestHyundaiFingerprint(TestFwFingerprintBase):
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

  def test_platform_code_ecus_available(self):
    no_eps_platforms = CANFD_CAR | {CAR.KIA_SORENTO, CAR.KIA_OPTIMA_G4, CAR.KIA_OPTIMA_G4_FL,
                                    CAR.SONATA_LF, CAR.TUCSON, CAR.GENESIS_G90, CAR.GENESIS_G80}

    # Asserts ECU keys essential for fuzzy fingerprinting are available on all platforms
    for car_model, ecus in FW_VERSIONS.items():
      with self.subTest(car_model=car_model):
        for fuzzy_ecu in FW_QUERY_CONFIG.platform_code_ecus:
          if fuzzy_ecu in (Ecu.fwdRadar, Ecu.eps) and car_model == CAR.HYUNDAI_GENESIS:
            continue
          if fuzzy_ecu == Ecu.eps and car_model in no_eps_platforms:
            continue
          self.assertIn(fuzzy_ecu, [e[0] for e in ecus])

  # def test_fuzzy_part_numbers(self):
  #   pattern =
  #   match =

  def test_fw_part_number(self):
    # Hyundai places the ECU part number in their FW versions, assert all parsable
    # Some examples of valid formats: '56310-L0010', '56310L0010', '56310/M6300'
    for car_model, ecus in FW_VERSIONS.items():
      with self.subTest(car_model=car_model):
        if car_model == CAR.HYUNDAI_GENESIS:
          raise unittest.SkipTest("No part numbers for car model")

        for ecu, fws in ecus.items():
          if ecu[0] not in FW_QUERY_CONFIG.platform_code_ecus:
            continue

          for fw in fws:
            match = PART_NUMBER_FW_PATTERN.search(fw)
            self.assertIsNotNone(match, fw)

  def test_fuzzy_fw_dates(self):
    # Some newer platforms have date codes in a different format we don't yet parse,
    # for now assert date format is consistent for all FW across each platform
    for car_model, ecus in FW_VERSIONS.items():
      with self.subTest(car_model=car_model):
        for ecu, fws in ecus.items():
          if ecu[0] not in FW_QUERY_CONFIG.platform_code_ecus:
            continue

          codes = set()
          for fw in fws:
            codes |= FW_QUERY_CONFIG.fuzzy_get_platform_codes_new([fw])

          # Either no dates should be parsed or all dates should be parsed
          self.assertEqual(len({code[1] is not None for code in codes}), 1)
          self.assertEqual(len({code[2] is not None for code in codes}), 1)

  def test_fuzzy_platform_codes(self):
    codes = get_platform_codes_new([b'\xf1\x00DH LKAS 1.1 -150210'])
    print('codes0', codes)

    codes = get_platform_codes_new([
      b'\xf1\x00DH LKAS 1.1 -150210',
      b'\xf1\x00AEhe SCC H-CUP      1.01 1.01 96400-G2000         ',
      b'\xf1\x00CV1_ RDR -----      1.00 1.01 99110-CV000         ',
    ])
    print('codes1', codes)

    codes = get_platform_codes_new([
      b'\xf1\x00DN8 MFC  AT KOR LHD 1.00 1.02 99211-L1000 190422',
      b'\xf1\x00DN8 MFC  AT RUS LHD 1.00 1.03 99211-L1000 190705',
      b'\xf1\x00DN8 MFC  AT USA LHD 1.00 1.00 99211-L0000 190716',
      b'\xf1\x00DN8 MFC  AT USA LHD 1.00 1.01 99211-L0000 191016',
      b'\xf1\x00DN8 MFC  AT USA LHD 1.00 1.03 99211-L0000 210603',
      b'\xf1\x00DN8 MFC  AT USA LHD 1.00 1.05 99211-L1000 201109',
      b'\xf1\x00DN8 MFC  AT USA LHD 1.00 1.06 99211-L1000 210325',
      b'\xf1\x00DN8 MFC  AT USA LHD 1.00 1.07 99211-L1000 211223',
    ])
    print('codes2', codes)

    # Asserts basic platform code parsing behavior
    codes = FW_QUERY_CONFIG.fuzzy_get_platform_codes([b'\xf1\x00DH LKAS 1.1 -150210'])
    self.assertEqual(codes, {b"DH-1502"})

    # Some cameras and all radars do not have dates
    codes = FW_QUERY_CONFIG.fuzzy_get_platform_codes([b'\xf1\x00AEhe SCC H-CUP      1.01 1.01 96400-G2000         '])
    self.assertEqual(codes, {b"AEhe"})

    codes = FW_QUERY_CONFIG.fuzzy_get_platform_codes([b'\xf1\x00CV1_ RDR -----      1.00 1.01 99110-CV000         '])
    self.assertEqual(codes, {b"CV1"})

    codes = FW_QUERY_CONFIG.fuzzy_get_platform_codes([
      b'\xf1\x00DH LKAS 1.1 -150210',
      b'\xf1\x00AEhe SCC H-CUP      1.01 1.01 96400-G2000         ',
      b'\xf1\x00CV1_ RDR -----      1.00 1.01 99110-CV000         ',
    ])
    self.assertEqual(codes, {b"DH-1502", b"AEhe", b"CV1"})

    # Returned platform codes must inclusively contain start/end dates
    codes = FW_QUERY_CONFIG.fuzzy_get_platform_codes([
      b'\xf1\x00LX2 MFC  AT USA LHD 1.00 1.07 99211-S8100 220222',
      b'\xf1\x00LX2 MFC  AT USA LHD 1.00 1.08 99211-S8100 211103',
      b'\xf1\x00ON  MFC  AT USA LHD 1.00 1.01 99211-S9100 190405',
      b'\xf1\x00ON  MFC  AT USA LHD 1.00 1.03 99211-S9100 190720',
    ])
    self.assertEqual(codes, {b'LX2-2111', b'LX2-2112', b'LX2-2201', b'LX2-2202',
                             b'ON-1904', b'ON-1905', b'ON-1906', b'ON-1907'})

  def test_excluded_platforms_new(self):
    # Asserts a list of platforms that will not fuzzy fingerprint with platform codes due to them being shared.
    # This list can be shrunk as we combine platforms and detect features
    excluded_platforms = {
      CAR.GENESIS_G70,
      CAR.GENESIS_G70_2020,
      CAR.TUCSON_4TH_GEN,
      CAR.TUCSON_HYBRID_4TH_GEN,
      CAR.KIA_SPORTAGE_HYBRID_5TH_GEN,
      CAR.SANTA_CRUZ_1ST_GEN,
      CAR.KIA_SPORTAGE_5TH_GEN,
      CAR.KIA_OPTIMA_G4_FL,
      CAR.KIA_NIRO_EV_2ND_GEN,
      CAR.KIA_NIRO_HEV_2021,
      CAR.SANTA_FE,
      CAR.SANTA_FE_HEV_2022,
      CAR.IONIQ_EV_2020,
      CAR.SANTA_FE_2022,
      CAR.IONIQ_PHEV,
      CAR.KIA_OPTIMA_G4,
      CAR.KIA_NIRO_PHEV,
      CAR.IONIQ,
      CAR.KIA_NIRO_HEV_2ND_GEN,
      CAR.SANTA_FE_PHEV_2022,
      CAR.HYUNDAI_GENESIS,
      CAR.KIA_STINGER,
      CAR.IONIQ_PHEV_2019,
      CAR.IONIQ_EV_LTD,
      CAR.KIA_STINGER_2022,
      CAR.IONIQ_HEV_2022,
    }

    platforms_with_shared_codes = set()
    for platform, fw_by_addr in FW_VERSIONS.items():
      car_fw = []
      for ecu, fw_versions in fw_by_addr.items():
        # Only test fuzzy ECUs so excluded platforms for platforms codes are accurate
        # We can still fuzzy match via exact FW matches
        ecu_name, addr, sub_addr = ecu
        if ecu_name not in FW_QUERY_CONFIG.platform_code_ecus:
          continue

        for fw in fw_versions:
          car_fw.append({"ecu": ecu_name, "fwVersion": fw, 'brand': 'hyundai',
                         "address": addr, "subAddress": 0 if sub_addr is None else sub_addr})

      CP = car.CarParams.new_message(carFw=car_fw)
      _, matches = match_fw_to_car(CP.carFw, allow_exact=False, log=False)
      if len(matches):
        self.assertFingerprints(matches, platform)
      else:
        platforms_with_shared_codes.add(platform)

    self.assertEqual(platforms_with_shared_codes, excluded_platforms)


if __name__ == "__main__":
  unittest.main()
