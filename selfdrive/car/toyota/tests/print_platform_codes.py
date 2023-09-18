#!/usr/bin/env python3
from collections import defaultdict
from cereal import car
from openpilot.selfdrive.car.toyota.values import FW_VERSIONS, PLATFORM_CODE_ECUS, get_platform_codes

Ecu = car.CarParams.Ecu
ECU_NAME = {v: k for k, v in Ecu.schema.enumerants.items()}

if __name__ == "__main__":
  parts_for_ecu: dict = defaultdict(set)

  for car_model, ecus in FW_VERSIONS.items():
    print()
    print(car_model)
    for ecu in sorted(ecus, key=lambda x: int(x[0])):
      if ecu[0] not in PLATFORM_CODE_ECUS:
        continue

      platform_codes = get_platform_codes(ecus[ecu])
      parts_for_ecu[ecu] |= {code.split(b'-')[0] for code in platform_codes}
      print(f'  (Ecu.{ECU_NAME[ecu[0]]}, {hex(ecu[1])}, {ecu[2]}):')
      print(f'    Codes: {platform_codes}')

  print('\nECU parts:')
  for ecu, parts in parts_for_ecu.items():
    print(f'  (Ecu.{ECU_NAME[ecu[0]]}, {hex(ecu[1])}, {ecu[2]}): {parts}')
