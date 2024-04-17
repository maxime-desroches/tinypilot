from cereal import car
from openpilot.selfdrive.car.tesla.values import CAR

Ecu = car.CarParams.Ecu

FW_VERSIONS = {
  CAR.TESLA_AP2_MODELS: {
    (Ecu.adas, 0x649, None): [
      b'\x01\x00\x8b\x07\x01\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x11',
    ],
    (Ecu.electricBrakeBooster, 0x64d, None): [
      b'1037123-00-A',
    ],
    (Ecu.fwdRadar, 0x671, None): [
      b'\x01\x00W\x00\x00\x00\x07\x00\x00\x00\x00\x08\x01\x00\x00\x00\x07\xff\xfe',
    ],
    (Ecu.eps, 0x730, None): [
      b'\x10#\x01',
    ],
  },
  CAR.TESLA_MODELS_RAVEN: {
    (Ecu.electricBrakeBooster, 0x64d, None): [
      b'1037123-00-A',
    ],
    (Ecu.fwdRadar, 0x671, None): [
      b'\x01\x00\x99\x02\x01\x00\x10\x00\x00AP8.3.03\x00\x10',
    ],
    (Ecu.eps, 0x730, None): [
      b'SX_0.0.0 (99),SR013.7',
    ],
  },
  CAR.TESLA_AP3_MODEL3: {
    (Ecu.eps, 0x730, None): [
      b'TeMYG4_DCS_Update_0.0.0 (9),E4014.26.0',
      b'TeMYG4_DCS_Update_0.0.0 (13),E4014.28.1',
    ],
    (Ecu.engine, 0x606, None): [
      b'\x01\x00\x05 N\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00Z\xa7',
      b'\x01\x00\x05 N\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x003\xf2',
      b'\x01\x00\x05 [\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x003\xd8',
    ],
  },
  CAR.TESLA_AP3_MODELY: {
    (Ecu.eps, 0x730, None): [
      b'TeM3_E014p10_0.0.0 (16),Y002.18.00',
    ],
    (Ecu.engine, 0x606, None): [
      b'\x01\x00\x05 m\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00Z\xd5',
    ],
  },
}
