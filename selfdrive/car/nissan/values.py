from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from cereal import car
from panda.python import uds
from selfdrive.car import AngleRateLimit, dbc_dict
from selfdrive.car.docs_definitions import CarInfo, CarHarness, CarParts
from selfdrive.car.fw_query_definitions import FwQueryConfig, Request, StdQueries

Ecu = car.CarParams.Ecu


class CarControllerParams:
  ANGLE_RATE_LIMIT_UP = AngleRateLimit(speed_bp=[0., 5., 15.], angle_v=[5., .8, .15])
  ANGLE_RATE_LIMIT_DOWN = AngleRateLimit(speed_bp=[0., 5., 15.], angle_v=[5., 3.5, 0.4])
  LKAS_MAX_TORQUE = 1               # A value of 1 is easy to overpower
  STEER_THRESHOLD = 1.0

  def __init__(self, CP):
    pass


class CAR:
  XTRAIL = "NISSAN X-TRAIL 2017"
  LEAF = "NISSAN LEAF 2018"
  # Leaf with ADAS ECU found behind instrument cluster instead of glovebox
  # Currently the only known difference between them is the inverted seatbelt signal.
  LEAF_IC = "NISSAN LEAF 2018 Instrument Cluster"
  ROGUE = "NISSAN ROGUE 2019"
  ALTIMA = "NISSAN ALTIMA 2020"


@dataclass
class NissanCarInfo(CarInfo):
  package: str = "ProPILOT Assist"
  car_parts: CarParts = field(default_factory=CarParts.common([CarHarness.nissan_a]))


CAR_INFO: Dict[str, Optional[Union[NissanCarInfo, List[NissanCarInfo]]]] = {
  CAR.XTRAIL: NissanCarInfo("Nissan X-Trail 2017"),
  CAR.LEAF: NissanCarInfo("Nissan Leaf 2018-23", video_link="https://youtu.be/vaMbtAh_0cY"),
  CAR.LEAF_IC: None,  # same platforms
  CAR.ROGUE: NissanCarInfo("Nissan Rogue 2018-20"),
  CAR.ALTIMA: NissanCarInfo("Nissan Altima 2019-20", car_parts=CarParts.common([CarHarness.nissan_b])),
}

NISSAN_DIAGNOSTIC_REQUEST_KWP = bytes([uds.SERVICE_TYPE.DIAGNOSTIC_SESSION_CONTROL, 0x81])
NISSAN_DIAGNOSTIC_RESPONSE_KWP = bytes([uds.SERVICE_TYPE.DIAGNOSTIC_SESSION_CONTROL + 0x40, 0x81])

NISSAN_VERSION_REQUEST_KWP = b'\x21\x83'
NISSAN_VERSION_RESPONSE_KWP = b'\x61\x83'

NISSAN_RX_OFFSET = 0x20

FW_QUERY_CONFIG = FwQueryConfig(
  requests=[
    Request(
      [NISSAN_DIAGNOSTIC_REQUEST_KWP, NISSAN_VERSION_REQUEST_KWP],
      [NISSAN_DIAGNOSTIC_RESPONSE_KWP, NISSAN_VERSION_RESPONSE_KWP],
    ),
    Request(
      [NISSAN_DIAGNOSTIC_REQUEST_KWP, NISSAN_VERSION_REQUEST_KWP],
      [NISSAN_DIAGNOSTIC_RESPONSE_KWP, NISSAN_VERSION_RESPONSE_KWP],
      rx_offset=NISSAN_RX_OFFSET,
    ),
    Request(
      [StdQueries.MANUFACTURER_SOFTWARE_VERSION_REQUEST],
      [StdQueries.MANUFACTURER_SOFTWARE_VERSION_RESPONSE],
      rx_offset=NISSAN_RX_OFFSET,
    ),
  ],
)

DBC = {
  CAR.XTRAIL: dbc_dict('nissan_x_trail_2017_generated', None),
  CAR.LEAF: dbc_dict('nissan_leaf_2018_generated', None),
  CAR.LEAF_IC: dbc_dict('nissan_leaf_2018_generated', None),
  CAR.ROGUE: dbc_dict('nissan_x_trail_2017_generated', None),
  CAR.ALTIMA: dbc_dict('nissan_x_trail_2017_generated', None),
}
