from typing import Dict

from cereal import car
from selfdrive.car import dbc_dict
from selfdrive.car.docs_definitions import CarInfo
from selfdrive.car.fw_query_definitions import FwQueryConfig, Request, StdQueries

Ecu = car.CarParams.Ecu

SPEED_FROM_RPM = 0.008587
KNEE_RAW_ANGLE_TO_DEGREES = 0.021972656 # 14 bit reading from angle sensor


class CarControllerParams:
  ANGLE_DELTA_BP = [0., 5., 15.]
  ANGLE_DELTA_V = [5., .8, .15]     # windup limit
  ANGLE_DELTA_VU = [5., 3.5, 0.4]   # unwind limit
  LKAS_MAX_TORQUE = 1               # A value of 1 is easy to overpower
  STEER_THRESHOLD = 1.0


class CAR:
  BODY = "COMMA BODY"
  BODY_KNEE = "COMMA BODY WITH KNEE"


CAR_INFO: Dict[str, CarInfo] = {
  CAR.BODY: CarInfo("comma body", package="All"),
  CAR.BODY_KNEE: CarInfo("comma body + knee", package="All"),
}

FW_QUERY_CONFIG = FwQueryConfig(
  requests=[
    Request(
      [StdQueries.TESTER_PRESENT_REQUEST, StdQueries.UDS_VERSION_REQUEST],
      [StdQueries.TESTER_PRESENT_RESPONSE, StdQueries.UDS_VERSION_RESPONSE],
      bus=0,
    ),
  ],
)

FW_VERSIONS = {
  CAR.BODY: {
    (Ecu.engine, 0x720, None): [
      b'0.0.03',
    ],
    (Ecu.debug, 0x721, None): [
      b'32023af9' # git hash of the firmware used
    ],
  },
  CAR.BODY_KNEE: {
    (Ecu.engine, 0x720, None): [
      b'0.0.03',
    ],
    (Ecu.debug, 0x721, None): [
      b'32023af9' # git hash of the firmware used
    ],
    # knee ECUs
    (Ecu.engine, 0x730, None): [
      b'0.0.03',
    ],
    (Ecu.debug, 0x731, None): [
      b'32023af9' # git hash of the firmware used
    ],
  },
}

DBC = {
  CAR.BODY: dbc_dict('comma_body', None),
  CAR.BODY_KNEE: dbc_dict('comma_body', None),
}
