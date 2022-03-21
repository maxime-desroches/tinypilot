from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Union

from cereal import car
from selfdrive.car import dbc_dict
from selfdrive.car.docs_definitions import CarInfo

Ecu = car.CarParams.Ecu
GearShifter = car.CarState.GearShifter

class CarControllerParams:
  LKAS_STEP = 2                  # LKAS message frequency 50Hz

  # TODO: placeholder values pending discovery of true EPS limits
  STEER_MAX = 300                # As-yet unknown fault boundary, guessing 300 / 3.0Nm for now
  STEER_DELTA_UP = 3             # 3 unit/sec observed from factory LKAS, fault boundary unknown
  STEER_DELTA_DOWN = 3           # 3 unit/sec observed from factory LKAS, fault boundary unknown
  STEER_DRIVER_ALLOWANCE = 25
  STEER_DRIVER_MULTIPLIER = 3    # weight driver torque heavily
  STEER_DRIVER_FACTOR = 1        # from dbc


class CANBUS:
  pt = 0
  cam = 2


class DBC_FILES:
  faw = "faw"  # Used for all cars with MQB-style CAN messaging


DBC = defaultdict(lambda: dbc_dict(DBC_FILES.faw, None))  # type: Dict[str, Dict[str, str]]


class CAR:
  HONGQI_HS5_G1 = "HONGQI HS5 1ST GEN"          # First generation FAW Hongqi HS5 SUV


@dataclass
class FAWCarInfo(CarInfo):
  package: str = "Who Knows"  # FIXME
  good_torque: bool = True  # TODO


CAR_INFO: Dict[str, Union[FAWCarInfo, List[FAWCarInfo]]] = {
  CAR.HONGQI_HS5_G1: FAWCarInfo("FAW Hongqi HS5 ????"),
}


# TODO -- UDS fingerprinting known to work relatively easily
# FW_VERSIONS = {}
