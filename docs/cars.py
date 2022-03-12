#!/usr/bin/env python3
from collections import defaultdict, namedtuple
from enum import Enum
import jinja2
import os
from sortedcontainers import SortedList
from typing import Dict

from common.basedir import BASEDIR
from common.params import Params
from selfdrive.car.car_helpers import interfaces, get_interface_attr
from selfdrive.car.chrysler.values import CAR as CHRYSLER
from selfdrive.car.gm.values import CAR as GM
from selfdrive.car.honda.values import CAR as HONDA
from selfdrive.car.hyundai.radar_interface import RADAR_START_ADDR as HKG_RADAR_START_ADDR
from selfdrive.car.hyundai.values import CAR as HYUNDAI
from selfdrive.car.toyota.values import CAR as TOYOTA
from selfdrive.car.volkswagen.values import CAR as VOLKSWAGEN
from selfdrive.test.test_routes import non_tested_cars


class Tier(Enum):
  GOLD = "Gold"
  SILVER = "Silver"
  BRONZE = "Bronze"


class Column(Enum):
  MAKE = "Make"
  MODEL = "Model"
  PACKAGE = "Supported Package"
  LONGITUDINAL = "openpilot Longitudinal"
  FSR_LONGITUDINAL = "FSR Longitudinal"
  FSR_STEERING = "FSR Steering"
  STEERING_TORQUE = "Steering Torque"
  SUPPORTED = "Actively Maintained"


StarColumns = list(Column)[3:]
CarException = namedtuple("CarException", ["cars", "text", "column", "star"], defaults=[None])


def get_star_icon(variant):
  return '<img src="assets/icon-star-{}.png" width="22" />'.format(variant)


def get_exceptions(CP) -> Dict[Column, CarException]:
  exceptions = {}
  for car_exception in CAR_EXCEPTIONS:
    if CP.carFingerprint in car_exception.cars:
      exceptions[car_exception.column] = car_exception
  return exceptions


CARS_MD_OUT = os.path.join(BASEDIR, "docs", "CARS_generated.md")

# TODO: which other makes?
MAKES_GOOD_STEERING_TORQUE = ["toyota", "hyundai", "volkswagen"]
CAR_EXCEPTIONS = [
  CarException([TOYOTA.LEXUS_CTH, TOYOTA.LEXUS_ESH, TOYOTA.LEXUS_NX, TOYOTA.LEXUS_NXH, TOYOTA.LEXUS_RX,
                TOYOTA.LEXUS_RXH, TOYOTA.AVALON, TOYOTA.AVALONH_2019, TOYOTA.COROLLA, TOYOTA.HIGHLANDER,
                TOYOTA.HIGHLANDERH, TOYOTA.PRIUS, TOYOTA.PRIUS_V, TOYOTA.RAV4, TOYOTA.RAV4H, TOYOTA.SIENNA],
               "When disconnecting the Driver Support Unit (DSU), openpilot Adaptive Cruise Control (ACC) will replace "
               "stock Adaptive Cruise Control (ACC). NOTE: disconnecting the DSU disables Automatic Emergency Braking (AEB).",
               Column.LONGITUDINAL, star="half"),
  CarException([TOYOTA.CAMRY, TOYOTA.CAMRY_TSS2, TOYOTA.CAMRYH],
               "28mph for Camry 4CYL L, 4CYL LE and 4CYL SE which don't have Full-Speed Range Dynamic Radar Cruise Control.",
               Column.FSR_LONGITUDINAL),
  CarException([GM.ESCALADE_ESV, GM.VOLT, GM.ACADIA],
               "Requires an [OBD-II](https://comma.ai/shop/products/comma-car-harness) car harness and [community built ASCM harness]"
               "(https://github.com/commaai/openpilot/wiki/GM#hardware). NOTE: disconnecting the ASCM disables Automatic Emergency Braking (AEB).",
               Column.MODEL),
  CarException([VOLKSWAGEN.SKODA_KAMIQ_MK1],
               "Not including the China market Kamiq, which is based on the (currently) unsupported PQ34 platform.",
               Column.MODEL),
  CarException([VOLKSWAGEN.PASSAT_MK8],
               "Not including the USA/China market Passat, which is based on the (currently) unsupported PQ35/NMS platform.",
               Column.MODEL),
  CarException([VOLKSWAGEN.ARTEON_MK1, VOLKSWAGEN.ATLAS_MK1, VOLKSWAGEN.TRANSPORTER_T61, VOLKSWAGEN.TCROSS_MK1,
                VOLKSWAGEN.TROC_MK1, VOLKSWAGEN.TAOS_MK1, VOLKSWAGEN.TIGUAN_MK2],
               'Model-years 2021 and beyond may have a new camera harness design, which isn\'t yet available from the comma '
               'store. Before ordering, remove the Lane Assist camera cover and check to see if the connector is black '
               '(older design) or light brown (newer design). For the newer design, in the interim, choose "VW J533 Development" '
               'from the vehicle drop-down for a harness that integrates at the CAN gateway inside the dashboard.',
               Column.MODEL),
  CarException([TOYOTA.PRIUS, TOYOTA.PRIUS_V],
               "An inaccurate steering wheel angle sensor makes precise control difficult.",
               Column.STEERING_TORQUE, star="half"),
]

CAR_VIDEOS = {
  HYUNDAI.ELANTRA_2021: "https://youtu.be/_EdYQtV52-c",
  HYUNDAI.ELANTRA_HEV_2021: "https://youtu.be/_EdYQtV52-c",
  HYUNDAI.KONA_HEV: "https://youtu.be/_EdYQtV52-c",
  HYUNDAI.PALISADE: "https://youtu.be/TAnDqjF4fDY?t=456",
  HYUNDAI.SONATA: "https://www.youtube.com/watch?v=ix63r9kE3Fw",
  HYUNDAI.KIA_NIRO_EV: "https://www.youtube.com/watch?v=lT7zcG6ZpGo",
  TOYOTA.COROLLA_TSS2: "https://www.youtube.com/watch?v=_66pXk0CBYA",
  TOYOTA.PRIUS_TSS2: "https://www.youtube.com/watch?v=J58TvCpUd4U",
  HYUNDAI.KIA_STINGER: "https://www.youtube.com/watch?v=MJ94qoofYw0",
  TOYOTA.CAMRY: "https://www.youtube.com/watch?v=fkcjviZY9CM",
  TOYOTA.CAMRYH: "https://www.youtube.com/watch?v=Q2DYY0AWKgk",
  TOYOTA.SIENNA: "https://www.youtube.com/watch?v=q1UPOo4Sh68",
  TOYOTA.HIGHLANDER: "https://www.youtube.com/watch?v=0wS0wXSLzoo",
  TOYOTA.PRIUS: "https://www.youtube.com/watch?v=8zopPJI8XQ0",
  TOYOTA.RAV4_TSS2: "https://www.youtube.com/watch?v=wJxjDd42gGA",
  HONDA.ACCORD: "https://www.youtube.com/watch?v=mrUwlj3Mi58",
  HONDA.CIVIC_BOSCH: "https://www.youtube.com/watch?v=4Iz1Mz5LGF8",
  CHRYSLER.JEEP_CHEROKEE: "https://www.youtube.com/watch?v=eLR9o2JkuRk",
  CHRYSLER.JEEP_CHEROKEE_2019: "https://www.youtube.com/watch?v=jBe4lWnRSu4",
  HYUNDAI.KIA_SORENTO: "https://www.youtube.com/watch?v=Fkh3s6WHJz8",
}


class Car:
  def __init__(self, car_info, CP):
    self.make, self.model = car_info.name.split(' ', 1)
    self.package = car_info.package
    self.exceptions = get_exceptions(CP)
    self.stars = self._calculate_stars(CP, car_info)

  @property
  def row(self):
    # TODO: add YouTube videos
    row = [self.make, self.model, self.package, *map(get_star_icon, self.stars)]

    # Check for car exceptions
    for row_idx, column in enumerate(Column):
      exception = self.exceptions.get(column, None)
      if exception is not None:
        superscript_number = CAR_EXCEPTIONS.index(exception) + 1
        row[row_idx] += "<sup>{}</sup>".format(superscript_number)

    return row

  @property
  def tier(self):
    return {5: Tier.GOLD, 4: Tier.SILVER}.get(self.stars.count("full"), Tier.BRONZE)

  def _calculate_stars(self, CP, car_info):
    # TODO: can we incorporate this into row()?
    # Some minimum steering speeds are not yet in CarParams
    min_steer_speed = CP.minSteerSpeed
    if car_info.min_steer_speed is not None:
      min_steer_speed = car_info.min_steer_speed
      assert CP.minSteerSpeed == 0, "Minimum steer speed set in both CarInfo and CarParams for {}".format(CP.carFingerprint)

    min_enable_speed = CP.minEnableSpeed
    if car_info.min_enable_speed is not None:
      min_enable_speed = car_info.min_enable_speed

    # TODO: make sure well supported check is complete
    stars = [CP.openpilotLongitudinalControl and not CP.radarOffCan, min_enable_speed <= 1e-3, min_steer_speed <= 1e-3,
             CP.carName in MAKES_GOOD_STEERING_TORQUE, CP.carFingerprint not in non_tested_cars]

    # Check for star demotions from exceptions
    for idx, (star, column) in enumerate(zip(stars, StarColumns)):
      star = "full" if star else "empty"
      exception = self.exceptions.get(column, None)
      if exception is not None and exception.star is not None:
        star = exception.star.lower()
      stars[idx] = star
    return stars


def get_tiered_cars():
  # Keep track of cars while sorting by make, model name, and year
  tiered_cars = {tier: SortedList(key=lambda car: car.make + car.model) for tier in Tier}

  for _, models in get_interface_attr("CAR_INFO").items():
    for model, car_info in models.items():
      # Hyundai exception: all have openpilot longitudinal
      fingerprint = defaultdict(dict)
      fingerprint[1] = {HKG_RADAR_START_ADDR: 8}
      CP = interfaces[model][0].get_params(model, fingerprint=fingerprint)
      # Skip community supported
      if CP.dashcamOnly:
        continue

      # Some candidates have multiple variants
      if not isinstance(car_info, list):
        car_info = [car_info]

      for _car_info in car_info:
        car = Car(_car_info, CP)
        tiered_cars[car.tier].add(car)

  # Return tier name and car rows for each tier
  for tier, cars in tiered_cars.items():
    yield [tier.name.title(), map(lambda car: car.row, cars)]


def generate_cars_md(tiered_cars):
  template_fn = os.path.join(BASEDIR, "docs", "CARS_template.md")
  with open(template_fn, "r") as f:
    template = jinja2.Template(f.read(), trim_blocks=True, lstrip_blocks=True)  # TODO: remove lstrip_blocks if not needed

  exceptions = [exception.text for exception in CAR_EXCEPTIONS]
  return template.render(tiers=tiered_cars, columns=[column.value for column in Column], exceptions=exceptions)


if __name__ == "__main__":
  # TODO: add argparse for generating json or html (undecided)
  # Cars that can disable radar have openpilot longitudinal
  Params().put_bool("DisableRadar", True)

  tiered_cars = get_tiered_cars()
  with open(CARS_MD_OUT, 'w') as f:
    f.write(generate_cars_md(tiered_cars))

  print('Generated and written to {}'.format(CARS_MD_OUT))
