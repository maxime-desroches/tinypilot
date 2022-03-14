#!/usr/bin/env python3
import jinja2
import os
from collections import defaultdict, namedtuple
from enum import Enum
from sortedcontainers import SortedList
from typing import Dict

from common.basedir import BASEDIR
from common.params import Params
from selfdrive.car.car_helpers import interfaces, get_interface_attr
from selfdrive.car.gm.values import CAR as GM
from selfdrive.car.hyundai.radar_interface import RADAR_START_ADDR as HKG_RADAR_START_ADDR
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
CarFootnote = namedtuple("CarFootnote", ["text", "column", "star"], defaults=[None])


def get_star_icon(variant):
  return '<img src="assets/icon-star-{}.svg" width="22" />'.format(variant)


def get_footnote(car_info, column) -> CarFootnote:
  if car_info.footnotes is None:
    return None

  for fn in car_info.footnotes:
    if CAR_FOOTNOTES[fn].column == column:
      return CAR_FOOTNOTES[fn]


CARS_MD_OUT = os.path.join(BASEDIR, "docs", "CARS_generated.md")

# TODO: which other makes?
MAKES_GOOD_STEERING_TORQUE = ["toyota", "hyundai", "volkswagen"]
CAR_FOOTNOTES = {
  1: CarFootnote("When disconnecting the Driver Support Unit (DSU), openpilot Adaptive Cruise Control (ACC) will replace " \
                 "stock Adaptive Cruise Control (ACC). NOTE: disconnecting the DSU disables Automatic Emergency Braking (AEB).",
                 Column.LONGITUDINAL, star="half"),
  2: CarFootnote("28mph for Camry 4CYL L, 4CYL LE and 4CYL SE which don't have Full-Speed Range Dynamic Radar Cruise Control.",
                 Column.FSR_LONGITUDINAL),
  3: CarFootnote("Requires an [OBD-II](https://comma.ai/shop/products/comma-car-harness) car harness and [community built ASCM harness]" \
                 "(https://github.com/commaai/openpilot/wiki/GM#hardware). NOTE: disconnecting the ASCM disables Automatic Emergency Braking (AEB).",
                 Column.MODEL),
  4: CarFootnote("Not including the China market Kamiq, which is based on the (currently) unsupported PQ34 platform.",
                 Column.MODEL),
  5: CarFootnote("Not including the USA/China market Passat, which is based on the (currently) unsupported PQ35/NMS platform.",
                 Column.MODEL),
  6: CarFootnote("Model-years 2021 and beyond may have a new camera harness design, which isn't yet available from the comma " \
                 "store. Before ordering, remove the Lane Assist camera cover and check to see if the connector is black " \
                 "(older design) or light brown (newer design). For the newer design, in the interim, choose \"VW J533 Development\" " \
                 "from the vehicle drop-down for a harness that integrates at the CAN gateway inside the dashboard.",
                 Column.MODEL),
  7: CarFootnote("An inaccurate steering wheel angle sensor makes precise control difficult.",
                 Column.STEERING_TORQUE, star="half"),
}


class Car:
  def __init__(self, car_info, CP):
    self.make, self.model = car_info.name.split(' ', 1)
    self.row, star_count = self.get_row(car_info, CP)
    self.tier = {5: Tier.GOLD, 4: Tier.SILVER}.get(star_count, Tier.BRONZE)

  def get_row(self, car_info, CP):
    # TODO: add YouTube videos
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
    row = [self.make, self.model, car_info.package, *map(lambda star: "full" if star else "empty", stars)]

    # Check for car footnotes and star demotions
    star_count = 0
    for row_idx, column in enumerate(Column):
      footnote = get_footnote(car_info, column)
      if column in StarColumns:
        # Demote if footnote specifies a star
        if footnote is not None and footnote.star is not None:
          row[row_idx] = footnote.star
        star_count += row[row_idx] == "full"
        row[row_idx] = get_star_icon(row[row_idx])

      if footnote is not None:
        superscript_number = list(CAR_FOOTNOTES.values()).index(footnote) + 1
        row[row_idx] += "<sup>{}</sup>".format(superscript_number)

    return row, star_count


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

  footnotes = [footnote.text for footnote in CAR_FOOTNOTES.values()]
  return template.render(tiers=tiered_cars, columns=[column.value for column in Column], footnotes=footnotes)


if __name__ == "__main__":
  # TODO: add argparse for generating json or html (undecided)
  # Cars that can disable radar have openpilot longitudinal
  Params().put_bool("DisableRadar", True)

  tiered_cars = get_tiered_cars()
  with open(CARS_MD_OUT, 'w') as f:
    f.write(generate_cars_md(tiered_cars))

  print('Generated and written to {}'.format(CARS_MD_OUT))
