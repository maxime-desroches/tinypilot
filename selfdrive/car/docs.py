#!/usr/bin/env python3
import argparse
from collections import defaultdict
import jinja2
import os
from enum import Enum
from natsort import natsorted
from typing import Any, Dict, List

from common.basedir import BASEDIR
from selfdrive.car.docs_definitions import STAR_DESCRIPTIONS, CarInfo, Column, StarColumns, Star, Tier
from selfdrive.car.car_helpers import interfaces, get_interface_attr
from selfdrive.car.hyundai.radar_interface import RADAR_START_ADDR as HKG_RADAR_START_ADDR
from selfdrive.car.tests.routes import non_tested_cars


def get_all_footnotes() -> Dict[Enum, int]:
  all_footnotes = []
  for footnotes in get_interface_attr("Footnote", ignore_none=True).values():
    all_footnotes += footnotes
  return {fn: idx + 1 for idx, fn in enumerate(all_footnotes)}


ALL_FOOTNOTES: Dict[Enum, int] = get_all_footnotes()
CARS_MD_OUT = os.path.join(BASEDIR, "docs", "CARS.md")
CARS_MD_TEMPLATE = os.path.join(BASEDIR, "selfdrive", "car", "CARS_template.md")


def get_all_car_info() -> List[CarInfo]:
  all_car_info: List[CarInfo] = []
  for model, car_info in get_interface_attr("CAR_INFO", combine_brands=True).items():
    # Hyundai exception: those with radar have openpilot longitudinal
    fingerprint = {0: {}, 1: {HKG_RADAR_START_ADDR: 8}, 2: {}, 3: {}}
    CP = interfaces[model][0].get_params(model, fingerprint=fingerprint, disable_radar=True)

    if CP.dashcamOnly or car_info is None:
      continue

    # A platform can include multiple car models
    if not isinstance(car_info, list):
      car_info = (car_info,)

    for _car_info in car_info:
      all_car_info.append(_car_info.init(CP, non_tested_cars, ALL_FOOTNOTES))

  # Sort cars by make and model + year
  sorted_cars: List[CarInfo] = natsorted(all_car_info, key=lambda car: (car.make + car.model).lower())
  return sorted_cars


def sort_car_info(all_car_info: List[CarInfo], by: str = "tier") -> Dict[Any, List[CarInfo]]:
  """tier sorts car info by tier, make sorts car info by market-standard vehicle make"""
  sorted_car_info = defaultdict(list)

  for car_info in all_car_info:
    sorted_car_info[getattr(car_info, by)].append(car_info)

  # Sort cars by model + year
  for key, cars in sorted_car_info.items():
    sorted_car_info[key] = natsorted(cars, key=lambda car: (car.make + car.model).lower())

  if by == "tier":
    tier_sort = [Tier.GOLD, Tier.SILVER, Tier.BRONZE]
    sorted_car_info = dict(natsorted(sorted_car_info.items(), key=lambda i: tier_sort.index(i[0])))
  elif by == "make":
    sorted_car_info = dict(natsorted(sorted_car_info.items(), key=lambda i: i[0].lower()))

  return sorted_car_info


def generate_cars_md(all_car_info: List[CarInfo], template_fn: str) -> str:
  with open(template_fn, "r") as f:
    template = jinja2.Template(f.read(), trim_blocks=True, lstrip_blocks=True)

  footnotes = [fn.value.text for fn in ALL_FOOTNOTES]
  cars_md: str = template.render(sort_car_info=sort_car_info, all_car_info=all_car_info, footnotes=footnotes,
                                 Star=Star, StarColumns=StarColumns, Column=Column, STAR_DESCRIPTIONS=STAR_DESCRIPTIONS)
  return cars_md


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Auto generates supported cars documentation",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--template", default=CARS_MD_TEMPLATE, help="Override default template filename")
  parser.add_argument("--out", default=CARS_MD_OUT, help="Override default generated filename")
  args = parser.parse_args()

  with open(args.out, 'w') as f:
    f.write(generate_cars_md(get_all_car_info(), args.template))
  print(f"Generated and written to {args.out}")
