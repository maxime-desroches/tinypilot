from cereal import car
from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union, no_type_check


class Tier(Enum):
  GOLD = "The best openpilot experience. Great highway driving and beyond."
  SILVER = "A solid highway driving experience, but is limited by stock longitudinal. May be upgraded in the future."
  BRONZE = "A good highway experience, but may have limited performance in traffic and on sharp turns."


class Column(Enum):
  MAKE = "Make"
  MODEL = "Model"
  PACKAGE = "Supported Package"
  LONGITUDINAL = "openpilot ACC"
  FSR_LONGITUDINAL = "Stop and Go"
  FSR_STEERING = "Steer to 0"
  STEERING_TORQUE = "Steering Torque"
  MAINTAINED = "Actively Maintained"


class Star(Enum):
  FULL = "full"
  HALF = "half"
  EMPTY = "empty"


StarColumns = list(Column)[3:]
CarFootnote = namedtuple("CarFootnote", ["text", "column", "star"], defaults=[None])
HarnessInfo = namedtuple("HarnessInfo", ["harness", "name"])


def get_footnote(footnotes: Optional[List[Enum]], column: Column) -> Optional[Enum]:
  # Returns applicable footnote given current column
  if footnotes is not None:
    for fn in footnotes:
      if fn.value.column == column:
        return fn
  return None


@dataclass
class CarInfo:
  name: str
  package: str
  video_link: Optional[str] = None
  footnotes: Optional[List[Enum]] = None
  min_steer_speed: Optional[float] = None
  min_enable_speed: Optional[float] = None
  good_torque: bool = False
  harness: Optional[Enum] = None

  def init(self, CP: car.CarParams, non_tested_cars: List[str], all_footnotes: Dict[Enum, int]):
    # TODO: set all the min steer speeds in carParams and remove this
    min_steer_speed = CP.minSteerSpeed
    if self.min_steer_speed is not None:
      min_steer_speed = self.min_steer_speed
      assert CP.minSteerSpeed == 0, f"Minimum steer speed set in both CarInfo and CarParams for {CP.carFingerprint}"

    # TODO: set all the min enable speeds in carParams correctly and remove this
    min_enable_speed = CP.minEnableSpeed
    if self.min_enable_speed is not None:
      min_enable_speed = self.min_enable_speed

    self.car_name = CP.carName
    self.make, self.model = self.name.split(' ', 1)
    self.row = {
      Column.MAKE: self.make,
      Column.MODEL: self.model,
      Column.PACKAGE: self.package,
      # StarColumns
      Column.LONGITUDINAL: CP.openpilotLongitudinalControl and not CP.radarOffCan,
      Column.FSR_LONGITUDINAL: min_enable_speed <= 0.,
      Column.FSR_STEERING: min_steer_speed <= 0.,
      Column.STEERING_TORQUE: self.good_torque,
      # TODO: when all harnesses are added, add check to maintained
      Column.MAINTAINED: CP.carFingerprint not in non_tested_cars,
    }

    if CP.notCar:
      for col in StarColumns:
        self.row[col] = True

    self.all_footnotes = all_footnotes
    for column in StarColumns:
      self.row[column] = Star.FULL if self.row[column] else Star.EMPTY

      # Demote if footnote specifies a star
      footnote = get_footnote(self.footnotes, column)
      if footnote is not None and footnote.value.star is not None:
        self.row[column] = footnote.value.star

    self.tier = {5: Tier.GOLD, 4: Tier.SILVER}.get(list(self.row.values()).count(Star.FULL), Tier.BRONZE)

  @no_type_check
  def get_column(self, column: Column, star_icon: str, footnote_tag: str) -> str:
    item: Union[str, Star] = self.row[column]
    if column in StarColumns:
      item = star_icon.format(item.value)

    footnote = get_footnote(self.footnotes, column)
    if footnote is not None:
      item += footnote_tag.format(self.all_footnotes[footnote])

    return item


class Harness(Enum):
  # TODO: use strings, remove HarnessInfo, and use Harness(Enum).name, .value
  nidec = HarnessInfo("nidec", "Honda Nidec")
  bosch = HarnessInfo("bosch", "Honda Bosch")
  toyota = HarnessInfo("toyota", "Tubaru")
  subaru = HarnessInfo("subaru", "Subaru")
  fca = HarnessInfo("fca", "FCA")
  vw = HarnessInfo("vw", "VW")
  j533 = HarnessInfo("j533", "J533")
  hyundai_abababab = HarnessInfo("hyundai-abababab", "Hyundai A")
  hyundai_a = HarnessInfo("hyundai-a", "Hyundai A")
  hyundai_b = HarnessInfo("hyundai-b", "Hyundai B")
  hyundai_c = HarnessInfo("hyundai-c", "Hyundai C")
  hyundai_d = HarnessInfo("hyundai-d", "Hyundai D")
  hyundai_e = HarnessInfo("hyundai-e", "Hyundai E")
  hyundai_f = HarnessInfo("hyundai-f", "Hyundai F")
  hyundai_g = HarnessInfo("hyundai-g", "Hyundai G")
  hyundai_h = HarnessInfo("hyundai-h", "Hyundai H")
  hyundai_i = HarnessInfo("hyundai-i", "Hyundai I")
  hyundai_j = HarnessInfo("hyundai-j", "Hyundai J")
  hyundai_k = HarnessInfo("hyundai-k", "Hyundai K")
  hyundai_l = HarnessInfo("hyundai-l", "Hyundai L")
  hyundai_m = HarnessInfo("hyundai-m", "Hyundai M")
  hyundai_n = HarnessInfo("hyundai-n", "Hyundai N")
  hyundai_o = HarnessInfo("hyundai-o", "Hyundai O")
  custom = HarnessInfo("custom", "Developer")
  obd_ii = HarnessInfo("obd-ii", "OBD-II")
  nissan_a = HarnessInfo("nissan-a", "Nissan A")
  nissan_b = HarnessInfo("nissan-b", "Nissan B")
  mazda = HarnessInfo("mazda", "Mazda")
