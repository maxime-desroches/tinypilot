from enum import IntFlag
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from cereal import car
from panda.python import uds
from selfdrive.car import dbc_dict
from selfdrive.car.docs_definitions import CarInfo, CarPart, CarParts
from selfdrive.car.fw_query_definitions import FwQueryConfig, Request, p16

Ecu = car.CarParams.Ecu


class ChryslerFlags(IntFlag):
  HIGHER_MIN_STEERING_SPEED = 1


class CAR:
  # Chrysler
  PACIFICA_2017_HYBRID = "CHRYSLER PACIFICA HYBRID 2017"
  PACIFICA_2018_HYBRID = "CHRYSLER PACIFICA HYBRID 2018"
  PACIFICA_2019_HYBRID = "CHRYSLER PACIFICA HYBRID 2019"
  PACIFICA_2018 = "CHRYSLER PACIFICA 2018"
  PACIFICA_2020 = "CHRYSLER PACIFICA 2020"

  # Jeep
  JEEP_CHEROKEE = "JEEP GRAND CHEROKEE V6 2018"   # includes 2017 Trailhawk
  JEEP_CHEROKEE_2019 = "JEEP GRAND CHEROKEE 2019" # includes 2020 Trailhawk

  # Ram
  RAM_1500 = "RAM 1500 5TH GEN"
  RAM_HD = "RAM HD 5TH GEN"


class CarControllerParams:
  def __init__(self, CP):
    self.STEER_STEP = 2  # 50 Hz
    self.STEER_ERROR_MAX = 80
    if CP.carFingerprint in RAM_HD:
      self.STEER_DELTA_UP = 14
      self.STEER_DELTA_DOWN = 14
      self.STEER_MAX = 361  # higher than this faults the EPS
    elif CP.carFingerprint in RAM_DT:
      self.STEER_DELTA_UP = 6
      self.STEER_DELTA_DOWN = 6
      self.STEER_MAX = 261  # EPS allows more, up to 350?
    else:
      self.STEER_DELTA_UP = 3
      self.STEER_DELTA_DOWN = 3
      self.STEER_MAX = 261  # higher than this faults the EPS


STEER_THRESHOLD = 120

RAM_DT = {CAR.RAM_1500, }
RAM_HD = {CAR.RAM_HD, }
RAM_CARS = RAM_DT | RAM_HD


@dataclass
class ChryslerCarInfo(CarInfo):
  package: str = "Adaptive Cruise Control (ACC)"
  car_parts: CarParts = CarParts.default([CarPart.fca_connector])


CAR_INFO: Dict[str, Optional[Union[ChryslerCarInfo, List[ChryslerCarInfo]]]] = {
  CAR.PACIFICA_2017_HYBRID: ChryslerCarInfo("Chrysler Pacifica Hybrid 2017-18"),
  CAR.PACIFICA_2018_HYBRID: None,  # same platforms
  CAR.PACIFICA_2019_HYBRID: ChryslerCarInfo("Chrysler Pacifica Hybrid 2019-23"),
  CAR.PACIFICA_2018: ChryslerCarInfo("Chrysler Pacifica 2017-18"),
  CAR.PACIFICA_2020: [
    ChryslerCarInfo("Chrysler Pacifica 2019-20"),
    ChryslerCarInfo("Chrysler Pacifica 2021", package="All"),
  ],
  CAR.JEEP_CHEROKEE: ChryslerCarInfo("Jeep Grand Cherokee 2016-18", video_link="https://www.youtube.com/watch?v=eLR9o2JkuRk"),
  CAR.JEEP_CHEROKEE_2019: ChryslerCarInfo("Jeep Grand Cherokee 2019-21", video_link="https://www.youtube.com/watch?v=jBe4lWnRSu4"),
  CAR.RAM_1500: ChryslerCarInfo("Ram 1500 2019-23", car_parts=CarParts.default([CarPart.ram_connector])),
  CAR.RAM_HD: [
    ChryslerCarInfo("Ram 2500 2020-22", car_parts=CarParts.default([CarPart.ram_connector])),
    ChryslerCarInfo("Ram 3500 2019-22", car_parts=CarParts.default([CarPart.ram_connector])),
  ],
}

# Unique CAN messages:
# Only the hybrids have 270: 8
# Only the gas have 55: 8, 416: 7
# For 564, All 2017 have length 4, whereas 2018-19 have length 8.
# For 924, Pacifica 2017 has length 3, whereas all 2018-19 have length 8.
# For 560, All 2019 have length 8, whereas all 2017-18 have length 4.

# Jeep Grand Cherokee unique messages:
# 2017 Trailhawk: 618: 8
# For 924, Trailhawk 2017 has length 3, whereas 2018 V6 has length 8.

FINGERPRINTS = {
  CAR.PACIFICA_2017_HYBRID: [{
    168: 8, 257: 5, 258: 8, 264: 8, 268: 8, 270: 8, 274: 2, 280: 8, 284: 8, 288: 7, 290: 6, 291: 8, 292: 8, 294: 8, 300: 8, 308: 8, 320: 8, 324: 8, 331: 8, 332: 8, 344: 8, 368: 8, 376: 3, 384: 8, 388: 4, 448: 6, 456: 4, 464: 8, 469: 8, 480: 8, 500: 8, 501: 8, 512: 8, 514: 8, 515: 7, 516: 7, 517: 7, 518: 7, 520: 8, 528: 8, 532: 8, 542: 8, 544: 8, 557: 8, 559: 8, 560: 4, 564: 4, 571: 3, 584: 8, 608: 8, 624: 8, 625: 8, 632: 8, 639: 8, 653: 8, 654: 8, 655: 8, 658: 6, 660: 8, 669: 3, 671: 8, 672: 8, 678: 8, 680: 8, 701: 8, 704: 8, 705: 8, 706: 8, 709: 8, 710: 8, 719: 8, 720: 6, 729: 5, 736: 8, 737: 8, 746: 5, 760: 8, 764: 8, 766: 8, 770: 8, 773: 8, 779: 8, 782: 8, 784: 8, 788: 3, 792: 8, 799: 8, 800: 8, 804: 8, 808: 8, 816: 8, 817: 8, 820: 8, 825: 2, 826: 8, 832: 8, 838: 2, 840: 8, 848: 8, 853: 8, 856: 4, 860: 6, 863: 8, 878: 8, 882: 8, 897: 8, 908: 8, 924: 3, 926: 3, 929: 8, 937: 8, 938: 8, 939: 8, 940: 8, 941: 8, 942: 8, 943: 8, 947: 8, 948: 8, 956: 8, 958: 8, 959: 8, 969: 4, 974: 5, 979: 8, 980: 8, 981: 8, 982: 8, 983: 8, 984: 8, 992: 8, 993: 7, 995: 8, 996: 8, 1000: 8, 1001: 8, 1002: 8, 1003: 8, 1008: 8, 1009: 8, 1010: 8, 1011: 8, 1012: 8, 1013: 8, 1014: 8, 1015: 8, 1024: 8, 1025: 8, 1026: 8, 1031: 8, 1033: 8, 1050: 8, 1059: 8, 1082: 8, 1083: 8, 1098: 8, 1100: 8, 1216: 8, 1218: 8, 1220: 8, 1225: 8, 1235: 8, 1242: 8, 1246: 8, 1250: 8, 1284: 8, 1537: 8, 1538: 8, 1562: 8, 1568: 8, 1856: 8, 1858: 8, 1860: 8, 1865: 8, 1875: 8, 1882: 8, 1886: 8, 1890: 8, 1892: 8, 2016: 8, 2024: 8
  }],
  CAR.PACIFICA_2018: [{
    55: 8, 257: 5, 258: 8, 264: 8, 268: 8, 274: 2, 280: 8, 284: 8, 288: 7, 290: 6, 292: 8, 294: 8, 300: 8, 308: 8, 320: 8, 324: 8, 331: 8, 332: 8, 344: 8, 368: 8, 376: 3, 384: 8, 388: 4, 416: 7, 448: 6, 456: 4, 464: 8, 469: 8, 480: 8, 500: 8, 501: 8, 512: 8, 514: 8, 516: 7, 517: 7, 520: 8, 524: 8, 526: 6, 528: 8, 532: 8, 542: 8, 544: 8, 557: 8, 559: 8, 560: 4, 564: 8, 571: 3, 579: 8, 584: 8, 608: 8, 624: 8, 625: 8, 632: 8, 639: 8, 656: 4, 658: 6, 660: 8, 669: 3, 671: 8, 672: 8, 678: 8, 680: 8, 705: 8, 706: 8, 709: 8, 710: 8, 719: 8, 720: 6, 729: 5, 736: 8, 746: 5, 752: 2, 760: 8, 764: 8, 766: 8, 770: 8, 773: 8, 779: 8, 784: 8, 792: 8, 799: 8, 800: 8, 804: 8, 808: 8, 816: 8, 817: 8, 820: 8, 825: 2, 826: 8, 832: 8, 838: 2, 848: 8, 853: 8, 856: 4, 860: 6, 863: 8, 882: 8, 897: 8, 924: 8, 926: 3, 937: 8, 947: 8, 948: 8, 969: 4, 974: 5, 979: 8, 980: 8, 981: 8, 982: 8, 983: 8, 984: 8, 992: 8, 993: 7, 995: 8, 996: 8, 1000: 8, 1001: 8, 1002: 8, 1003: 8, 1008: 8, 1009: 8, 1010: 8, 1011: 8, 1012: 8, 1013: 8, 1014: 8, 1015: 8, 1024: 8, 1025: 8, 1026: 8, 1031: 8, 1033: 8, 1050: 8, 1059: 8, 1098: 8, 1100: 8, 1537: 8, 1538: 8, 1562: 8
  },
  {
    55: 8, 58: 6, 257: 5, 258: 8, 264: 8, 268: 8, 274: 2, 280: 8, 284: 8, 288: 7, 290: 6, 292: 8, 294: 8, 300: 8, 308: 8, 320: 8, 324: 8, 331: 8, 332: 8, 344: 8, 368: 8, 376: 3, 384: 8, 388: 4, 416: 7, 448: 6, 456: 4, 464: 8, 469: 8, 480: 8, 500: 8, 501: 8, 512: 8, 514: 8, 516: 7, 517: 7, 520: 8, 524: 8, 526: 6, 528: 8, 532: 8, 542: 8, 544: 8, 557: 8, 559: 8, 560: 4, 564: 4, 571: 3, 584: 8, 608: 8, 624: 8, 625: 8, 632: 8, 639: 8, 656: 4, 658: 6, 660: 8, 669: 3, 671: 8, 672: 8, 678: 8, 680: 8, 705: 8, 706: 8, 709: 8, 710: 8, 719: 8, 720: 6, 729: 5, 736: 8, 746: 5, 752: 2, 760: 8, 764: 8, 766: 8, 770: 8, 773: 8, 779: 8, 784: 8, 792: 8, 799: 8, 800: 8, 804: 8, 808: 8, 816: 8, 817: 8, 820: 8, 825: 2, 826: 8, 832: 8, 838: 2, 848: 8, 853: 8, 856: 4, 860: 6, 863: 8, 882: 8, 897: 8, 924: 3, 926: 3, 937: 8, 947: 8, 948: 8, 956: 8, 969: 4, 974: 5, 979: 8, 980: 8, 981: 8, 982: 8, 983: 8, 984: 8, 992: 8, 993: 7, 995: 8, 996: 8, 1000: 8, 1001: 8, 1002: 8, 1003: 8, 1008: 8, 1009: 8, 1010: 8, 1011: 8, 1012: 8, 1013: 8, 1014: 8, 1015: 8, 1024: 8, 1025: 8, 1026: 8, 1031: 8, 1033: 8, 1050: 8, 1059: 8, 1098: 8, 1100: 8, 1537: 8, 1538: 8, 1562: 8
  }],
  CAR.PACIFICA_2020: [{
    55: 8, 179: 8, 181: 8, 257: 5, 258: 8, 264: 8, 268: 8, 274: 2, 280: 8, 284: 8, 288: 7, 290: 6, 292: 8, 294: 8, 300: 8, 308: 8, 320: 8, 324: 8, 331: 8, 332: 8, 344: 8, 352: 8, 362: 8, 368: 8, 376: 3, 384: 8, 388: 4, 416: 7, 448: 6, 456: 4, 464: 8, 469: 8, 480: 8, 500: 8, 501: 8, 512: 8, 514: 8, 516: 7, 517: 7, 520: 8, 524: 8, 526: 6, 528: 8, 532: 8, 536: 8, 542: 8, 544: 8, 557: 8, 559: 8, 560: 8, 564: 8, 571: 3, 579: 8, 584: 8, 608: 8, 624: 8, 625: 8, 632: 8, 639: 8, 650: 8, 656: 4, 658: 6, 660: 8, 669: 3, 671: 8, 672: 8, 676: 8, 678: 8, 680: 8, 683: 8, 703: 8, 705: 8, 706: 8, 709: 8, 710: 8, 711: 8, 719: 8, 720: 6, 729: 5, 736: 8, 746: 5, 752: 2, 754: 8, 760: 8, 764: 8, 766: 8, 770: 8, 773: 8, 776: 8, 779: 8, 782: 8, 784: 8, 792: 8, 793: 8, 794: 8, 795: 8, 799: 8, 800: 8, 801: 8, 802: 8, 803: 8, 804: 8, 808: 8, 816: 8, 817: 8, 820: 8, 825: 2, 826: 8, 832: 8, 838: 2, 847: 1, 848: 8, 853: 8, 856: 4, 860: 6, 863: 8, 882: 8, 886: 8, 897: 8, 906: 8, 924: 8, 926: 3, 937: 8, 938: 8, 939: 8, 940: 8, 941: 8, 942: 8, 943: 8, 947: 8, 948: 8, 962: 8, 969: 4, 973: 8, 974: 5, 979: 8, 980: 8, 981: 8, 982: 8, 983: 8, 984: 8, 992: 8, 993: 7, 995: 8, 996: 8, 1000: 8, 1001: 8, 1002: 8, 1003: 8, 1008: 8, 1009: 8, 1010: 8, 1011: 8, 1012: 8, 1013: 8, 1014: 8, 1015: 8, 1024: 8, 1025: 8, 1026: 8, 1031: 8, 1033: 8, 1050: 8, 1059: 8, 1098: 8, 1100: 8, 1216: 8, 1218: 8, 1220: 8, 1223: 7, 1225: 8, 1227: 8, 1235: 8, 1242: 8, 1246: 8, 1250: 8, 1251: 8, 1252: 8, 1284: 8, 1543: 8, 1568: 8, 1570: 8, 1856: 8, 1858: 8, 1860: 8, 1863: 8, 1865: 8, 1867: 8, 1875: 8, 1882: 8, 1886: 8, 1890: 8, 1891: 8, 1892: 8, 1898: 8, 2015: 8, 2016: 8, 2017:8, 2024: 8, 2025: 8
  }],
  CAR.PACIFICA_2018_HYBRID: [{
    68: 8, 168: 8, 257: 5, 258: 8, 264: 8, 268: 8, 270: 8, 274: 2, 280: 8, 284: 8, 288: 7, 290: 6, 291: 8, 292: 8, 294: 8, 300: 8, 308: 8, 320: 8, 324: 8, 331: 8, 332: 8, 344: 8, 368: 8, 376: 3, 384: 8, 388: 4, 448: 6, 456: 4, 464: 8, 469: 8, 480: 8, 500: 8, 501: 8, 512: 8, 514: 8, 520: 8, 528: 8, 532: 8, 544: 8, 557: 8, 559: 8, 560: 4, 564: 8, 571: 3, 579: 8, 584: 8, 608: 8, 624: 8, 625: 8, 632: 8, 639: 8, 653: 8, 654: 8, 655: 8, 658: 6, 660: 8, 669: 3, 671: 8, 672: 8, 680: 8, 701: 8, 704: 8, 705: 8, 706: 8, 709: 8, 710: 8, 719: 8, 720: 6, 736: 8, 737: 8, 746: 5, 760: 8, 764: 8, 766: 8, 770: 8, 773: 8, 779: 8, 782: 8, 784: 8, 792: 8, 799: 8, 800: 8, 804: 8, 808: 8, 816: 8, 817: 8, 820: 8, 825: 2, 826: 8, 832: 8, 838: 2, 848: 8, 853: 8, 856: 4, 860: 6, 863: 8, 878: 8, 882: 8, 897: 8, 908: 8, 924: 8, 926: 3, 929: 8, 937: 8, 938: 8, 939: 8, 940: 8, 941: 8, 942: 8, 943: 8, 947: 8, 948: 8, 958: 8, 959: 8, 969: 4, 974: 5, 979: 8, 980: 8, 981: 8, 982: 8, 983: 8, 984: 8, 992: 8, 993: 7, 995: 8, 996: 8, 1000: 8, 1001: 8, 1002: 8, 1003: 8, 1008: 8, 1009: 8, 1010: 8, 1011: 8, 1012: 8, 1013: 8, 1014: 8, 1015: 8, 1024: 8, 1025: 8, 1026: 8, 1031: 8, 1033: 8, 1050: 8, 1059: 8, 1082: 8, 1083: 8, 1098: 8, 1100: 8
  },
    # based on 9ae7821dc4e92455|2019-07-01--16-42-55
  {
    168: 8, 257: 5, 258: 8, 264: 8, 268: 8, 270: 8, 274: 2, 280: 8, 284: 8, 288: 7, 290: 6, 291: 8, 292: 8, 294: 8, 300: 8, 308: 8, 320: 8, 324: 8, 331: 8, 332: 8, 344: 8, 368: 8, 376: 3, 384: 8, 388: 4, 448: 6, 456: 4, 464: 8, 469: 8, 480: 8, 500: 8, 501: 8, 512: 8, 514: 8, 515: 7, 516: 7, 517: 7, 518: 7, 520: 8, 528: 8, 532: 8, 542: 8, 544: 8, 557: 8, 559: 8, 560: 4, 564: 8, 571: 3, 579: 8, 584: 8, 608: 8, 624: 8, 625: 8, 632: 8, 639: 8, 653: 8, 654: 8, 655: 8, 658: 6, 660: 8, 669: 3, 671: 8, 672: 8, 678: 8, 680: 8, 701: 8, 704: 8, 705: 8, 706: 8, 709: 8, 710: 8, 719: 8, 720: 6, 729: 5, 736: 8, 737: 8, 746: 5, 760: 8, 764: 8, 766: 8, 770: 8, 773: 8, 779: 8, 782: 8, 784: 8, 792: 8, 799: 8, 800: 8, 804: 8, 808: 8, 816: 8, 817: 8, 820: 8, 825: 2, 826: 8, 832: 8, 838: 2, 848: 8, 853: 8, 856: 4, 860: 6, 863: 8, 878: 8, 882: 8, 897: 8, 908: 8, 924: 8, 926: 3, 929: 8, 937: 8, 938: 8, 939: 8, 940: 8, 941: 8, 942: 8, 943: 8, 947: 8, 948: 8, 958: 8, 959: 8, 969: 4, 974: 5, 979: 8, 980: 8, 981: 8, 982: 8, 983: 8, 984: 8, 992: 8, 993: 7, 995: 8, 996: 8, 1000: 8, 1001: 8, 1002: 8, 1003: 8, 1008: 8, 1009: 8, 1010: 8, 1011: 8, 1012: 8, 1013: 8, 1014: 8, 1015: 8, 1024: 8, 1025: 8, 1026: 8, 1031: 8, 1033: 8, 1050: 8, 1059: 8, 1082: 8, 1083: 8, 1098: 8, 1100: 8, 1216: 8, 1218: 8, 1220: 8, 1225: 8, 1235: 8, 1242: 8, 1246: 8, 1250: 8, 1251: 8, 1252: 8, 1258: 8, 1259: 8, 1260: 8, 1262: 8, 1284: 8, 1537: 8, 1538: 8, 1562: 8, 1568: 8, 1856: 8, 1858: 8, 1860: 8, 1865: 8, 1875: 8, 1882: 8, 1886: 8, 1890: 8, 1891: 8, 1892: 8, 1898: 8, 1899: 8, 1900: 8, 1902: 8, 2016: 8, 2018: 8, 2019: 8, 2020: 8, 2023: 8, 2024: 8, 2026: 8, 2027: 8, 2028: 8, 2031: 8
  }],
  CAR.PACIFICA_2019_HYBRID: [{
    168: 8, 257: 5, 258: 8, 264: 8, 268: 8, 270: 8, 274: 2, 280: 8, 284: 8, 288: 7, 290: 6, 291: 8, 292: 8, 294: 8, 300: 8, 308: 8, 320: 8, 324: 8, 331: 8, 332: 8, 344: 8, 368: 8, 376: 3, 384: 8, 388: 4, 448: 6, 456: 4, 464: 8, 469: 8, 480: 8, 500: 8, 501: 8, 512: 8, 514: 8, 515: 7, 516: 7, 517: 7, 518: 7, 520: 8, 528: 8, 532: 8, 542: 8, 544: 8, 557: 8, 559: 8, 560: 8, 564: 8, 571: 3, 579: 8, 584: 8, 608: 8, 624: 8, 625: 8, 632: 8, 639: 8, 653: 8, 654: 8, 655: 8, 658: 6, 660: 8, 669: 3, 671: 8, 672: 8, 680: 8, 701: 8, 703: 8, 704: 8, 705: 8, 706: 8, 709: 8, 710: 8, 719: 8, 720: 6, 736: 8, 737: 8, 746: 5, 752: 2, 754: 8, 760: 8, 764: 8, 766: 8, 770: 8, 773: 8, 779: 8, 782: 8, 784: 8, 792: 8, 799: 8, 800: 8, 804: 8, 816: 8, 817: 8, 820: 8, 825: 2, 826: 8, 832: 8, 838: 2, 848: 8, 853: 8, 856: 4, 860: 6, 863: 8, 878: 8, 882: 8, 897: 8, 906: 8, 908: 8, 924: 8, 926: 3, 929: 8, 937: 8, 938: 8, 939: 8, 940: 8, 941: 8, 942: 8, 943: 8, 947: 8, 948: 8, 958: 8, 959: 8, 962: 8, 969: 4, 973: 8, 974: 5, 979: 8, 980: 8, 981: 8, 982: 8, 983: 8, 984: 8, 992: 8, 993: 7, 995: 8, 996: 8, 1000: 8, 1001: 8, 1002: 8, 1003: 8, 1008: 8, 1009: 8, 1010: 8, 1011: 8, 1012: 8, 1013: 8, 1014: 8, 1015: 8, 1024: 8, 1025: 8, 1026: 8, 1031: 8, 1033: 8, 1050: 8, 1059: 8, 1082: 8, 1083: 8, 1098: 8, 1100: 8, 1538: 8
  },
    # Based on 0607d2516fc2148f|2019-02-13--23-03-16
  {
    168: 8, 257: 5, 258: 8, 264: 8, 268: 8, 270: 8, 274: 2, 280: 8, 284: 8, 288: 7, 290: 6, 291: 8, 292: 8, 294: 8, 300: 8, 308: 8, 320: 8, 324: 8, 331: 8, 332: 8, 344: 8, 368: 8, 376: 3, 384: 8, 388: 4, 448: 6, 456: 4, 464: 8, 469: 8, 480: 8, 500: 8, 501: 8, 512: 8, 514: 8, 520: 8, 528: 8, 532: 8, 544: 8, 557: 8, 559: 8, 560: 8, 564: 8, 571: 3, 579: 8, 584: 8, 608: 8, 624: 8, 625: 8, 632: 8, 639: 8, 653: 8, 654: 8, 655: 8, 658: 6, 660: 8, 669: 3, 671: 8, 672: 8, 678: 8, 680: 8, 701: 8, 703: 8, 704: 8, 705: 8, 706: 8, 709: 8, 710: 8, 719: 8, 720: 6, 729: 5, 736: 8, 737: 8, 746: 5, 752: 2, 754: 8, 760: 8, 764: 8, 766: 8, 770: 8, 773: 8, 779: 8, 782: 8, 784: 8, 792: 8, 799: 8, 800: 8, 804: 8, 816: 8, 817: 8, 820: 8, 825: 2, 826: 8, 832: 8, 838: 2, 848: 8, 853: 8, 856: 4, 860: 6, 863: 8, 878: 8, 882: 8, 897: 8, 906: 8, 908: 8, 924: 8, 926: 3, 929: 8, 937: 8, 938: 8, 939: 8, 940: 8, 941: 8, 942: 8, 943: 8, 947: 8, 948: 8, 958: 8, 959: 8, 962: 8, 969: 4, 973: 8, 974: 5, 979: 8, 980: 8, 981: 8, 982: 8, 983: 8, 984: 8, 992: 8, 993: 7, 995: 8, 996: 8, 1000: 8, 1001: 8, 1002: 8, 1003: 8, 1008: 8, 1009: 8, 1010: 8, 1011: 8, 1012: 8, 1013: 8, 1014: 8, 1015: 8, 1024: 8, 1025: 8, 1026: 8, 1031: 8, 1033: 8, 1050: 8, 1059: 8, 1082: 8, 1083: 8, 1098: 8, 1100: 8, 1537: 8
  },
    # Based on 3c7ce223e3571b54|2019-05-11--20-16-14
  {
    168: 8, 257: 5, 258: 8, 264: 8, 268: 8, 270: 8, 274: 2, 280: 8, 284: 8, 288: 7, 290: 6, 291: 8, 292: 8, 294: 8, 300: 8, 308: 8, 320: 8, 324: 8, 331: 8, 332: 8, 344: 8, 368: 8, 376: 3, 384: 8, 388: 4, 448: 6, 456: 4, 464: 8, 469: 8, 480: 8, 500: 8, 501: 8, 512: 8, 514: 8, 520: 8, 528: 8, 532: 8, 544: 8, 557: 8, 559: 8, 560: 8, 564: 8, 571: 3, 579: 8, 584: 8, 608: 8, 624: 8, 625: 8, 632: 8, 639: 8, 653: 8, 654: 8, 655: 8, 658: 6, 660: 8, 669: 3, 671: 8, 672: 8, 678: 8, 680: 8, 701: 8, 703: 8, 704: 8, 705: 8, 706: 8, 709: 8, 710: 8, 719: 8, 720: 6, 729: 5, 736: 8, 737: 8, 746: 5, 752: 2, 754: 8, 760: 8, 764: 8, 766: 8, 770: 8, 773: 8, 779: 8, 782: 8, 784: 8, 792: 8, 799: 8, 800: 8, 804: 8, 808: 8, 816: 8, 817: 8, 820: 8, 825: 2, 826: 8, 832: 8, 838: 2, 848: 8, 853: 8, 856: 4, 860: 6, 863: 8, 878: 8, 882: 8, 897: 8, 906: 8, 908: 8, 924: 8, 926: 3, 929: 8, 937: 8, 938: 8, 939: 8, 940: 8, 941: 8, 942: 8, 943: 8, 947: 8, 948: 8, 958: 8, 959: 8, 962: 8, 969: 4, 973: 8, 974: 5, 979: 8, 980: 8, 981: 8, 982: 8, 983: 8, 984: 8, 992: 8, 993: 7, 995: 8, 996: 8, 1000: 8, 1001: 8, 1002: 8, 1003: 8, 1008: 8, 1009: 8, 1010: 8, 1011: 8, 1012: 8, 1013: 8, 1014: 8, 1015: 8, 1024: 8, 1025: 8, 1026: 8, 1031: 8, 1033: 8, 1050: 8, 1059: 8, 1082: 8, 1083: 8, 1098: 8, 1100: 8, 1562: 8, 1570: 8
  },
    # Based on "8190c7275a24557b|2020-02-24--09-57-23"
  {
    168: 8, 257: 5, 258: 8, 264: 8, 268: 8, 270: 8, 274: 2, 280: 8, 284: 8, 288: 7, 290: 6, 291: 8, 292: 8, 294: 8, 300: 8, 308: 8, 320: 8, 324: 8, 331: 8, 332: 8, 344: 8, 368: 8, 376: 3, 384: 8, 388: 4, 448: 6, 456: 4, 464: 8, 469: 8, 480: 8, 500: 8, 501: 8, 512: 8, 514: 8, 515: 7, 516: 7, 517: 7, 518: 7, 520: 8, 524: 8, 526: 6, 528: 8, 532: 8, 542: 8, 544: 8, 557: 8, 559: 8, 560: 8, 564: 8, 571: 3, 579: 8, 584: 8, 608: 8, 624: 8, 625: 8, 632: 8, 639: 8, 640: 1, 650: 8, 653: 8, 654: 8, 655: 8, 656: 4, 658: 6, 660: 8, 669: 3, 671: 8, 672: 8, 678: 8, 680: 8, 683: 8, 701: 8, 703: 8, 704: 8, 705: 8, 706: 8, 709: 8, 710: 8, 711: 8, 719: 8, 720: 6, 729: 5, 736: 8, 737: 8, 738: 8, 746: 5, 752: 2, 754: 8, 760: 8, 764: 8, 766: 8, 770: 8, 773: 8, 779: 8, 782: 8, 784: 8, 792: 8, 793: 8, 794: 8, 795: 8, 796: 8, 797: 8, 798: 8, 799: 8, 800: 8, 801: 8, 802: 8, 803: 8, 804: 8, 805: 8, 807: 8, 808: 8, 816: 8, 817: 8, 820: 8, 825: 2, 826: 8, 832: 8, 838: 2, 847: 1, 848: 8, 853: 8, 856: 4, 860: 6, 863: 8, 878: 8, 882: 8, 886: 8, 897: 8, 906: 8, 908: 8, 924: 8, 926: 3, 929: 8, 937: 8, 938: 8, 939: 8, 940: 8, 941: 8, 942: 8, 943: 8, 947: 8, 948: 8, 958: 8, 959: 8, 962: 8, 969: 4, 973: 8, 974: 5, 979: 8, 980: 8, 981: 8, 982: 8, 983: 8, 984: 8, 992: 8, 993: 7, 995: 8, 996: 8, 1000: 8, 1001: 8, 1002: 8, 1003: 8, 1008: 8, 1009: 8, 1010: 8, 1011: 8, 1012: 8, 1013: 8, 1014: 8, 1015: 8, 1024: 8, 1025: 8, 1026: 8, 1031: 8, 1033: 8, 1050: 8, 1059: 8, 1082: 8, 1083: 8, 1098: 8, 1100: 8, 1216: 8, 1218: 8, 1220: 8, 1225: 8, 1235: 8, 1242: 8, 1246: 8, 1250: 8, 1251: 8, 1252: 8, 1258: 8, 1259: 8, 1260: 8, 1262: 8, 1284: 8, 1536: 8, 1568: 8, 1570: 8, 1856: 8, 1858: 8, 1860: 8, 1863: 8, 1865: 8, 1875: 8, 1882: 8, 1886: 8, 1890: 8, 1891: 8, 1892: 8, 1898: 8, 1899: 8, 1900: 8, 1902: 8, 2015: 8, 2016: 8, 2017: 8, 2018: 8, 2019: 8, 2020: 8, 2023: 8, 2024: 8, 2026: 8, 2027: 8, 2028: 8, 2031: 8
  },
  {
    168: 8, 257: 5, 258: 8, 264: 8, 268: 8, 270: 8, 274: 2, 280: 8, 284: 8, 288: 7, 290: 6, 291: 8, 292: 8, 294: 8, 300: 8, 308: 8, 320: 8, 324: 8, 331: 8, 332: 8, 344: 8, 368: 8, 376: 3, 384: 8, 388: 4, 448: 6, 450: 8, 456: 4, 464: 8, 469: 8, 480: 8, 500: 8, 501: 8, 512: 8, 514: 8, 515: 7, 516: 7, 517: 7, 518: 7, 520: 8, 524: 8, 526: 6, 528: 8, 532: 8, 542: 8, 544: 8, 557: 8, 559: 8, 560: 8, 564: 8, 571: 3, 579: 8, 584: 8, 608: 8, 624: 8, 625: 8, 632: 8, 639: 8, 650: 8, 653: 8, 654: 8, 655: 8, 656: 4, 658: 6, 660: 8, 669: 3, 671: 8, 672: 8, 678: 8, 680: 8, 683: 8, 701: 8, 703: 8, 704: 8, 705: 8, 706: 8, 709: 8, 710: 8, 711: 8, 719: 8, 720: 6, 729: 5, 736: 8, 737: 8, 738: 8, 746: 5, 752: 2, 754: 8, 760: 8, 764: 8, 766: 8, 770: 8, 773: 8, 779: 8, 782: 8, 784: 8, 792: 8, 793: 8, 794: 8, 795: 8, 796: 8, 797: 8, 798: 8, 799: 8, 800: 8, 801: 8, 802: 8, 803: 8, 804: 8, 805: 8, 807: 8, 808: 8, 816: 8, 817: 8, 820: 8, 825: 2, 826: 8, 832: 8, 838: 2, 848: 8, 853: 8, 856: 4, 860: 6, 863: 8, 878: 8, 882: 8, 886: 8, 897: 8, 906: 8, 908: 8, 924: 8, 926: 3, 937: 8, 938: 8, 939: 8, 940: 8, 941: 8, 942: 8, 943: 8, 947: 8, 948: 8, 958: 8, 959: 8, 962: 8, 969: 4, 973: 8, 974: 5, 979: 8, 980: 8, 981: 8, 982: 8, 983: 8, 984: 8, 992: 8, 993: 7, 995: 8, 996: 8, 1000: 8, 1001: 8, 1002: 8, 1003: 8, 1008: 8, 1009: 8, 1010: 8, 1011: 8, 1012: 8, 1013: 8, 1014: 8, 1015: 8, 1024: 8, 1025: 8, 1026: 8, 1031: 8, 1033: 8, 1050: 8, 1059: 8, 1082: 8, 1083: 8, 1098: 8, 1100: 8, 1216: 8, 1218: 8, 1220: 8, 1225: 8, 1235: 8, 1242: 8, 1246: 8, 1250: 8, 1251: 8, 1252: 8, 1284: 8, 1568: 8, 1856: 8, 1858: 8, 1860: 8, 1863: 8, 1865: 8, 1875: 8, 1882: 8, 1886: 8, 1890: 8, 1891: 8, 1892: 8, 2018: 8, 2020: 8, 2026: 8, 2028: 8
  }],
  CAR.JEEP_CHEROKEE: [{
    55: 8, 168: 8, 181: 8, 256: 4, 257: 5, 258: 8, 264: 8, 268: 8, 272: 6, 273: 6, 274: 2, 280: 8, 284: 8, 288: 7, 290: 6, 292: 8, 300: 8, 308: 8, 320: 8, 324: 8, 331: 8, 332: 8, 344: 8, 352: 8, 362: 8, 368: 8, 376: 3, 384: 8, 388: 4, 416: 7, 448: 6, 456: 4, 464: 8, 500: 8, 501: 8, 512: 8, 514: 8, 520: 8, 532: 8, 544: 8, 557: 8, 559: 8, 560: 4, 564: 4, 571: 3, 579: 8, 584: 8, 608: 8, 618: 8, 624: 8, 625: 8, 632: 8, 639: 8, 656: 4, 658: 6, 660: 8, 671: 8, 672: 8, 676: 8, 678: 8, 680: 8, 683: 8, 684: 8, 703: 8, 705: 8, 706: 8, 709: 8, 710: 8, 719: 8, 720: 6, 729: 5, 736: 8, 737: 8, 738: 8, 746: 5, 752: 2, 754: 8, 760: 8, 761: 8, 764: 8, 766: 8, 773: 8, 776: 8, 779: 8, 782: 8, 783: 8, 784: 8, 785: 8, 788: 3, 792: 8, 799: 8, 800: 8, 804: 8, 806: 2, 808: 8, 810: 8, 816: 8, 817: 8, 820: 8, 825: 2, 826: 8, 831: 6, 832: 8, 838: 2, 840: 8, 844: 5, 847: 1, 848: 8, 853: 8, 856: 4, 860: 6, 863: 8, 874: 2, 882: 8, 897: 8, 906: 8, 924: 8, 937: 8, 938: 8, 939: 8, 940: 8, 941: 8, 942: 8, 943: 8, 947: 8, 948: 8, 956: 8, 968: 8, 969: 4, 970: 8, 973: 8, 974: 5, 975: 8, 976: 8, 977: 4, 979: 8, 980: 8, 981: 8, 982: 8, 983: 8, 984: 8, 992: 8, 993: 7, 995: 8, 996: 8, 1000: 8, 1001: 8, 1002: 8, 1003: 8, 1008: 8, 1009: 8, 1010: 8, 1011: 8, 1012: 8, 1013: 8, 1014: 8, 1015: 8, 1024: 8, 1025: 8, 1026: 8, 1031: 8, 1033: 8, 1050: 8, 1059: 8, 1062: 8, 1098: 8, 1100: 8, 1543: 8, 1562: 8, 2015: 8, 2016: 8, 2017: 8, 2024: 8, 2025: 8
  },
    # Based on c88f65eeaee4003a|2022-08-04--15-37-16
  {
    257: 5, 258: 8, 264: 8, 268: 8, 274: 2, 280: 8, 284: 8, 288: 7, 290: 6, 292: 8, 300: 8, 308: 8, 320: 8, 324: 8, 331: 8, 332: 8, 344: 8, 352: 8, 362: 8, 368: 8, 376: 3, 384: 8, 388: 4, 416: 7, 448: 6, 456: 4, 464: 8, 500: 8, 501: 8, 512: 8, 514: 8, 520: 8, 532: 8, 544: 8, 557: 8, 559: 8, 560: 4, 564: 4, 571: 3, 584: 8, 608: 8, 624: 8, 625: 8, 632: 8, 639: 8, 658: 6, 660: 8, 671: 8, 672: 8, 678: 8, 680: 8, 684: 8, 703: 8, 705: 8, 706: 8, 709: 8, 710: 8, 719: 8, 720: 6, 729: 5, 736: 8, 737: 8, 746: 5, 752: 2, 760: 8, 761: 8, 764: 8, 766: 8, 773: 8, 776: 8, 779: 8, 783: 8, 784: 8, 792: 8, 799: 8, 800: 8, 804: 8, 806: 2, 810: 8, 816: 8, 817: 8, 820: 8, 825: 2, 826: 8, 831: 6, 832: 8, 838: 2, 844: 5, 848: 8, 853: 8, 856: 4, 860: 6, 863: 8, 882: 8, 897: 8, 924: 3, 937: 8, 947: 8, 948: 8, 969: 4, 974: 5, 977: 4, 979: 8, 980: 8, 981: 8, 982: 8, 983: 8, 984: 8, 992: 8, 993: 7, 995: 8, 996: 8, 1000: 8, 1001: 8, 1002: 8, 1003: 8, 1008: 8, 1009: 8, 1010: 8, 1011: 8, 1012: 8, 1013: 8, 1014: 8, 1015: 8, 1024: 8, 1025: 8, 1026: 8, 1031: 8, 1033: 8, 1050: 8, 1059: 8, 1062: 8, 1098: 8, 1100: 8, 1216: 8, 1218: 8, 1220: 8, 1223: 8, 1235: 8, 1242: 8, 1252: 8, 1792: 8, 1798: 8, 1799: 8, 1810: 8, 1813: 8, 1824: 8, 1825: 8, 1840: 8, 1856: 8, 1858: 8, 1859: 8, 1860: 8, 1862: 8, 1863: 8, 1872: 8, 1875: 8, 1879: 8, 1882: 8, 1888: 8, 1892: 8, 1927: 8, 1937: 8, 1953: 8, 1968: 8, 1988: 8, 2000: 8, 2001: 8, 2004: 8, 2015: 8, 2016: 8, 2017: 8, 2024: 8, 2025: 8
  }],
  CAR.JEEP_CHEROKEE_2019: [{
    # Jeep Grand Cherokee 2019, including most 2020 models
    55: 8, 168: 8, 179: 8, 181: 8, 256: 4, 257: 5, 258: 8, 264: 8, 268: 8, 272: 6, 273: 6, 274: 2, 280: 8, 284: 8, 288: 7, 290: 6, 292: 8, 300: 8, 308: 8, 320: 8, 324: 8, 331: 8, 332: 8, 341: 8, 344: 8, 352: 8, 362: 8, 368: 8, 376: 3, 384: 8, 388: 4, 416: 7, 448: 6, 456: 4, 464: 8, 500: 8, 501: 8, 512: 8, 514: 8, 520: 8, 530: 8, 532: 8, 544: 8, 557: 8, 559: 8, 560: 8, 564: 8, 571: 3, 579: 8, 584: 8, 608: 8, 618: 8, 624: 8, 625: 8, 632: 8, 639: 8, 640: 1, 656: 4, 658: 6, 660: 8, 671: 8, 672: 8, 676: 8, 678: 8, 680: 8, 683: 8, 684: 8, 703: 8, 705: 8, 706: 8, 709: 8, 710: 8, 719: 8, 720: 6, 729: 5, 736: 8, 737: 8, 738: 8, 746: 5, 752: 2, 754: 8, 760: 8, 761: 8, 764: 8, 766: 8, 773: 8, 776: 8, 779: 8, 782: 8, 783: 8, 784: 8, 785: 8, 792: 8, 799: 8, 800: 8, 804: 8, 806: 2, 808: 8, 810: 8, 816: 8, 817: 8, 820: 8, 825: 2, 826: 8, 831: 6, 832: 8, 838: 2, 840: 8, 844: 5, 847: 1, 848: 8, 853: 8, 856: 4, 860: 6, 863: 8, 874: 2, 882: 8, 897: 8, 906: 8, 924: 8, 937: 8, 938: 8, 939: 8, 940: 8, 941: 8, 942: 8, 943: 8, 947: 8, 948: 8, 960: 4, 968: 8, 969: 4, 970: 8, 973: 8, 974: 5, 976: 8, 977: 4, 979: 8, 980: 8, 981: 8, 982: 8, 983: 8, 984: 8, 992: 8, 993: 7, 995: 8, 996: 8, 1000: 8, 1001: 8, 1002: 8, 1003: 8, 1008: 8, 1009: 8, 1010: 8, 1011: 8, 1012: 8, 1013: 8, 1014: 8, 1015: 8, 1024: 8, 1025: 8, 1026: 8, 1031: 8, 1033: 8, 1050: 8, 1059: 8, 1062: 8, 1098: 8, 1100: 8, 1216: 8, 1218: 8, 1220: 8, 1223: 8, 1225: 8, 1227: 8, 1235: 8, 1242: 8, 1250: 8, 1251: 8, 1252: 8, 1254: 8, 1264: 8, 1284: 8, 1536: 8, 1537: 8, 1538: 8, 1543: 8, 1545: 8, 1562: 8, 1568: 8, 1570: 8, 1572: 8, 1593: 8, 1856: 8, 1858: 8, 1860: 8, 1863: 8, 1865: 8, 1867: 8, 1875: 8, 1882: 8, 1890: 8, 1891: 8, 1892: 8, 1894: 8, 1896: 8, 1904: 8, 2015: 8, 2016: 8, 2017: 8, 2024: 8, 2025: 8
  }],
}

CHRYSLER_VERSION_REQUEST = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER]) + \
  p16(0xf132)
CHRYSLER_VERSION_RESPONSE = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER + 0x40]) + \
  p16(0xf132)

CHRYSLER_SOFTWARE_VERSION_REQUEST = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER]) + \
  p16(uds.DATA_IDENTIFIER_TYPE.SYSTEM_SUPPLIER_ECU_SOFTWARE_NUMBER)
CHRYSLER_SOFTWARE_VERSION_RESPONSE = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER + 0x40]) + \
  p16(uds.DATA_IDENTIFIER_TYPE.SYSTEM_SUPPLIER_ECU_SOFTWARE_NUMBER)

CHRYSLER_RX_OFFSET = -0x280

FW_QUERY_CONFIG = FwQueryConfig(
  requests=[
    Request(
      [CHRYSLER_VERSION_REQUEST],
      [CHRYSLER_VERSION_RESPONSE],
      whitelist_ecus=[Ecu.abs, Ecu.eps, Ecu.srs, Ecu.fwdRadar, Ecu.fwdCamera, Ecu.combinationMeter],
      rx_offset=CHRYSLER_RX_OFFSET,
      bus=0,
    ),
    Request(
      [CHRYSLER_VERSION_REQUEST],
      [CHRYSLER_VERSION_RESPONSE],
      whitelist_ecus=[Ecu.abs, Ecu.hcp, Ecu.engine, Ecu.transmission],
      bus=0,
    ),
    Request(
      [CHRYSLER_SOFTWARE_VERSION_REQUEST],
      [CHRYSLER_SOFTWARE_VERSION_RESPONSE],
      whitelist_ecus=[Ecu.engine, Ecu.transmission],
      bus=0,
    ),
  ],
  extra_ecus=[
    (Ecu.hcp, 0x7e2, None),  # manages transmission on hybrids
    (Ecu.abs, 0x7e4, None),  # alt address for abs on hybrids
  ],
)

FW_VERSIONS = {
  CAR.JEEP_CHEROKEE_2019: {
    (Ecu.combinationMeter, 0x742, None): [
      b'68402971AD',
    ],
    (Ecu.srs, 0x744, None): [
      b'68355363AB',
    ],
    (Ecu.abs, 0x747, None): [
      b'68408639AD',
    ],
    (Ecu.fwdRadar, 0x753, None): [
      b'68456722AC',
    ],
    (Ecu.eps, 0x75A, None): [
      b'68453431AA',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'05035674AB ',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'05035707AA',
    ],
  },

  CAR.RAM_1500: {
    (Ecu.combinationMeter, 0x742, None): [
      b'68294051AG',
      b'68294051AI',
      b'68294052AG',
      b'68294063AG',
      b'68294063AH',
      b'68294063AI',
      b'68434846AC',
      b'68434858AC',
      b'68434860AC',
      b'68453503AC',
      b'68453505AC',
      b'68453511AC',
      b'68453513AD',
      b'68453514AD',
      b'68510283AG',
      b'68527375AD',
      b'68527346AE',
    ],
    (Ecu.srs, 0x744, None): [
      b'68428609AB',
      b'68441329AB',
      b'68473844AB',
      b'68490898AA',
      b'68500728AA',
      b'68615033AA',
    ],
    (Ecu.abs, 0x747, None): [
      b'68292406AH',
      b'68432418AB',
      b'68432418AD',
      b'68436004AD',
      b'68436004AE',
      b'68438454AC',
      b'68438454AD',
      b'68438456AE',
      b'68438456AF',
      b'68535469AB',
      b'68535470AC',
      b'68586307AB',
      b'68548900AB',
    ],
    (Ecu.fwdRadar, 0x753, None): [
      b'04672892AB',
      b'04672932AB',
      b'68320950AH',
      b'68320950AI',
      b'68320950AJ',
      b'68320950AL',
      b'68320950AM',
      b'68454268AB',
      b'68475160AE',
      b'68475160AF',
      b'68475160AG',
    ],
    (Ecu.eps, 0x75A, None): [
      b'68273275AF',
      b'68273275AG',
      b'68312176AE',
      b'68312176AG',
      b'68440789AC',
      b'68466110AB',
      b'68469901AA',
      b'68522583AB',
      b'68522585AB',
      b'68552788AA',
      b'68552790AA',
      b'68585112AB',
      b'68552789AA',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'05036065AE ',
      b'05036066AE ',
      b'68378701AI ',
      b'68378758AM ',
      b'68448163AJ',
      b'68448165AK',
      b'68500630AD',
      b'68500630AE',
      b'68539650AD',
      b'05149846AA ',
    ],
    (Ecu.transmission, 0x7e1, None): [
      b'68360078AL',
      b'68360080AM',
      b'68360081AM',
      b'68360085AL',
      b'68384328AD',
      b'68384332AD',
      b'68445533AB',
      b'68484467AC',
      b'68502994AD',
      b'68540431AB',
      b'68520867AE',
    ],
  },

  CAR.RAM_HD: {
    (Ecu.combinationMeter, 0x742, None): [
      b'68361606AH',
      b'68437735AC',
      b'68492693AD',
      b'68525485AB',
      b'68525487AB',
      b'68525498AB',
    ],
    (Ecu.srs, 0x744, None): [
      b'68399794AC',
      b'68428503AA',
      b'68428505AA',
      b'68428507AA',
    ],
    (Ecu.abs, 0x747, None): [
      b'68334977AH',
      b'68455481AC',
      b'68504022AA',
      b'68504022AB',
      b'68504022AC',
      b'68530686AB',
      b'68530686AC',
    ],
    (Ecu.fwdRadar, 0x753, None): [
      b'04672895AB',
      b'56029827AG',
      b'56029827AH',
      b'68462657AE',
      b'68484694AD',
      b'68484694AE',
    ],
    (Ecu.eps, 0x761, None): [
      b'68421036AC',
      b'68507906AB',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'52370131AF',
      b'52370231AF',
      b'52370231AG',
      b'52370931CT',
      b'52401032AE',
      b'52421132AF',
      b'68527616AD ',
      b'M2370131MB',
      b'M2421132MB',
    ],
  },
}

DBC = {
  CAR.PACIFICA_2017_HYBRID: dbc_dict('chrysler_pacifica_2017_hybrid_generated', 'chrysler_pacifica_2017_hybrid_private_fusion'),
  CAR.PACIFICA_2018: dbc_dict('chrysler_pacifica_2017_hybrid_generated', 'chrysler_pacifica_2017_hybrid_private_fusion'),
  CAR.PACIFICA_2020: dbc_dict('chrysler_pacifica_2017_hybrid_generated', 'chrysler_pacifica_2017_hybrid_private_fusion'),
  CAR.PACIFICA_2018_HYBRID: dbc_dict('chrysler_pacifica_2017_hybrid_generated', 'chrysler_pacifica_2017_hybrid_private_fusion'),
  CAR.PACIFICA_2019_HYBRID: dbc_dict('chrysler_pacifica_2017_hybrid_generated', 'chrysler_pacifica_2017_hybrid_private_fusion'),
  CAR.JEEP_CHEROKEE: dbc_dict('chrysler_pacifica_2017_hybrid_generated', 'chrysler_pacifica_2017_hybrid_private_fusion'),
  CAR.JEEP_CHEROKEE_2019: dbc_dict('chrysler_pacifica_2017_hybrid_generated', 'chrysler_pacifica_2017_hybrid_private_fusion'),
  CAR.RAM_1500: dbc_dict('chrysler_ram_dt_generated', None),
  CAR.RAM_HD: dbc_dict('chrysler_ram_hd_generated', None),
}
