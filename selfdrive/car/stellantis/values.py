# flake8: noqa

from selfdrive.car import dbc_dict
from cereal import car
Ecu = car.CarParams.Ecu

class CarControllerParams:
  STEER_STEP = 2
  HUD_STEP = 8
  STEER_MAX = 255
  STEER_DELTA_UP = 5
  STEER_DELTA_DOWN = 10
  STEER_DRIVER_ALLOWANCE = 80
  STEER_DRIVER_MULTIPLIER = 3
  STEER_DRIVER_FACTOR = 1
  STEER_THRESHOLD = 100

class CAR:
  RAM_1500_DT = "RAM 1500 5TH GEN"

# TODO: really need some UDS fingerprinting up in here

FINGERPRINTS = {
  CAR.RAM_1500_DT: [
    {35: 8, 37: 8, 39: 8, 41: 8, 43: 8, 47: 8, 49: 8, 53: 8, 55: 8, 113: 8, 119: 2, 121: 8, 123: 7, 125: 6, 127: 8, 129: 8, 131: 8, 133: 8, 135: 8, 137: 8, 139: 8, 141: 8, 145: 8, 147: 8, 149: 7, 153: 8, 155: 8, 157: 8, 163: 8, 164: 8, 166: 8, 167: 8, 169: 8, 171: 8, 173: 5, 177: 3, 179: 8, 181: 8, 213: 3, 221: 8, 232: 8, 250: 8, 278: 8, 289: 5, 293: 3, 295: 8, 296: 8, 297: 4, 298: 8, 299: 8, 305: 8, 307: 8, 311: 8, 315: 8, 317: 8, 319: 8, 323: 8, 333: 8, 334: 8, 341: 8, 343: 8, 345: 8, 347: 8, 409: 6, 421: 8, 448: 6, 456: 4, 464: 8, 489: 8, 491: 8, 502: 8, 503: 8, 505: 8, 507: 5, 516: 7, 517: 7, 524: 8, 526: 6, 557: 8, 560: 8, 584: 8, 601: 8, 605: 8, 607: 8, 609: 8, 611: 8, 613: 8, 623: 8, 631: 8, 633: 8, 634: 8, 635: 8, 637: 8, 641: 8, 643: 8, 645: 2, 649: 8, 650: 8, 651: 8, 656: 4, 657: 8, 659: 5, 663: 8, 664: 8, 673: 8, 676: 8, 679: 8, 685: 8, 687: 8, 689: 5, 706: 8, 709: 8, 710: 8, 711: 8, 720: 6, 752: 2, 754: 8, 773: 8, 788: 3, 792: 8, 808: 8, 818: 8, 819: 8, 822: 8, 823: 8, 825: 2, 838: 2, 840: 8, 848: 8, 856: 4, 860: 6, 862: 8, 875: 2, 897: 8, 906: 8, 910: 8, 926: 3, 929: 8, 930: 8, 931: 8, 932: 8, 937: 8, 938: 8, 939: 8, 940: 8, 941: 8, 942: 8, 943: 8, 947: 8, 948: 8, 956: 8, 961: 8, 962: 8, 969: 4, 971: 8, 972: 8, 973: 8, 979: 8, 980: 8, 981: 8, 982: 8, 983: 8, 984: 8, 992: 8, 993: 7, 995: 8, 996: 8, 1000: 8, 1001: 8, 1002: 8, 1003: 8, 1008: 8, 1009: 8, 1010: 8, 1011: 8, 1012: 8, 1013: 8, 1014: 8, 1015: 8, 1024: 8, 1025: 8, 1026: 8, 1031: 8, 1033: 8, 1050: 8, 1059: 8, 1062: 8, 1098: 8, 1100: 8, 2015: 8, 2016: 8, 2017: 8, 2024: 8, 2025: 8},
    {35: 8, 37: 8, 39: 8, 43: 8, 47: 8, 49: 8, 53: 8, 55: 8, 113: 8, 119: 2, 121: 8, 123: 7, 125: 6, 127: 8, 129: 8, 131: 8, 133: 8, 135: 8, 137: 8, 139: 8, 141: 8, 145: 8, 147: 8, 149: 7, 153: 8, 155: 8, 157: 8, 163: 8, 164: 8, 166: 8, 167: 8, 169: 8, 171: 8, 173: 5, 177: 3, 179: 8, 181: 8, 213: 3, 221: 8, 232: 8, 250: 8, 276: 8, 277: 8, 278: 8, 289: 5, 293: 3, 295: 8, 296: 8, 297: 4, 299: 8, 301: 8, 302: 8, 305: 8, 307: 8, 311: 8, 317: 8, 319: 8, 323: 8, 327: 8, 333: 8, 334: 8, 341: 8, 343: 8, 345: 8, 347: 8, 421: 8, 448: 6, 456: 4, 457: 8, 464: 8, 489: 8, 491: 8, 502: 8, 503: 8, 507: 5, 516: 7, 517: 7, 524: 8, 526: 6, 557: 8, 560: 8, 584: 8, 601: 8, 605: 8, 607: 8, 609: 8, 613: 8, 623: 8, 631: 8, 633: 8, 634: 8, 635: 8, 637: 8, 641: 8, 643: 8, 645: 2, 649: 8, 650: 8, 651: 8, 656: 4, 657: 8, 663: 8, 673: 8, 676: 8, 679: 8, 685: 8, 687: 8, 689: 5, 706: 8, 709: 8, 710: 8, 711: 8, 720: 6, 738: 8, 752: 2, 754: 8, 773: 8, 792: 8, 808: 8, 812: 8, 813: 8, 814: 8, 818: 8, 819: 8, 821: 8, 822: 8, 823: 8, 825: 2, 838: 2, 840: 8, 847: 1, 848: 8, 856: 4, 860: 6, 862: 8, 874: 2, 876: 8, 897: 8, 906: 8, 910: 8, 926: 3, 929: 8, 930: 8, 931: 8, 932: 8, 937: 8, 938: 8, 939: 8, 940: 8, 941: 8, 942: 8, 943: 8, 947: 8, 948: 8, 961: 8, 962: 8, 969: 4, 971: 8, 972: 8, 973: 8, 975: 8, 976: 8, 979: 8, 980: 8, 981: 8, 982: 8, 983: 8, 984: 8, 992: 8, 993: 7, 995: 8, 996: 8, 1000: 8, 1001: 8, 1002: 8, 1003: 8, 1008: 8, 1009: 8, 1010: 8, 1011: 8, 1012: 8, 1013: 8, 1014: 8, 1015: 8, 1024: 8, 1025: 8, 1026: 8, 1030: 8, 1031: 8, 1033: 8, 1050: 8, 1059: 8, 1098: 8, 1100: 8},
    {35: 8, 37: 8, 39: 8, 43: 8, 47: 8, 49: 8, 53: 8, 55: 8, 113: 8, 119: 2, 121: 8, 123: 7, 125: 6, 127: 8, 129: 8, 131: 8, 133: 8, 135: 8, 137: 8, 139: 8, 141: 8, 145: 8, 147: 8, 149: 7, 153: 8, 155: 8, 157: 8, 163: 8, 164: 8, 166: 8, 167: 8, 169: 8, 171: 8, 173: 5, 177: 3, 179: 8, 181: 8, 213: 3, 221: 8, 232: 8, 250: 8, 289: 5, 293: 3, 295: 8, 296: 8, 297: 4, 299: 8, 301: 8, 302: 8, 305: 8, 307: 8, 311: 8, 317: 8, 319: 8, 323: 8, 334: 8, 337: 8, 343: 8, 347: 8, 409: 6, 421: 8, 448: 6, 456: 4, 464: 8, 489: 8, 491: 8, 502: 8, 503: 8, 507: 5, 516: 7, 517: 7, 524: 8, 526: 6, 557: 8, 560: 8, 584: 8, 601: 8, 605: 8, 607: 8, 609: 8, 613: 8, 623: 8, 631: 8, 633: 8, 634: 8, 635: 8, 637: 8, 641: 8, 643: 8, 645: 2, 649: 8, 650: 8, 651: 8, 656: 4, 657: 8, 659: 5, 663: 8, 664: 8, 673: 8, 676: 8, 679: 8, 685: 8, 687: 8, 689: 5, 706: 8, 709: 8, 710: 8, 711: 8, 720: 6, 752: 2, 754: 8, 773: 8, 788: 3, 792: 8, 808: 8, 812: 8, 813: 8, 814: 8, 818: 8, 819: 8, 821: 8, 822: 8, 825: 2, 838: 2, 840: 8, 847: 1, 848: 8, 856: 4, 860: 6, 862: 8, 876: 8, 897: 8, 906: 8, 910: 8, 926: 3, 929: 8, 930: 8, 931: 8, 932: 8, 937: 8, 938: 8, 939: 8, 940: 8, 941: 8, 942: 8, 943: 8, 947: 8, 948: 8, 956: 8, 961: 8, 962: 8, 969: 4, 971: 8, 972: 8, 973: 8, 979: 8, 980: 8, 981: 8, 982: 8, 983: 8, 984: 8, 992: 8, 993: 7, 995: 8, 996: 8, 1000: 8, 1001: 8, 1002: 8, 1003: 8, 1008: 8, 1009: 8, 1010: 8, 1011: 8, 1012: 8, 1013: 8, 1014: 8, 1015: 8, 1024: 8, 1025: 8, 1026: 8, 1030: 8, 1031: 8, 1033: 8, 1050: 8, 1059: 8, 1062: 8, 1098: 8, 1100: 8},
    {35: 8, 37: 8, 39: 8, 41: 8, 43: 8, 47: 8, 49: 8, 53: 8, 55: 8, 113: 8, 119: 2, 121: 8, 123: 7, 125: 6, 127: 8, 129: 8, 131: 8, 133: 8, 135: 8, 137: 8, 139: 8, 141: 8, 145: 8, 147: 8, 149: 7, 153: 8, 155: 8, 157: 8, 163: 8, 164: 8, 166: 8, 167: 8, 169: 8, 171: 8, 173: 5, 177: 3, 179: 8, 181: 8, 213: 3, 221: 8, 232: 8, 250: 8, 289: 5, 293: 3, 295: 8, 296: 8, 297: 4, 298: 8, 299: 8, 305: 8, 307: 8, 311: 8, 315: 8, 317: 8, 319: 8, 323: 8, 333: 8, 334: 8, 343: 8, 347: 8, 421: 8, 448: 6, 456: 4, 464: 8, 489: 8, 491: 8, 502: 8, 503: 8, 505: 8, 507: 5, 516: 7, 517: 7, 524: 8, 526: 6, 557: 8, 560: 8, 584: 8, 601: 8, 605: 8, 607: 8, 609: 8, 611: 8, 613: 8, 623: 8, 631: 8, 633: 8, 634: 8, 635: 8, 641: 8, 643: 8, 649: 8, 650: 8, 656: 4, 657: 8, 663: 8, 673: 8, 676: 8, 679: 8, 687: 8, 689: 5, 706: 8, 709: 8, 710: 8, 711: 8, 720: 6, 752: 2, 754: 8, 773: 8, 788: 3, 792: 8, 808: 8, 818: 8, 822: 8, 825: 2, 838: 2, 840: 8, 847: 1, 848: 8, 856: 4, 860: 6, 875: 2, 897: 8, 906: 8, 910: 8, 926: 3, 937: 8, 938: 8, 939: 8, 940: 8, 941: 8, 942: 8, 943: 8, 947: 8, 948: 8, 962: 8, 969: 4, 973: 8, 979: 8, 980: 8, 981: 8, 982: 8, 983: 8, 984: 8, 992: 8, 993: 7, 995: 8, 996: 8, 1000: 8, 1001: 8, 1002: 8, 1003: 8, 1008: 8, 1009: 8, 1010: 8, 1011: 8, 1012: 8, 1013: 8, 1014: 8, 1015: 8, 1024: 8, 1025: 8, 1026: 8, 1031: 8, 1033: 8, 1050: 8, 1059: 8, 1098: 8, 1100: 8}
  ],
}

DBC = {
  CAR.RAM_1500_DT: dbc_dict('stellantis_dasm', 'stellantis_dasm'),
}
