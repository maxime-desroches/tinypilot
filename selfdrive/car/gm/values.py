from cereal import car
from selfdrive.car import dbc_dict

class CAR:
  VOLT = "CHEVROLET VOLT PREMIER 2017"
  CADILLAC_CT6 = "CADILLAC CT6 SUPERCRUISE 2018"

class CruiseButtons:
  UNPRESS     = 1
  RES_ACCEL   = 2
  DECEL_SET   = 3
  MAIN        = 5
  CANCEL      = 6

def is_eps_status_ok(eps_status, car_fingerprint):
  valid_eps_status = []
  if car_fingerprint == CAR.VOLT:
    valid_eps_status += [0, 1]
  elif car_fingerprint == CAR.CADILLAC_CT6:
    valid_eps_status += [0, 1, 4, 5, 6]
  return eps_status in valid_eps_status

def parse_gear_shifter(can_gear):
  if can_gear == 0:
    return car.CarState.GearShifter.park
  elif can_gear == 1:
    return car.CarState.GearShifter.neutral
  elif can_gear == 2:
    return car.CarState.GearShifter.drive
  elif can_gear == 3:
    return car.CarState.GearShifter.reverse
  else:
    return car.CarState.GearShifter.unknown

FINGERPRINTS = {
  CAR.VOLT: [
  # Volt Premier w/ ACC 2017
  {
    170: 8, 171: 8, 189: 7, 190: 6, 193: 8, 197: 8, 199: 4, 201: 8, 209: 7, 211: 2, 241: 6, 288: 5, 289: 8, 298: 8, 304: 1, 308: 4, 309: 8, 311: 8, 313: 8, 320: 3, 328: 1, 352: 5, 381: 6, 386: 8, 388: 8, 389: 2, 390: 7, 417: 7, 419: 1, 426: 7, 451: 8, 452: 8, 453: 6, 454: 8, 456: 8, 479: 3, 481: 7, 485: 8, 489: 8, 493: 8, 495: 4, 497: 8, 499: 3, 500: 6, 501: 8, 508: 8, 528: 4, 532: 6, 546: 7, 550: 8, 554: 3, 558: 8, 560: 8, 562: 8, 563: 5, 564: 5, 565: 5, 566: 5, 567: 3, 568: 1, 573: 1, 577: 8, 647: 3, 707: 8, 711: 6, 761: 7, 810: 8, 840: 5, 842: 5, 844: 8, 866: 4, 961: 8, 969: 8, 977: 8, 979: 7, 988: 6, 989: 8, 995: 7, 1001: 8, 1005: 6, 1009: 8, 1017: 8, 1019: 2, 1020: 8, 1105: 6, 1187: 4, 1217: 8, 1221: 5, 1223: 3, 1225: 7, 1227: 4, 1233: 8, 1249: 8, 1257: 6, 1265: 8, 1267: 1, 1273: 3, 1275: 3, 1280: 4, 1300: 8, 1322: 6, 1323: 4, 1328: 4, 1417: 8, 1601: 8, 1905: 7, 1906: 7, 1907: 7, 1910: 7, 1912: 7, 1922: 7, 1927: 7, 1928: 7, 2016: 8, 2020: 8, 2024: 8, 2028: 8
  },
  # Volt Premier w/ ACC 2018
  {
    170: 8, 171: 8, 189: 7, 190: 6, 193: 8, 197: 8, 199: 4, 201: 8, 209: 7, 211: 2, 241: 6, 288: 5, 298: 8, 304: 1, 308: 4, 309: 8, 311: 8, 313: 8, 320: 3, 328: 1, 352: 5, 381: 6, 384: 4, 386: 8, 388: 8, 389: 2, 390: 7, 417: 7, 419: 1, 426: 7, 451: 8, 452: 8, 453: 6, 454: 8, 456: 8, 479: 3, 481: 7, 485: 8, 489: 8, 493: 8, 495: 4, 497: 8, 499: 3, 500: 6, 501: 8, 508: 8, 528: 4, 532: 6, 546: 7, 550: 8, 554: 3, 558: 8, 560: 8, 562: 8, 563: 5, 564: 5, 565: 5, 566: 5, 567: 3, 568: 1, 573: 1, 577: 8, 608: 8, 609: 6, 610: 6, 611: 6, 612: 8, 613: 8, 647: 3, 707: 8, 711: 6, 715: 8, 717: 5, 761: 7, 810: 8, 840: 5, 842: 5, 844: 8, 866: 4, 869: 4, 880: 6, 961: 8, 967: 4, 969: 8, 977: 8, 979: 7, 988: 6, 989: 8, 995: 7, 1001: 8, 1005: 6, 1009: 8, 1017: 8, 1019: 2, 1020: 8, 1033: 7, 1034: 7, 1105: 6, 1187: 4, 1217: 8, 1221: 5, 1223: 3, 1225: 7, 1227: 4, 1233: 8, 1249: 8, 1257: 6, 1265: 8, 1267: 1, 1273: 3, 1275: 3, 1280: 4, 1296: 4, 1300: 8, 1322: 6, 1323: 4, 1328: 4, 1417: 8, 1601: 8, 1905: 7, 1906: 7, 1907: 7, 1910: 7, 1912: 7, 1922: 7, 1927: 7, 1930: 7, 2016: 8, 2020: 8, 2024: 8, 2028: 8
  },
  # GMC Acadia Denali w/ ACC & LKAS 2018
  {
    190: 6, 193: 8, 197: 8, 199: 4, 201: 8, 208: 8, 209: 7, 211: 2, 241: 6, 249: 8, 288: 5, 289: 8, 298: 8, 304: 1, 309: 8, 313: 8, 320: 3, 322: 7, 328: 1, 338: 6, 340: 6, 352: 5, 381: 8, 384: 4, 386: 8, 388: 8, 393: 8, 398: 8, 413: 8, 417: 7, 419: 1, 422: 4, 426: 7, 431: 8, 442: 8, 451: 8, 452: 8, 453: 6, 454: 8, 455: 7, 462: 4, 463: 3, 479: 3, 481: 7, 485: 8, 489: 8, 497: 8, 499: 3, 500: 6, 501: 8, 508: 8, 510: 8, 532: 6, 554: 3, 560: 8, 562: 8, 563: 5, 564: 5, 567: 5, 573: 1, 577: 8, 608: 8, 609: 6, 610: 6, 611: 6, 612: 8, 613: 8, 647: 6, 707: 8, 715: 8, 717: 5, 753: 5, 761: 7, 840: 5, 842: 5, 844: 8, 866: 4, 869: 4, 880: 6, 961: 8, 969: 8, 977: 8, 979: 8, 985: 5, 1001: 8, 1005: 6, 1009: 8, 1017: 8, 1020: 8, 1033: 7, 1034: 7, 1105: 6, 1217: 8, 1221: 5, 1225: 8, 1233: 8, 1249: 8, 1257: 6, 1265: 8, 1267: 1, 1280: 4, 1296: 4, 1300: 8, 1322: 6, 1328: 4, 1417: 8, 1601: 8, 1906: 7, 1907: 7, 1912: 7, 1914: 7, 1919: 7, 1920: 7, 1930: 7, 2016: 8, 2024: 8
  }],
  CAR.CADILLAC_CT6: [{
    190: 6, 193: 8, 197: 8, 199: 4, 201: 8, 209: 7, 211: 2, 241: 6, 249: 8, 288: 5, 298: 8, 304: 1, 309: 8, 313: 8, 320: 3, 322: 7, 328: 1, 336: 1, 338: 6, 340: 6, 352: 5, 354: 5, 356: 8, 368: 3, 372: 5, 381: 8, 386: 8, 393: 7, 398: 8, 407: 7, 413: 8, 417: 7, 419: 1, 422: 4, 426: 7, 431: 8, 442: 8, 451: 8, 452: 8, 453: 6, 455: 7, 456: 8, 458: 5, 460: 5, 462: 4, 463: 3, 479: 3, 481: 7, 485: 8, 487: 8, 489: 8, 495: 4, 497: 8, 499: 3, 500: 6, 501: 8, 508: 8, 528: 5, 532: 6, 534: 2, 554: 3, 560: 8, 562: 8, 563: 5, 564: 5, 565: 5, 567: 5, 569: 3, 573: 1, 577: 8, 608: 8, 609: 6, 610: 6, 611: 6, 612: 8, 613: 8, 647: 6, 707: 8, 715: 8, 717: 5, 719: 5, 723: 2, 753: 5, 761: 7, 800: 6, 801: 8, 804: 3, 810: 8, 832: 8, 833: 8, 834: 8, 835: 6, 836: 5, 837: 8, 838: 8, 839: 8, 840: 5, 842: 5, 844: 8, 866: 4, 869: 4, 880: 6, 884: 8, 961: 8, 969: 8, 977: 8, 979: 8, 985: 5, 1001: 8, 1005: 6, 1009: 8, 1011: 6, 1013: 1, 1017: 8, 1019: 2, 1020: 8, 1105: 6, 1217: 8, 1221: 5, 1223: 3, 1225: 7, 1233: 8, 1249: 8, 1257: 6, 1259: 8, 1261: 7, 1263: 4, 1265: 8, 1267: 1, 1280: 4, 1296: 4, 1300: 8, 1322: 6, 1417: 8, 1601: 8, 1906: 7, 1907: 7, 1912: 7, 1914: 7, 1918: 7, 1919: 7, 1934: 7, 2016: 8, 2024: 8
  }],
}

STEER_THRESHOLD = 1.0

DBC = {
  CAR.VOLT: dbc_dict('gm_global_a_powertrain', 'gm_global_a_object', chassis_dbc='gm_global_a_chassis'),
  CAR.CADILLAC_CT6: dbc_dict('cadillac_ct6_powertrain', 'cadillac_ct6_object', chassis_dbc='cadillac_ct6_chassis'),
}
