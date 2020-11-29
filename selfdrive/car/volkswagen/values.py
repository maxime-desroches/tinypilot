from cereal import car
from selfdrive.car import dbc_dict

class CarControllerParams:
  HCA_STEP = 2                   # HCA message frequency 50Hz on all vehicles
  MQB_LDW_STEP = 10              # LDW message frequency 10Hz on MQB
  PQ_LDW_STEP = 5                # LDW message frequency 20Hz on PQ35/PQ46/NMS
  GRA_ACC_STEP = 3               # GRA_ACC_01 message frequency 33Hz

  GRA_VBP_STEP = 100             # Send ACC virtual button presses once a second
  GRA_VBP_COUNT = 16             # Send VBP messages for ~0.5s (GRA_ACC_STEP * 16)

  # Observed documented MQB limits: 3.00 Nm max, rate of change 5.00 Nm/sec.
  # Limiting rate-of-change based on real-world testing and Comma's safety
  # requirements for minimum time to lane departure.
  STEER_MAX = 300                # Max heading control assist torque 3.00 Nm
  STEER_DELTA_UP = 10            # Max HCA reached in 0.60s (STEER_MAX / (50Hz * 0.60))
  STEER_DELTA_DOWN = 10          # Min HCA reached in 0.60s (STEER_MAX / (50Hz * 0.60))
  STEER_DRIVER_ALLOWANCE = 80
  STEER_DRIVER_MULTIPLIER = 3    # weight driver torque heavily
  STEER_DRIVER_FACTOR = 1        # from dbc

class CANBUS:
  pt = 0
  cam = 2

NWL = car.CarParams.NetworkLocation
TRANS = car.CarParams.TransmissionType
GEAR = car.CarState.GearShifter

BUTTON_STATES = {
  "accelCruise": False,
  "decelCruise": False,
  "cancel": False,
  "setCruise": False,
  "resumeCruise": False,
  "gapAdjustCruise": False
}

MQB_LDW_MESSAGES = {
  "none": 0,                            # Nothing to display
  "laneAssistUnavailChime": 1,          # "Lane Assist currently not available." with chime
  "laneAssistUnavailNoSensorChime": 3,  # "Lane Assist not available. No sensor view." with chime
  "laneAssistTakeOverUrgent": 4,        # "Lane Assist: Please Take Over Steering" with urgent beep
  "emergencyAssistUrgent": 6,           # "Emergency Assist: Please Take Over Steering" with urgent beep
  "laneAssistTakeOverChime": 7,         # "Lane Assist: Please Take Over Steering" with chime
  "laneAssistTakeOverSilent": 8,        # "Lane Assist: Please Take Over Steering" silent
  "emergencyAssistChangingLanes": 9,    # "Emergency Assist: Changing lanes..." with urgent beep
  "laneAssistDeactivated": 10,          # "Lane Assist deactivated." silent with persistent icon afterward
}

class CAR:
  GENERICMQB = "Generic Volkswagen MQB Platform Vehicle"
  GENERICPQ = "Generic Volkswagen PQ35/PQ46/NMS Platform Vehicle"

FINGERPRINTS = {
  CAR.GENERICMQB: [
    {178: 8, 1600: 8, 1601: 8, 1603: 8, 1605: 8, 695: 8, 1624: 8, 1626: 8, 1629: 8, 1631: 8, 1122: 8, 1123: 8,
     1124: 8, 1646: 8, 1648: 8, 1153: 8, 134: 8, 1162: 8, 1175: 8, 159: 8, 795: 8, 679: 8, 681: 8, 173: 8, 1712: 6,
     1714: 8, 1716: 8, 1717: 8, 1719: 8, 1720: 8, 1721: 8, 1312: 8, 806: 8, 253: 8, 1792: 8, 257: 8, 260: 8, 262: 8,
     897: 8, 264: 8, 779: 8, 780: 8, 783: 8, 278: 8, 279: 8, 792: 8, 283: 8, 285: 8, 286: 8, 901: 8, 288: 8, 289: 8,
     290: 8, 804: 8, 294: 8, 807: 8, 808: 8, 809: 8, 299: 8, 302: 8, 1351: 8, 346: 8, 870: 8, 1385: 8, 896: 8, 64: 8,
     898: 8, 1413: 8, 917: 8, 919: 8, 927: 8, 1440: 5, 929: 8, 930: 8, 427: 8, 949: 8, 958: 8, 960: 4, 418: 8, 981: 8,
     987: 8, 988: 8, 991: 8, 997: 8, 1000: 8, 1514: 8, 1515: 8, 1520: 8, 1019: 8, 385: 8, 668: 8, 1120: 8,
     1438: 8, 1461: 8, 391: 8, 1511: 8, 1516: 8, 568: 8, 569: 8, 826: 8, 827: 8, 1156: 8, 1157: 8, 1158: 8, 1471: 8,
     1635: 8, 376: 8, 295: 8, 791: 8, 799: 8, 838: 8, 389: 8, 840: 8, 841: 8, 842: 8, 843: 8, 844: 8, 845: 8,
     314: 8, 787: 8, 788: 8, 789: 8, 802: 8, 839: 8, 1332: 8, 1872: 8, 1976: 8, 1977: 8, 1985: 8, 2015: 8, 592: 8,
     593: 8, 594: 8, 595: 8, 596: 8, 684: 8, 506: 8, 846: 8, 847: 8, 1982: 8
     }],
  CAR.GENERICPQ: [
    # kamold, Edgy, austinc3030, Roy_001
    {80: 4, 194: 8, 208: 6, 210: 5, 294: 8, 416: 8, 428: 8, 640: 8, 648: 8, 800: 8, 835: 3, 870: 8, 872: 8, 878: 8,
     896: 8, 906: 4, 912: 8, 914: 8, 919: 8, 928: 8, 978: 7, 1056: 8, 1088: 8, 1152: 8, 1175: 8, 1184: 8, 1192: 8,
     1312: 8, 1386: 8, 1392: 5, 1394: 1, 1408: 8, 1440: 8, 1463: 8, 1470: 5, 1472: 8, 1488: 8, 1490: 8, 1500: 8,
     1550: 2, 1651: 3, 1652: 8, 1654: 2, 1658: 4, 1691: 3, 1736: 2, 1757: 8, 1824: 7, 1845: 7, 2000: 8, 1420: 8},
    {80: 4, 194: 8, 208: 6, 210: 5, 416: 8, 428: 8, 513: 6, 640: 8, 648: 8, 768: 8, 769: 6, 770: 8, 784: 8, 785: 8,
     787: 8, 788: 8, 790: 8, 791: 8, 793: 8, 794: 8, 796: 8, 797: 8, 799: 8, 800: 8, 802: 8, 803: 8, 805: 8, 806: 8,
     808: 8, 809: 8, 811: 8, 812: 8, 814: 8, 815: 8, 817: 8, 818: 8, 820: 8, 821: 8, 823: 8, 824: 8, 826: 8, 827: 8,
     829: 8, 830: 8, 832: 8, 833: 8, 835: 8, 836: 8, 838: 8, 839: 8, 841: 8, 842: 8, 844: 8, 845: 8, 847: 8, 848: 8,
     850: 8, 851: 8, 853: 8, 854: 8, 856: 8, 857: 8, 859: 8, 860: 8, 862: 8, 863: 8, 865: 8, 866: 8, 868: 8, 869: 8,
     871: 8, 872: 8, 874: 8, 875: 8, 877: 8, 878: 8, 881: 8, 882: 8, 884: 8, 885: 8, 887: 8, 888: 8, 890: 8, 891: 8,
     893: 8, 894: 8, 895: 8, 896: 8, 897: 8, 898: 8, 899: 4, 906: 4, 912: 8, 914: 8, 928: 8, 978: 7, 1023: 3, 1056: 8,
     1088: 8, 1152: 8, 1175: 8, 1184: 8, 1192: 8, 1281: 8, 1312: 8, 1329: 8, 1330: 3, 1331: 8, 1332: 8, 1333: 8,
     1334: 8, 1335: 8, 1336: 8, 1337: 8, 1338: 8, 1339: 8, 1340: 8, 1341: 8, 1342: 8, 1343: 8, 1344: 8, 1345: 8,
     1346: 8, 1392: 5, 1394: 1, 1408: 8, 1420: 8, 1440: 8, 1463: 8, 1470: 5, 1488: 8, 1490: 8, 1500: 8, 1523: 3,
     1550: 2, 1585: 8, 1626: 8, 1651: 8, 1654: 3, 1658: 2, 1691: 4, 1736: 2, 1792: 8, 1824: 7, 1977: 8, 2000: 8},
    # cd (powertrain CAN direct)
    {16: 7, 17: 7, 80: 4, 174: 8, 194: 8, 208: 6, 416: 8, 428: 8, 640: 8, 648: 8, 672: 8, 800: 8, 896: 8, 906: 4,
     912: 8, 914: 8, 915: 8, 919: 8, 928: 8, 946: 8, 976: 6, 978: 7, 1056: 8, 1152: 8, 1160: 8, 1162: 8, 1164: 8,
     1175: 8, 1184: 8, 1192: 8, 1306: 8, 1312: 8, 1344: 8, 1360: 8, 1386: 8, 1392: 5, 1394: 1, 1408: 8, 1416: 8,
     1420: 8, 1423: 8, 1440: 8, 1463: 8, 1488: 8, 1490: 8, 1494: 2, 1500: 8, 1504: 8, 1523: 8, 1527: 4, 1654: 2,
     1658: 2, 1754: 8, 1824: 7, 1827: 7, 2000: 8},
    # khonsu's drivetrain
    {16: 7, 17: 7, 80: 4, 174: 8, 194: 8, 208: 6, 416: 8, 428: 8, 640: 8, 648: 8, 672: 8, 800: 8, 896: 8, 906: 4,
     912: 8, 914: 8, 915: 8, 919: 8, 928: 8, 946: 8, 976: 6, 978: 7, 1056: 8, 1088: 8, 1096: 5, 1152: 8, 1160: 8,
     1162: 8, 1184: 8, 1192: 8, 1312: 8, 1344: 8, 1352: 3, 1360: 8, 1386: 8, 1392: 5, 1394: 1, 1408: 8, 1416: 8,
     1420: 8, 1423: 8, 1440: 8, 1463: 8, 1488: 8, 1490: 8, 1500: 8, 1504: 8, 1512: 8, 1523: 8, 1527: 4, 1824: 7,
     1827: 7, 2000: 8},
  ],
}

MQB_CARS = [CAR.GENERICMQB]
PQ_CARS = [CAR.GENERICPQ]

DBC = {
  CAR.GENERICMQB: dbc_dict('vw_mqb_2010', None),
  CAR.GENERICPQ: dbc_dict('vw_golf_mk4', None),
}
