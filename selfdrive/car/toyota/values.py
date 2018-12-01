from selfdrive.car import dbc_dict

class CAR:
  PRIUS = "TOYOTA PRIUS 2017"
  RAV4H = "TOYOTA RAV4 HYBRID 2017"
  RAV4 = "TOYOTA RAV4 2017"
  COROLLA = "TOYOTA COROLLA 2017"
  LEXUS_RXH = "LEXUS RX HYBRID 2017"
  CHR = "TOYOTA C-HR 2018"
  CHRH = "TOYOTA C-HR HYBRID 2018"
  CAMRY = "TOYOTA CAMRY 2018"
  CAMRYH = "TOYOTA CAMRY HYBRID 2018"
  HIGHLANDER = "TOYOTA HIGHLANDER 2017"
  HIGHLANDERH = "TOYOTA HIGHLANDER HYBRID 2018"


class ECU:
  CAM = 0 # camera
  DSU = 1 # driving support unit
  APGS = 2 # advanced parking guidance system


# addr: (ecu, cars, bus, 1/freq*100, vl)
STATIC_MSGS = [
  (0x130, ECU.CAM, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4, CAR.COROLLA, CAR.HIGHLANDER, CAR.HIGHLANDERH), 1, 100, '\x00\x00\x00\x00\x00\x00\x38'),
  (0x240, ECU.CAM, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4, CAR.COROLLA, CAR.HIGHLANDER, CAR.HIGHLANDERH), 1,   5, '\x00\x10\x01\x00\x10\x01\x00'),
  (0x241, ECU.CAM, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4, CAR.COROLLA, CAR.HIGHLANDER, CAR.HIGHLANDERH), 1,   5, '\x00\x10\x01\x00\x10\x01\x00'),
  (0x244, ECU.CAM, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4, CAR.COROLLA, CAR.HIGHLANDER, CAR.HIGHLANDERH), 1,   5, '\x00\x10\x01\x00\x10\x01\x00'),
  (0x245, ECU.CAM, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4, CAR.COROLLA, CAR.HIGHLANDER, CAR.HIGHLANDERH), 1,   5, '\x00\x10\x01\x00\x10\x01\x00'),
  (0x248, ECU.CAM, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4, CAR.COROLLA, CAR.HIGHLANDER, CAR.HIGHLANDERH), 1,   5, '\x00\x00\x00\x00\x00\x00\x01'),
  (0x367, ECU.CAM, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4, CAR.COROLLA, CAR.HIGHLANDER, CAR.HIGHLANDERH), 0,  40, '\x06\x00'),
  (0x414, ECU.CAM, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4, CAR.COROLLA, CAR.HIGHLANDER, CAR.HIGHLANDERH), 0, 100, '\x00\x00\x00\x00\x00\x00\x17\x00'),
  (0x466, ECU.CAM, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4, CAR.HIGHLANDER, CAR.HIGHLANDERH), 1, 100, '\x20\x20\xAD'),
  (0x466, ECU.CAM, (CAR.COROLLA), 1, 100, '\x24\x20\xB1'),
  (0x489, ECU.CAM, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4, CAR.COROLLA, CAR.HIGHLANDER, CAR.HIGHLANDERH), 0, 100, '\x00\x00\x00\x00\x00\x00\x00'),
  (0x48a, ECU.CAM, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4, CAR.COROLLA, CAR.HIGHLANDER, CAR.HIGHLANDERH), 0, 100, '\x00\x00\x00\x00\x00\x00\x00'),
  (0x48b, ECU.CAM, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4, CAR.COROLLA, CAR.HIGHLANDER, CAR.HIGHLANDERH), 0, 100, '\x66\x06\x08\x0a\x02\x00\x00\x00'),
  (0x4d3, ECU.CAM, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4, CAR.COROLLA), 0, 100, '\x1C\x00\x00\x01\x00\x00\x00\x00'),

  (0x128, ECU.DSU, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4, CAR.COROLLA), 1,   3, '\xf4\x01\x90\x83\x00\x37'),
  (0x128, ECU.DSU, (CAR.HIGHLANDER, CAR.HIGHLANDERH), 1,   3, '\x03\x00\x20\x00\x00\x52'),
  (0x141, ECU.DSU, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4, CAR.COROLLA, CAR.HIGHLANDER, CAR.HIGHLANDERH), 1,   2, '\x00\x00\x00\x46'),
  (0x160, ECU.DSU, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4, CAR.COROLLA, CAR.HIGHLANDER, CAR.HIGHLANDERH), 1,   7, '\x00\x00\x08\x12\x01\x31\x9c\x51'),
  (0x161, ECU.DSU, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4, CAR.COROLLA), 1,   7, '\x00\x1e\x00\x00\x00\x80\x07'),
  (0X161, ECU.DSU, (CAR.HIGHLANDERH, CAR.HIGHLANDER), 1,  7, '\x00\x1e\x00\xd4\x00\x00\x5b'),
  (0x283, ECU.DSU, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4, CAR.COROLLA, CAR.HIGHLANDER, CAR.HIGHLANDERH), 0,   3, '\x00\x00\x00\x00\x00\x00\x8c'),
  (0x2E6, ECU.DSU, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH), 0,   3, '\xff\xf8\x00\x08\x7f\xe0\x00\x4e'),
  (0x2E7, ECU.DSU, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH), 0,   3, '\xa8\x9c\x31\x9c\x00\x00\x00\x02'),
  (0x33E, ECU.DSU, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH), 0,  20, '\x0f\xff\x26\x40\x00\x1f\x00'),
  (0x344, ECU.DSU, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4, CAR.COROLLA, CAR.HIGHLANDER, CAR.HIGHLANDERH), 0,   5, '\x00\x00\x01\x00\x00\x00\x00\x50'),
  (0x365, ECU.DSU, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.HIGHLANDERH), 0,  20, '\x00\x00\x00\x80\x03\x00\x08'),
  (0x365, ECU.DSU, (CAR.RAV4, CAR.COROLLA, CAR.HIGHLANDER), 0,  20, '\x00\x00\x00\x80\xfc\x00\x08'),
  (0x366, ECU.DSU, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.HIGHLANDERH), 0,  20, '\x00\x00\x4d\x82\x40\x02\x00'),
  (0x366, ECU.DSU, (CAR.RAV4, CAR.COROLLA, CAR.HIGHLANDER), 0,  20, '\x00\x72\x07\xff\x09\xfe\x00'),
  (0x470, ECU.DSU, (CAR.PRIUS, CAR.LEXUS_RXH), 1, 100, '\x00\x00\x02\x7a'),
  (0x470, ECU.DSU, (CAR.HIGHLANDER, CAR.HIGHLANDERH, CAR.RAV4H), 1,  100, '\x00\x00\x01\x79'),
  (0x4CB, ECU.DSU, (CAR.PRIUS, CAR.RAV4H, CAR.LEXUS_RXH, CAR.RAV4, CAR.COROLLA, CAR.HIGHLANDERH, CAR.HIGHLANDER), 0, 100, '\x0c\x00\x00\x00\x00\x00\x00\x00'),

  (0x292, ECU.APGS, (CAR.PRIUS), 0,   3, '\x00\x00\x00\x00\x00\x00\x00\x9e'),
  (0x32E, ECU.APGS, (CAR.PRIUS), 0,  20, '\x00\x00\x00\x00\x00\x00\x00\x00'),
  (0x396, ECU.APGS, (CAR.PRIUS), 0, 100, '\xBD\x00\x00\x00\x60\x0F\x02\x00'),
  (0x43A, ECU.APGS, (CAR.PRIUS), 0, 100, '\x84\x00\x00\x00\x00\x00\x00\x00'),
  (0x43B, ECU.APGS, (CAR.PRIUS), 0, 100, '\x00\x00\x00\x00\x00\x00\x00\x00'),
  (0x497, ECU.APGS, (CAR.PRIUS), 0, 100, '\x00\x00\x00\x00\x00\x00\x00\x00'),
  (0x4CC, ECU.APGS, (CAR.PRIUS), 0, 100, '\x0D\x00\x00\x00\x00\x00\x00\x00'),
]

ECU_FINGERPRINT = {
  ECU.CAM: 0x2e4,   # steer torque cmd
  ECU.DSU: 0x343,   # accel cmd
  ECU.APGS: 0x835,  # angle cmd
}


def check_ecu_msgs(fingerprint, ecu):
  # return True if fingerprint contains messages normally sent by a given ecu
  return ECU_FINGERPRINT[ecu] in fingerprint


FINGERPRINTS = {
  CAR.RAV4: [{
    36: 8, 37: 8, 170: 8, 180: 8, 186: 4, 426: 6, 452: 8, 464: 8, 466: 8, 467: 8, 547: 8, 548: 8, 552: 4, 562: 4, 608: 8, 610: 5, 643: 7, 705: 8, 725: 2, 740: 5, 800: 8, 835: 8, 836: 8, 849: 4, 869: 7, 870: 7, 871: 2, 896: 8, 897: 8, 900: 6, 902: 6, 905: 8, 911: 8, 916: 3, 918: 7, 921: 8, 933: 8, 944: 8, 945: 8, 951: 8, 955: 4, 956: 8, 979: 2, 998: 5, 999: 7, 1000: 8, 1001: 8, 1005: 2, 1008: 2, 1014: 8, 1017: 8, 1041: 8, 1042: 8, 1043: 8, 1044: 8, 1056: 8, 1059: 1, 1114: 8, 1161: 8, 1162: 8, 1163: 8, 1176: 8, 1177: 8, 1178: 8, 1179: 8, 1180: 8, 1181: 8, 1190: 8, 1191: 8, 1192: 8, 1196: 8, 1227: 8, 1228: 8, 1235: 8, 1237: 8, 1263: 8, 1264: 8, 1279: 8, 1408: 8, 1409: 8, 1410: 8, 1552: 8, 1553: 8, 1554: 8, 1555: 8, 1556: 8, 1557: 8, 1561: 8, 1562: 8, 1568: 8, 1569: 8, 1570: 8, 1571: 8, 1572: 8, 1584: 8, 1589: 8, 1592: 8, 1593: 8, 1595: 8, 1596: 8, 1597: 8, 1600: 8, 1656: 8, 1664: 8, 1728: 8, 1745: 8, 1779: 8, 1904: 8, 1912: 8, 1990: 8, 1998: 8
  }],
  CAR.RAV4H: [{
    36: 8, 37: 8, 170: 8, 180: 8, 186: 4, 296: 8, 426: 6, 452: 8, 464: 8, 466: 8, 467: 8, 547: 8, 548: 8, 550: 8, 552: 4, 560: 7, 562: 4, 581: 5, 608: 8, 610: 5, 643: 7, 705: 8, 713: 8, 725: 2, 740: 5, 800: 8, 835: 8, 836: 8, 849: 4, 869: 7, 870: 7, 871: 2, 896: 8, 897: 8, 900: 6, 902: 6, 905: 8, 911: 8, 916: 3, 918: 7, 921: 8, 933: 8, 944: 8, 945: 8, 950: 8, 951: 8, 953: 3, 955: 8, 956: 8, 979: 2, 998: 5, 999: 7, 1000: 8, 1001: 8, 1005: 2, 1008: 2, 1014: 8, 1017: 8, 1041: 8, 1042: 8, 1043: 8, 1044: 8, 1056: 8, 1059: 1, 1114: 8, 1161: 8, 1162: 8, 1163: 8, 1176: 8, 1177: 8, 1178: 8, 1179: 8, 1180: 8, 1181: 8, 1184: 8, 1185: 8, 1186: 8, 1190: 8, 1191: 8, 1192: 8, 1196: 8, 1197: 8, 1198: 8, 1199: 8, 1212: 8, 1227: 8, 1228: 8, 1232: 8, 1235: 8, 1237: 8, 1263: 8, 1264: 8, 1279: 8, 1408: 8, 1409: 8, 1410: 8, 1552: 8, 1553: 8, 1554: 8, 1555: 8, 1556: 8, 1557: 8, 1561: 8, 1562: 8, 1568: 8, 1569: 8, 1570: 8, 1571: 8, 1572: 8, 1584: 8, 1589: 8, 1592: 8, 1593: 8, 1595: 8, 1596: 8, 1597: 8, 1600: 8, 1656: 8, 1664: 8, 1728: 8, 1745: 8, 1779: 8, 1904: 8, 1912: 8, 1990: 8, 1998: 8
  },
  # CAR.RAV4 XLE_2018: (MINE!)
  {
    36: 8, 37: 8, 170: 8, 180: 8, 186: 4, 426: 6, 452: 8, 464: 8, 466: 8, 467: 8, 547: 8, 548: 8, 552: 4, 562: 4, 608: 8, 610: 5, 643: 7, 705: 8, 725: 2, 740: 5, 800: 8, 835: 8, 836: 8, 849: 4, 869: 7, 870: 7, 871: 2, 896: 8, 897: 8, 900: 6, 902: 6, 905: 8, 911: 8, 916: 3, 921: 8, 933: 8, 944: 8, 945: 8, 951: 8, 955: 4, 956: 8, 979: 2, 998: 5, 999: 7, 1000: 8, 1001: 8, 1005: 2, 1008: 2, 1014: 8, 1017: 8, 1041: 8, 1042: 8, 1043: 8, 1044: 8, 1056: 8, 1059: 1, 1114: 8, 1161: 8, 1162: 8, 1163: 8, 1176: 8, 1177: 8, 1178: 8, 1179: 8, 1180: 8, 1181: 8, 1190: 8, 1191: 8, 1192: 8, 1196: 8, 1227: 8, 1235: 8, 1237: 8, 1264: 8, 1279: 8, 1408: 8, 1409: 8, 1410: 8, 1552: 8, 1553: 8, 1554: 8, 1555: 8, 1556: 8, 1557: 8, 1561: 8, 1562: 8, 1568: 8, 1569: 8, 1570: 8, 1571: 8, 1572: 8, 1584: 8, 1589: 8, 1592: 8, 1593: 8, 1595: 8, 1596: 8, 1597: 8, 1600: 8, 1656: 8, 1664: 8, 1728: 8, 1745: 8, 1779: 8, 1904: 8, 1912: 8, 1990: 8, 1998: 8
  },
    # Chinese RAV4
  {
    36: 8, 37: 8, 170: 8, 180: 8, 186: 4, 355: 5, 426: 6, 452: 8, 464: 8, 466: 8, 467: 8, 547: 8, 548: 8, 552: 4, 562: 4, 608: 8, 610: 5, 643: 7, 705: 8, 725: 2, 740: 5, 742: 8, 743: 8, 800: 8, 830: 7, 835: 8, 836: 8, 849: 4, 869: 7, 870: 7, 871: 2, 896: 8, 897: 8, 900: 6, 902: 6, 905: 8, 911: 8, 916: 3, 921: 8, 922: 8, 933: 8, 944: 8, 945: 8, 951: 8, 955: 8, 956: 8, 979: 2, 998: 5, 999: 7, 1000: 8, 1001: 8, 1008: 2, 1017: 8, 1041: 8, 1042: 8, 1043: 8, 1044: 8, 1056: 8, 1059: 1, 1114: 8, 1161: 8, 1162: 8, 1163: 8, 1176: 8, 1177: 8, 1178: 8, 1179: 8, 1180: 8, 1181: 8, 1190: 8, 1191: 8, 1192: 8, 1196: 8, 1207: 8, 1227: 8, 1235: 8, 1263: 8, 1279: 8, 1552: 8, 1553: 8, 1554: 8, 1555: 8, 1556: 8, 1557: 8, 1561: 8, 1562: 8, 1568: 8, 1569: 8, 1570: 8, 1571: 8, 1572: 8, 1584: 8, 1589: 8, 1592: 8, 1593: 8, 1595: 8, 1596: 8, 1597: 8, 1600: 8, 1664: 8, 1728: 8, 1745: 8, 1779: 8
  }],
  CAR.PRIUS: [{
    36: 8, 37: 8, 166: 8, 170: 8, 180: 8, 295: 8, 296: 8, 426: 6, 452: 8, 466: 8, 467: 8, 550: 8, 552: 4, 560: 7, 562: 6, 581: 5, 608: 8, 610: 8, 614: 8, 643: 7, 658: 8, 713: 8, 740: 5, 742: 8, 743: 8, 800: 8, 810: 2, 814: 8, 829: 2, 830: 7, 835: 8, 836: 8, 863: 8, 869: 7, 870: 7, 871: 2, 898: 8, 900: 6, 902: 6, 905: 8, 918: 8, 921: 8, 933: 8, 944: 8, 945: 8, 950: 8, 951: 8, 953: 8, 955: 8, 956: 8, 971: 7, 975: 5, 993: 8, 998: 5, 999: 7, 1000: 8, 1001: 8, 1014: 8, 1017: 8, 1020: 8, 1041: 8, 1042: 8, 1044: 8, 1056: 8, 1057: 8, 1059: 1, 1071: 8, 1077: 8, 1082: 8, 1083: 8, 1084: 8, 1085: 8, 1086: 8, 1114: 8, 1132: 8, 1161: 8, 1162: 8, 1163: 8, 1175: 8, 1227: 8, 1228: 8, 1235: 8, 1237: 8, 1279: 8, 1552: 8, 1553: 8, 1556: 8, 1557: 8, 1568: 8, 1570: 8, 1571: 8, 1572: 8, 1595: 8, 1777: 8, 1779: 8, 1904: 8, 1912: 8, 1990: 8, 1998: 8
  },
  # Prius Prime
  {
    36: 8, 37: 8, 166: 8, 170: 8, 180: 8, 295: 8, 296: 8, 426: 6, 452: 8, 466: 8, 467: 8, 550: 8, 552: 4, 560: 7, 562: 6, 581: 5, 608: 8, 610: 8, 614: 8, 643: 7, 658: 8, 713: 8, 740: 5, 742: 8, 743: 8, 800: 8, 810: 2, 814: 8, 824: 2, 829: 2, 830: 7, 835: 8, 836: 8, 863: 8, 869: 7, 870: 7, 871: 2,898: 8, 900: 6, 902: 6, 905: 8, 913: 8, 918: 8, 921: 8, 933: 8, 944: 8, 945: 8, 950: 8, 951: 8, 953: 8, 955: 8, 956: 8, 971: 7, 974: 8, 975: 5, 993: 8, 998: 5, 999: 7, 1000: 8, 1001: 8, 1014: 8, 1017: 8, 1020: 8, 1041: 8, 1042: 8, 1044: 8, 1056: 8, 1057: 8, 1059: 1, 1071: 8, 1076: 8, 1077: 8, 1082: 8, 1083: 8, 1084: 8, 1085: 8, 1086: 8, 1114: 8, 1132: 8, 1161: 8, 1162: 8, 1163: 8, 1164: 8, 1165: 8, 1166: 8, 1167: 8, 1175: 8, 1227: 8, 1228: 8, 1235: 8, 1237: 8, 1279: 8, 1552: 8, 1553: 8, 1556: 8, 1557: 8, 1568: 8, 1570: 8, 1571: 8, 1572: 8, 1595: 8, 1777: 8, 1779: 8, 1904: 8, 1912: 8, 1990: 8, 1998: 8
  },
  # Taiwanese Prius Prime
  {
    36: 8, 37: 8, 166: 8, 170: 8, 180: 8, 295: 8, 296: 8, 426: 6, 452: 8, 466: 8, 467: 8, 550: 8, 552: 4, 560: 7, 562: 6, 581: 5, 608: 8, 610: 8, 614: 8, 643: 7, 658: 8, 713: 8, 740: 5, 742: 8, 743: 8, 800: 8, 810: 2, 814: 8, 824: 2, 829: 2, 830: 7, 835: 8, 836: 8, 845: 5, 863: 8, 869: 7, 870: 7, 871: 2,898: 8, 900: 6, 902: 6, 905: 8, 913: 8, 918: 8, 921: 8, 933: 8, 944: 8, 945: 8, 950: 8, 951: 8, 953: 8, 955: 8, 956: 8, 971: 7, 974: 8, 975: 5, 993: 8, 998: 5, 999: 7, 1000: 8, 1001: 8, 1005: 2, 1014: 8, 1017: 8, 1020: 8, 1041: 8, 1042: 8, 1044: 8, 1056: 8, 1057: 8, 1059: 1, 1071: 8, 1076: 8, 1077: 8, 1082: 8, 1083: 8, 1084: 8, 1085: 8, 1086: 8, 1114: 8, 1132: 8, 1161: 8, 1162: 8, 1163: 8, 1164: 8, 1165: 8, 1166: 8, 1167: 8, 1175: 8, 1227: 8, 1228: 8, 1235: 8, 1237: 8, 1264: 8, 1279: 8, 1552: 8, 1553: 8, 1556: 8, 1557: 8, 1568: 8, 1570: 8, 1571: 8, 1572: 8, 1595: 8, 1777: 8, 1779: 8, 1904: 8, 1912: 8, 1990: 8, 1998: 8
  }],
  CAR.COROLLA: [{
    36: 8, 37: 8, 170: 8, 180: 8, 186: 4, 426: 6, 452: 8, 464: 8, 466: 8, 467: 8, 547: 8, 548: 8, 552: 4, 608: 8, 610: 5, 643: 7, 705: 8, 740: 5, 800: 8, 835: 8, 836: 8, 849: 4, 869: 7, 870: 7, 871: 2, 896: 8, 897: 8, 900: 6, 902: 6, 905: 8, 911: 8, 916: 2, 921: 8, 933: 8, 944: 8, 945: 8, 951: 8, 955: 4, 956: 8, 979: 2, 992: 8, 998: 5, 999: 7, 1000: 8, 1001: 8, 1017: 8, 1041: 8, 1042: 8, 1043: 8, 1044: 8, 1056: 8, 1059: 1, 1114: 8, 1161: 8, 1162: 8, 1163: 8, 1196: 8, 1227: 8, 1235: 8, 1279: 8, 1552: 8, 1553: 8, 1556: 8, 1557: 8, 1561: 8, 1562: 8, 1568: 8, 1569: 8, 1570: 8, 1571: 8, 1572: 8, 1584: 8, 1589: 8, 1592: 8, 1596: 8, 1597: 8, 1600: 8, 1664: 8, 1728: 8, 1779: 8, 1904: 8, 1912: 8, 1990: 8, 1998: 8
  },
  # Corolla LE 2017
  {
    36: 8, 37: 8, 170: 8, 180: 8, 186: 4, 426: 6, 452: 8, 464: 8, 466: 8, 467: 8, 547: 8, 548: 8, 552: 4, 608: 8, 610: 5, 643: 7, 705: 8, 740: 5, 800: 8, 835: 8, 836: 8, 849: 4, 869: 7, 870: 7, 871: 2, 896: 8, 897: 8, 900: 6, 902: 6, 905: 8, 911: 8, 916: 2, 921: 8, 933: 8, 944: 8, 945: 8, 951: 8, 955: 4, 956: 8, 979: 2, 998: 5, 999: 7, 1000: 8, 1001: 8, 1017: 8, 1041: 8, 1042: 8, 1043: 8, 1044: 8, 1056: 8, 1059: 1, 1114: 8, 1161: 8, 1162: 8, 1163: 8, 1196: 8, 1227: 8, 1235: 8, 1279: 8, 1552: 8, 1553: 8, 1556: 8, 1557: 8, 1561: 8, 1562: 8, 1568: 8, 1569: 8, 1570: 8, 1571: 8, 1572: 8, 1592: 8, 1596: 8, 1597: 8, 1600: 8, 1664: 8, 1779: 8, 1904: 8, 1912: 8, 1990: 8, 1998: 8, 2016: 8, 2017: 8, 2018: 8, 2019: 8, 2020: 8, 2021: 8, 2022: 8, 2023: 8, 2024: 8
  }],
  CAR.LEXUS_RXH: [{
    36: 8, 37: 8, 166: 8, 170: 8, 180: 8, 295: 8, 296: 8, 426: 6, 452: 8, 466: 8, 467: 8, 550: 8, 552: 4, 560: 7, 562: 6, 581: 5, 608: 8, 610: 5, 643: 7, 658: 8, 713: 8, 740: 5, 742: 8, 743: 8, 800: 8, 810: 2, 812: 3, 814: 8, 830: 7, 835: 8, 836: 8, 845: 5, 863: 8, 869: 7, 870: 7, 871: 2, 898: 8, 900: 6, 902: 6, 905: 8, 913: 8, 918: 8, 921: 8, 933: 8, 944: 8, 945: 8, 950: 8, 951: 8, 953: 8, 955: 8, 956: 8, 971: 7, 975: 6, 993: 8, 998: 5, 999: 7, 1000: 8, 1001: 8, 1005: 2, 1014: 8, 1017: 8, 1020: 8, 1041: 8, 1042: 8, 1044: 8, 1056: 8, 1059: 1, 1063: 8, 1071: 8, 1077: 8, 1082: 8, 1114: 8, 1161: 8, 1162: 8, 1163: 8, 1164: 8, 1165: 8, 1166: 8, 1167: 8, 1227: 8, 1228: 8, 1235: 8, 1237: 8, 1264: 8, 1279: 8, 1552: 8, 1553: 8, 1556: 8, 1557: 8, 1568: 8, 1570: 8, 1571: 8, 1572: 8, 1575: 8, 1595: 8, 1777: 8, 1779: 8, 1808: 8, 1810: 8, 1816: 8, 1818: 8, 1840: 8, 1848: 8, 1904: 8, 1912: 8, 1940: 8, 1941: 8, 1948: 8, 1949: 8, 1952: 8, 1956: 8, 1960: 8, 1964: 8, 1986: 8, 1990: 8, 1994: 8, 1998: 8, 2004: 8, 2012: 8
  }],
  CAR.CHR: [{
    36: 8, 37: 8, 170: 8, 180: 8, 186: 4, 426: 6, 452: 8, 464: 8, 466: 8, 467: 8, 544: 4, 550: 8, 552: 4, 562: 6, 608: 8, 610: 8, 643: 7, 705: 8, 740: 5, 800: 8, 810: 2, 812: 8, 830: 7, 835: 8, 836: 8, 869: 7, 870: 7, 871: 2, 921: 8, 944: 8, 945: 8, 951: 8, 955: 8, 956: 8, 976: 1, 1017: 8, 1020: 8, 1021: 8, 1041: 8, 1042: 8, 1044: 8, 1056: 8, 1059: 1, 1114: 8, 1161: 8, 1162: 8, 1163: 8, 1235: 8, 1279: 8, 1552: 8, 1553: 8, 1556: 8, 1557: 8, 1568: 8, 1570: 8, 1571: 8, 1572: 8, 1595: 8, 1745: 8, 1779: 8
  }],
  CAR.CHRH: [{
    36: 8, 37: 8, 166: 8, 170: 8, 180: 8, 295: 8, 296: 8, 426: 6, 452: 8, 466: 8, 467: 8, 550: 8, 552: 4, 560: 7, 562: 6, 581: 5, 608: 8, 610: 8, 614: 8, 643: 7, 658: 8, 713: 8, 740: 5, 800: 8, 810: 2, 812: 8, 814: 8, 829: 2, 830: 7, 835: 8, 836: 8, 845: 5, 869: 7, 870: 7, 871: 2, 898: 8, 900: 6, 902: 6, 905: 8, 913: 8, 918: 8, 921: 8, 933: 8, 944: 8, 945: 8, 950: 8, 951: 8, 953: 8, 955: 8, 956: 8, 971: 7, 975: 5, 993: 8, 998: 5, 999: 7, 1000: 8, 1001: 8, 1014: 8, 1017: 8, 1020: 8, 1021: 8, 1041: 8, 1042: 8, 1044: 8, 1056: 8, 1057: 8, 1059: 1, 1071: 8, 1076: 8, 1077: 8, 1082: 8, 1083: 8, 1114: 8, 1161: 8, 1162: 8, 1163: 8, 1175: 8, 1228: 8, 1235: 8, 1237: 8, 1279: 8, 1552: 8, 1553: 8, 1556: 8, 1557: 8, 1568: 8, 1570: 8, 1571: 8, 1572: 8, 1595: 8, 1745: 8, 1779: 8, 1904: 8, 1912: 8, 1990: 8, 1998: 8
  }],
  CAR.CAMRY: [
  #CAR.CAMRYLE_2018: [{ (MINE)
  {
    36: 8, 37: 8, 119: 6, 170: 8, 180: 8, 186: 4, 426: 6, 452: 8, 464: 8, 466: 8, 467: 8, 544: 4, 550: 8, 552: 4, 562: 6, 608: 8, 610: 8, 643: 7, 705: 8, 728: 8, 740: 5, 761: 8, 764: 8, 800: 8, 824: 8, 830: 7, 835: 8, 836: 8, 869: 7, 870: 7, 898: 8, 905: 8, 921: 8, 933: 8, 944: 8, 945: 8, 951: 8, 955: 8, 956: 8, 976: 1, 1002: 8, 1017: 8, 1020: 8, 1041: 8, 1042: 8, 1056: 8, 1059: 1, 1076: 8, 1077: 8, 1114: 8, 1164: 8, 1165: 8, 1166: 8, 1167: 8, 1279: 8, 1541: 8, 1552: 8, 1553: 8, 1556: 8, 1568: 8, 1570: 8, 1571: 8, 1572: 8, 1595: 8, 1745: 8, 1779: 8, 1904: 8
  },
  #XLE and LE
  {
    36: 8, 37: 8, 119: 6, 170: 8, 180: 8, 186: 4, 426: 6, 452: 8, 464: 8, 466: 8, 467: 8, 544: 4, 550: 8, 552: 4, 562: 6, 608: 8, 610: 8, 643: 7, 658: 8, 705: 8, 728: 8, 740: 5, 761: 8, 764: 8, 800: 8, 810: 2, 812: 8, 814: 8, 818: 8, 822: 8, 824: 8, 830: 7, 835: 8, 836: 8, 869: 7, 870: 7, 871: 2, 888: 8, 889: 8, 891: 8, 898: 8, 900: 6, 902: 6, 905: 8, 918: 8, 921: 8, 933: 8, 934: 8, 935: 8, 944: 8, 945: 8, 951: 8, 955: 8, 956: 8, 976: 1, 983: 8, 984: 8, 998: 5, 999: 7, 1000: 8, 1001: 8, 1002: 8, 1011: 8, 1014: 8, 1017: 8, 1020: 8, 1041: 8, 1042: 8, 1044: 8, 1056: 8, 1059: 1, 1076: 8, 1077: 8, 1082: 8, 1114: 8, 1161: 8, 1162: 8, 1163: 8, 1164: 8, 1165: 8, 1166: 8, 1167: 8, 1228: 8, 1235: 8, 1237: 8, 1263: 8, 1264: 8, 1279: 8, 1412: 8, 1541: 8, 1552: 8, 1553: 8, 1556: 8, 1557: 8, 1568: 8, 1570: 8, 1571: 8, 1572: 8, 1595: 8, 1745: 8, 1779: 8, 1786: 8, 1787: 8, 1788: 8, 1789: 8, 1808: 8, 1816: 8, 1904: 8, 1912: 8, 1990: 8, 1998: 8
  },
  #XLE and LE (NOT MINE!)
  {
    36: 8, 37: 8, 114: 5, 119: 6, 120: 4, 170: 8, 180: 8, 186: 4, 426: 6, 452: 8, 464: 8, 466: 8, 467: 8, 544: 4, 550: 8, 552: 4, 562: 6, 608: 8, 610: 8, 643: 7, 658: 8, 705: 8, 728: 8, 761: 8, 764: 8, 800: 8, 810: 2, 812: 8, 814: 8, 818: 8, 822: 8, 824: 8, 830: 7, 835: 8, 836: 8, 869: 7, 870: 7, 888: 8, 889: 8, 891: 8, 898: 8, 900: 6, 902: 6, 905: 8, 918: 8, 921: 8, 933: 8, 934: 8, 935: 8, 944: 8, 945: 8, 951: 8, 955: 8, 956: 8, 976: 1, 983: 8, 984: 8, 998: 5, 999: 7, 1000: 8, 1001: 8, 1002: 8, 1011: 8, 1014: 8, 1017: 8, 1020: 8, 1041: 8, 1056: 8, 1059: 1, 1076: 8, 1077: 8, 1082: 8, 1114: 8, 1164: 8, 1165: 8, 1166: 8, 1167: 8, 1228: 8, 1237: 8, 1263: 8, 1264: 8, 1279: 8, 1412: 8, 1541: 8, 1552: 8, 1553: 8, 1556: 8, 1557: 8, 1568: 8, 1570: 8, 1571: 8, 1572: 8, 1595: 8, 1745: 8, 1779: 8, 1786: 8, 1787: 8, 1788: 8, 1789: 8, 1808: 8, 1816: 8, 1904: 8, 1912: 8, 1990: 8, 1998: 8
  }],
  CAR.CAMRYH: [
  #LE
  {
    36: 8, 37: 8, 166: 8, 170: 8, 180: 8, 295: 8, 296: 8, 426: 6, 452: 8, 466: 8, 467: 8, 550: 8, 552: 4, 560: 7, 562: 6, 581: 5, 608: 8, 610: 8, 643: 7, 713: 8, 728: 8, 740: 5, 761: 8, 764: 8, 800: 8, 810: 2, 812: 8, 824: 8, 829: 2, 830: 7, 835: 8, 836: 8, 869: 7, 870: 7, 871: 2, 898: 8, 900: 6, 902: 6, 905: 8, 921: 8, 933: 8, 934: 8, 935: 8, 944: 8, 945: 8, 950: 8, 951: 8, 953: 8, 955: 8, 956: 8, 971: 7, 975: 5, 993: 8, 998: 5, 999: 7, 1000: 8, 1001: 8, 1002: 8, 1017: 8, 1020: 8, 1041: 8, 1042: 8, 1044: 8, 1056: 8, 1057: 8, 1059: 1, 1071: 8, 1076: 8, 1077: 8, 1084: 8, 1085: 8, 1086: 8, 1114: 8, 1132: 8, 1161: 8, 1162: 8, 1163: 8, 1164: 8, 1165: 8, 1166: 8, 1167: 8, 1235: 8, 1279: 8, 1541: 8, 1552: 8, 1553: 8, 1556: 8, 1557: 8, 1568: 8, 1570: 8, 1571: 8, 1572:8, 1595: 8, 1745: 8, 1779: 8, 1786: 8, 1787: 8, 1788: 8, 1789: 8, 1904: 8, 1912: 8, 1990: 8, 1998: 8
  }],
  CAR.HIGHLANDER: [{
    36: 8, 37: 8, 114: 5, 119: 6, 120: 4, 170: 8, 180: 8, 186: 4, 238: 4, 355: 5, 426: 6, 452: 8, 464: 8, 466: 8, 467: 8, 544: 4, 545: 5, 550: 8, 552: 4, 608: 8, 610: 5, 643: 7, 705: 8, 725: 2, 740: 5, 800: 8, 835: 8, 836: 8, 849: 4, 869: 7, 870: 7, 871: 2, 896: 8, 900: 6, 902: 6, 905: 8, 911: 8, 916: 3, 921: 8, 922: 8, 933: 8, 944: 8, 945: 8, 951: 8, 955: 8, 956: 8, 979: 2, 998: 5, 999: 7, 1000: 8, 1001: 8, 1008: 2, 1014: 8, 1017: 8, 1020: 8, 1041: 8, 1042: 8, 1043: 8, 1044: 8, 1056: 8, 1059: 1, 1114: 8, 1161: 8, 1162: 8, 1163: 8, 1176: 8, 1177: 8, 1178: 8, 1179: 8, 1180: 8, 1181: 8, 1182: 8, 1183: 8, 1189: 8, 1190: 8, 1191: 8, 1192: 8, 1196: 8, 1197: 8, 1198: 8, 1199: 8, 1206: 8, 1207: 8, 1212: 8, 1227: 8, 1235: 8, 1237: 8, 1279: 8, 1408: 8, 1409: 8, 1410: 8, 1552: 8, 1553: 8, 1554: 8, 1556: 8, 1557: 8, 1561: 8, 1562: 8, 1568: 8, 1569: 8, 1570: 8, 1571: 8, 1572: 8, 1584: 8, 1589: 8, 1592: 8, 1593: 8, 1595: 8, 1599: 8, 1656: 8, 1666: 8, 1667: 8, 1728: 8, 1745: 8, 1779: 8, 1904: 8, 1912: 8, 1990: 8, 1998: 8
  },
  # 2017 Highlander Limited
  {
    36: 8, 37: 8, 114: 5, 119: 6, 120: 4, 170: 8, 180: 8, 186: 4, 238: 4, 355: 5, 426: 6, 452: 8, 464: 8, 466: 8, 467: 8, 544: 4, 545: 5, 550: 8, 552: 4, 608: 8, 610: 5, 643: 7, 705: 8, 725: 2, 740: 5, 800: 8, 835: 8, 836: 8, 849: 4, 869: 7, 870: 7, 871: 2, 896: 8, 900: 6, 902: 6, 905: 8, 911: 8, 916: 3, 918: 7, 921: 8, 922: 8, 933: 8, 944: 8, 945: 8, 951: 8, 955: 8, 956: 8, 979: 2, 998: 5, 999: 7, 1000: 8, 1001: 8, 1005: 2, 1008: 2, 1014: 8, 1017: 8, 1020: 8, 1041: 8, 1042: 8, 1043: 8, 1044: 8, 1056: 8, 1059: 1, 1114: 8, 1161: 8, 1162: 8, 1163: 8, 1176: 8, 1177: 8, 1178: 8, 1179: 8, 1180: 8, 1181: 8, 1182: 8, 1183: 8, 1189: 8, 1190: 8, 1191: 8, 1192: 8, 1196: 8, 1197: 8, 1198: 8, 1199: 8, 1206: 8, 1207: 8, 1212: 8, 1227: 8, 1235: 8, 1237: 8, 1264: 8, 1279: 8, 1408: 8, 1409: 8, 1410: 8, 1552: 8, 1553: 8, 1554: 8, 1556: 8, 1557: 8, 1561: 8, 1562: 8, 1568: 8, 1569: 8, 1570: 8, 1571: 8, 1572: 8, 1584: 8, 1589: 8, 1592: 8, 1593: 8, 1595: 8, 1599: 8, 1656: 8, 1728: 8, 1745: 8, 1779: 8, 1904: 8, 1912: 8, 1990: 8, 1998: 8
  }],
  CAR.HIGHLANDERH: [{
    36: 8, 37: 8, 170: 8, 180: 8, 296: 8, 426: 6, 452: 8, 466: 8, 467: 8, 550: 8, 552: 4, 560: 7, 581: 5, 608: 8, 610: 5, 643: 7, 713: 8, 740: 5, 800: 8, 835: 8, 836: 8, 849: 4, 869: 7, 870: 7, 871: 2, 896: 8, 897: 8, 900: 6, 902: 6, 905: 8, 911: 8, 916: 3, 918: 7, 921: 8, 933: 8, 944: 8, 945: 8, 950: 8, 951: 8, 953: 3, 955: 8, 956: 8, 979: 2, 998: 5, 999: 7, 1000: 8, 1001: 8, 1005: 2, 1014: 8, 1017: 8, 1020: 8, 1041: 8, 1042: 8, 1043: 8, 1044: 8, 1056: 8, 1059: 1, 1112: 8, 1114: 8, 1161: 8, 1162: 8, 1163: 8, 1176: 8, 1177: 8, 1178: 8, 1179: 8, 1180: 8, 1181: 8, 1184: 8, 1185: 8, 1186: 8, 1189: 8, 1190: 8, 1191: 8, 1192: 8, 1196: 8, 1197: 8, 1198: 8, 1199: 8, 1206: 8, 1212: 8, 1227: 8, 1232: 8, 1235: 8, 1237: 8, 1264: 8, 1279: 8, 1552: 8, 1553: 8, 1554: 8, 1556: 8, 1557: 8, 1561: 8, 1562: 8, 1568: 8, 1569: 8, 1570: 8, 1571: 8, 1572: 8, 1584: 8, 1589: 8, 1592: 8, 1593: 8, 1595: 8, 1599: 8, 1656: 8, 1728: 8, 1745: 8, 1779: 8, 1904: 8, 1912: 8, 1990: 8, 1998: 8
  }],
}

STEER_THRESHOLD = 100

DBC = {
  CAR.RAV4H: dbc_dict('toyota_rav4_hybrid_2017_pt_generated', 'toyota_prius_2017_adas'),
  CAR.RAV4: dbc_dict('toyota_rav4_2017_pt_generated', 'toyota_prius_2017_adas'),
  CAR.PRIUS: dbc_dict('toyota_prius_2017_pt_generated', 'toyota_prius_2017_adas'),
  CAR.COROLLA: dbc_dict('toyota_corolla_2017_pt_generated', 'toyota_prius_2017_adas'),
  CAR.LEXUS_RXH: dbc_dict('lexus_rx_hybrid_2017_pt_generated', 'toyota_prius_2017_adas'),
  CAR.CHR: dbc_dict('toyota_chr_2018_pt_generated', 'toyota_prius_2017_adas'),
  CAR.CHRH: dbc_dict('toyota_chr_hybrid_2018_pt_generated', 'toyota_prius_2017_adas'),
  CAR.CAMRY: dbc_dict('toyota_chr_2018_pt_generated', 'toyota_prius_2017_adas'),
  CAR.CAMRYH: dbc_dict('toyota_camry_hybrid_2018_pt_generated', 'toyota_prius_2017_adas'),
  CAR.HIGHLANDER: dbc_dict('toyota_highlander_2017_pt_generated', 'toyota_prius_2017_adas'),
  CAR.HIGHLANDERH: dbc_dict('toyota_highlander_hybrid_2018_pt_generated', 'toyota_prius_2017_adas'),
}

NO_DSU_CAR = [CAR.CHR, CAR.CHRH, CAR.CAMRY, CAR.CAMRYH]
