from cereal import car
from selfdrive.car import dbc_dict

VisualAlert = car.CarControl.HUDControl.VisualAlert

# Car button codes
class CruiseButtons:
  RES_ACCEL   = 4
  DECEL_SET   = 3
  CANCEL      = 2
  MAIN        = 1

class AH:
  #[alert_idx, value]
  # See dbc files for info on values"
  NONE           = [0, 0]
  FCW            = [1, 1]
  STEER          = [2, 1]
  BRAKE_PRESSED  = [3, 10]
  GEAR_NOT_D     = [4, 6]
  SEATBELT       = [5, 5]
  SPEED_TOO_HIGH = [6, 8]

VISUAL_HUD = {
  VisualAlert.none: AH.NONE,
  VisualAlert.fcw: AH.FCW,
  VisualAlert.steerRequired: AH.STEER,
  VisualAlert.brakePressed: AH.BRAKE_PRESSED,
  VisualAlert.wrongGear: AH.GEAR_NOT_D,
  VisualAlert.seatbeltUnbuckled: AH.SEATBELT,
  VisualAlert.speedTooHigh: AH.SPEED_TOO_HIGH}

class ECU:
  CAM = 0

class CAR:
  ACCORD = "HONDA ACCORD 2018 SPORT 2T"
  ACCORD_15 = "HONDA ACCORD 2018 LX 1.5T"
  ACCORDH = "HONDA ACCORD 2018 HYBRID TOURING"
  CIVIC = "HONDA CIVIC 2016 TOURING"
  CIVIC_BOSCH = "HONDA CIVIC HATCHBACK 2017 SEDAN/COUPE 2019"
  ACURA_ILX = "ACURA ILX 2016 ACURAWATCH PLUS"
  CRV = "HONDA CR-V 2016 TOURING"
  CRV_5G = "HONDA CR-V 2017 EX"
  CRV_HYBRID = "HONDA CR-V 2019 HYBRID"
  FIT = "HONDA FIT 2018 EX"
  ODYSSEY = "HONDA ODYSSEY 2018 EX-L"
  ODYSSEY_CHN = "HONDA ODYSSEY 2019 EXCLUSIVE CHN"
  ACURA_RDX = "ACURA RDX 2018 ACURAWATCH PLUS"
  PILOT = "HONDA PILOT 2017 TOURING"
  PILOT_2019 = "HONDA PILOT 2019 ELITE"
  RIDGELINE = "HONDA RIDGELINE 2017 BLACK EDITION"

# diag message that in some Nidec cars only appear with 1s freq if VIN query is performed
DIAG_MSGS = {1600: 5, 1601: 8}

FINGERPRINTS = {
  CAR.ACCORD: [{
    148: 8, 228: 5, 304: 8, 330: 8, 344: 8, 380: 8, 399: 7, 419: 8, 420: 8, 427: 3, 432: 7, 441: 5, 446: 3, 450: 8, 464: 8, 477: 8, 479: 8, 495: 8, 545: 6, 662: 4, 773: 7, 777: 8, 780: 8, 804: 8, 806: 8, 808: 8, 829: 5, 862: 8, 884: 8, 891: 8, 927: 8, 929: 8, 1302: 8, 1600: 5, 1601: 8, 1652: 8
  }],
  CAR.ACCORD_15: [{
    148: 8, 228: 5, 304: 8, 330: 8, 344: 8, 380: 8, 399: 7, 401: 8, 420: 8, 427: 3, 432: 7, 441: 5, 446: 3, 450: 8, 464: 8, 477: 8, 479: 8, 495: 8, 545: 6, 662: 4, 773: 7, 777: 8, 780: 8, 804: 8, 806: 8, 808: 8, 829: 5, 862: 8, 884: 8, 891: 8, 927: 8, 929: 8, 1302: 8, 1600: 5, 1601: 8, 1652: 8
  }],
  CAR.ACCORDH: [{
    148: 8, 228: 5, 304: 8, 330: 8, 344: 8, 380: 8, 387: 8, 388: 8, 399: 7, 419: 8, 420: 8, 427: 3, 432: 7, 441: 5, 450: 8, 464: 8, 477: 8, 479: 8, 495: 8, 525: 8, 545: 6, 662: 4, 773: 7, 777: 8, 780: 8, 804: 8, 806: 8, 808: 8, 829: 5, 862: 8, 884: 8, 891: 8, 927: 8, 929: 8, 1302: 8, 1600: 5, 1601: 8, 1652: 8
  }],
  CAR.ACURA_ILX: [{
    57: 3, 145: 8, 228: 5, 304: 8, 316: 8, 342: 6, 344: 8, 380: 8, 398: 3, 399: 7, 419: 8, 420: 8, 422: 8, 428: 8, 432: 7, 464: 8, 476: 4, 490: 8, 506: 8, 512: 6, 513: 6, 542: 7, 545: 4, 597: 8, 660: 8, 773: 7, 777: 8, 780: 8, 800: 8, 804: 8, 808: 8, 819: 7, 821: 5, 829: 5, 882: 2, 884: 7, 887: 8, 888: 8, 892: 8, 923: 2, 929: 4, 983: 8, 985: 3, 1024: 5, 1027: 5, 1029: 8, 1030: 5, 1034: 5, 1036: 8, 1039: 8, 1057: 5, 1064: 7, 1108: 8, 1365: 5,
  }],
  # Acura RDX w/ Added Comma Pedal Support (512L & 513L)
  CAR.ACURA_RDX: [{
    57: 3, 145: 8, 229: 4, 308: 5, 316: 8, 342: 6, 344: 8, 380: 8, 392: 6, 398: 3, 399: 6, 404: 4, 420: 8, 422: 8, 426: 8, 432: 7, 464: 8, 474: 5, 476: 4, 487: 4, 490: 8, 506: 8, 512: 6, 513: 6, 542: 7, 545: 4, 597: 8, 660: 8, 773: 7, 777: 8, 780: 8, 800: 8, 804: 8, 808: 8, 819: 7, 821: 5, 829: 5, 882: 2, 884: 7, 887: 8, 888: 8, 892: 8, 923: 2, 929: 4, 963: 8, 965: 8, 966: 8, 967: 8, 983: 8, 985: 3, 1024: 5, 1027: 5, 1029: 8, 1033: 5, 1034: 5, 1036: 8, 1039: 8, 1057: 5, 1064: 7, 1108: 8, 1365: 5, 1424: 5, 1729: 1
  }],
  CAR.CIVIC: [{
    57: 3, 148: 8, 228: 5, 304: 8, 330: 8, 344: 8, 380: 8, 399: 7, 401: 8, 420: 8, 427: 3, 428: 8, 432: 7, 450: 8, 464: 8, 470: 2, 476: 7, 487: 4, 490: 8, 493: 5, 506: 8, 512: 6, 513: 6, 545: 6, 597: 8, 662: 4, 773: 7, 777: 8, 780: 8, 795: 8, 800: 8, 804: 8, 806: 8, 808: 8, 829: 5, 862: 8, 884: 8, 891: 8, 892: 8, 927: 8, 929: 8, 985: 3, 1024: 5, 1027: 5, 1029: 8, 1036: 8, 1039: 8, 1108: 8, 1302: 8, 1322: 5, 1361: 5, 1365: 5, 1424: 5, 1633: 8,
  }],
  CAR.CIVIC_BOSCH: [{
  # 2017 Civic Hatchback EX, 2019 Civic Sedan Touring Canadian, and 2018 Civic Hatchback Executive Premium 1.0L CVT European
    57: 3, 148: 8, 228: 5, 304: 8, 330: 8, 344: 8, 380: 8, 399: 7, 401: 8, 420: 8, 427: 3, 428: 8, 432: 7, 441: 5, 450: 8, 460: 3, 464: 8, 470: 2, 476: 7, 477: 8, 479: 8, 490: 8, 493: 5, 495: 8, 506: 8, 545: 6, 597: 8, 662: 4, 773: 7, 777: 8, 780: 8, 795: 8, 800: 8, 804: 8, 806: 8, 808: 8, 829: 5, 862: 8, 884: 8, 891: 8, 892: 8, 927: 8, 929: 8, 985: 3, 1024: 5, 1027: 5, 1029: 8, 1036: 8, 1039: 8, 1108: 8, 1302: 8, 1322: 5, 1361: 5, 1365: 5, 1424: 5, 1600: 5, 1601: 8, 1625: 5, 1629: 5, 1633: 8,
  },
  # 2017 Civic Hatchback LX
  {
    57: 3, 148: 8, 228: 5, 304: 8, 330: 8, 344: 8, 380: 8, 399: 7, 401: 8, 420: 8, 423: 2, 427: 3, 428: 8, 432: 7, 441: 5, 450: 8, 464: 8, 470: 2, 476: 7, 477: 8, 479: 8, 490: 8, 493: 5, 495: 8, 506: 8, 545: 6, 597: 8, 662: 4, 773: 7, 777: 8, 780: 8, 795: 8, 800: 8, 804: 8, 806: 8, 808: 8, 815: 8, 825: 4, 829: 5, 846: 8, 862: 8, 881: 8, 882: 4, 884: 8, 888: 8, 891: 8, 892: 8, 918: 7, 927: 8, 929: 8, 983: 8, 985: 3, 1024: 5, 1027: 5, 1029: 8, 1036: 8, 1039: 8, 1064: 7, 1092: 1, 1108: 8, 1125: 8, 1127: 2, 1296: 8, 1302: 8, 1322: 5, 1361: 5, 1365: 5, 1424: 5, 1600: 5, 1601: 8, 1633: 8
