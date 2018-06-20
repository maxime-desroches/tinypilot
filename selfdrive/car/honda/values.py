from selfdrive.car import dbc_dict

# Car button codes
class CruiseButtons:
  RES_ACCEL   = 4
  DECEL_SET   = 3
  CANCEL      = 2
  MAIN        = 1


#car chimes: enumeration from dbc file. Chimes are for alerts and warnings
class CM:
  MUTE = 0
  SINGLE = 3
  DOUBLE = 4
  REPEATED = 1
  CONTINUOUS = 2


#car beepss: enumeration from dbc file. Beeps are for activ and deactiv
class BP:
  MUTE = 0
  SINGLE = 3
  TRIPLE = 2
  REPEATED = 1

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


class CAR:
  ACCORD = "HONDA ACCORD 2018 SPORT 2T"
  CIVIC = "HONDA CIVIC 2016 TOURING"
  CIVIC_HATCH = "HONDA CIVIC HATCHBACK 2017 EX"
  ACURA_ILX = "ACURA ILX 2016 ACURAWATCH PLUS"
  CRV = "HONDA CR-V 2016 TOURING"
  CRV_5G = "HONDA CR-V 2017 EX"
  ODYSSEY = "HONDA ODYSSEY 2018 EX-L"
  ACURA_RDX = "ACURA RDX 2018 ACURAWATCH PLUS"
  PILOT = "HONDA PILOT 2017 TOURING"
  RIDGELINE = "HONDA RIDGELINE 2017 BLACK EDITION"


FINGERPRINTS = {
  CAR.ACCORD: [{
    148: 8, 228: 5, 304: 8, 330: 8, 344: 8, 380: 8, 399: 7, 419: 8, 420: 8, 427: 3, 432: 7, 441: 5, 446: 3, 450: 8, 464: 8, 477: 8, 479: 8, 495: 8, 545: 6, 662: 4, 773: 7, 777: 8, 780: 8, 804: 8, 806: 8, 808: 8, 829: 5, 862: 8, 884: 8, 891: 8, 927: 8, 929: 8, 1302: 8, 1600: 5, 1601: 8, 1652: 8 
  }], 
  CAR.ACURA_ILX: [{
    57L: 3, 145L: 8, 228L: 5, 304L: 8, 316L: 8, 342L: 6, 344L: 8, 380L: 8, 398L: 3, 399L: 7, 419L: 8, 420L: 8, 422L: 8, 428L: 8, 432L: 7, 464L: 8, 476L: 4, 490L: 8, 506L: 8, 512L: 6, 513L: 6, 542L: 7, 545L: 4, 597L: 8, 660L: 8, 773L: 7, 777L: 8, 780L: 8, 800L: 8, 804L: 8, 808L: 8, 819L: 7, 821L: 5, 829L: 5, 882L: 2, 884L: 7, 887L: 8, 888L: 8, 892L: 8, 923L: 2, 929L: 4, 983L: 8, 985L: 3, 1024L: 5, 1027L: 5, 1029L: 8, 1030L: 5, 1034L: 5, 1036L: 8, 1039L: 8, 1057L: 5, 1064L: 7, 1108L: 8, 1365L: 5,
  }],
  CAR.ACURA_RDX: [{
    57L: 3, 145L: 8, 229L: 4, 308L: 5, 316L: 8, 342L: 6, 344L: 8, 380L: 8, 392L: 6, 398L: 3, 399L: 6, 404L: 4, 420L: 8, 422L: 8, 426L: 8, 432L: 7, 464L: 8, 474L: 5, 476L: 4, 487L: 4, 490L: 8, 506L: 8, 512L: 6, 513L: 6, 542L: 7, 545L: 4, 597L: 8, 660L: 8, 773L: 7, 777L: 8, 780L: 8, 800L: 8, 804L: 8, 808L: 8, 819L: 7, 821L: 5, 829L: 5, 882L: 2, 884L: 7, 887L: 8, 888L: 8, 892L: 8, 923L: 2, 929L: 4, 963L: 8, 965L: 8, 966L: 8, 967L: 8, 983L: 8, 985L: 3, 1024L: 5, 1027L: 5, 1029L: 8, 1033L: 5, 1034L: 5, 1036L: 8, 1039L: 8, 1057L: 5, 1064L: 7, 1108L: 8, 1365L: 5, 1424L: 5, 1729L: 1
  }],
  CAR.CIVIC: [{
    57L: 3, 148L: 8, 228L: 5, 304L: 8, 330L: 8, 344L: 8, 380L: 8, 399L: 7, 401L: 8, 420L: 8, 427L: 3, 428L: 8, 432L: 7, 450L: 8, 464L: 8, 470L: 2, 476L: 7, 487L: 4, 490L: 8, 493L: 5, 506L: 8, 512L: 6, 513L: 6, 545L: 6, 597L: 8, 662L: 4, 773L: 7, 777L: 8, 780L: 8, 795L: 8, 800L: 8, 804L: 8, 806L: 8, 808L: 8, 829L: 5, 862L: 8, 884L: 8, 891L: 8, 892L: 8, 927L: 8, 929L: 8, 985L: 3, 1024L: 5, 1027L: 5, 1029L: 8, 1036L: 8, 1039L: 8, 1108L: 8, 1302L: 8, 1322L: 5, 1361L: 5, 1365L: 5, 1424L: 5, 1633L: 8,
  }],
  CAR.CIVIC_HATCH: [{
    57L: 3, 148L: 8, 228L: 5, 304L: 8, 330L: 8, 344L: 8, 380L: 8, 399L: 7, 401L: 8, 420L: 8, 427L: 3, 428L: 8, 432L: 7, 441L: 5, 450L: 8, 464L: 8, 470L: 2, 476L: 7, 477L: 8, 479L: 8, 490L: 8, 493L: 5, 495L: 8, 506L: 8, 545L: 6, 597L: 8, 662L: 4, 773L: 7, 777L: 8, 780L: 8, 795L: 8, 800L: 8, 804L: 8, 806L: 8, 808L: 8, 829L: 5, 862L: 8, 884L: 8, 891L: 8, 892L: 8, 927L: 8, 929L: 8, 985L: 3, 1024L: 5, 1027L: 5, 1029L: 8, 1036L: 8, 1039L: 8, 1108L: 8, 1302L: 8, 1322L: 5, 1361L: 5, 1365L: 5, 1424L: 5, 1600L: 5, 1601L: 8, 1633L: 8 
  }], 
  CAR.CRV: [{
    57L: 3, 145L: 8, 316L: 8, 340L: 8, 342L: 6, 344L: 8, 380L: 8, 398L: 3, 399L: 6, 401L: 8, 404L: 4, 420L: 8, 422L: 8, 426L: 8, 432L: 7, 464L: 8, 474L: 5, 476L: 4, 487L: 4, 490L: 8, 493L: 3, 506L: 8, 507L: 1, 512L: 6, 513L: 6, 542L: 7, 545L: 4, 597L: 8, 660L: 8, 661L: 4, 773L: 7, 777L: 8, 780L: 8, 800L: 8, 804L: 8, 808L: 8, 829L: 5, 882L: 2, 884L: 7, 888L: 8, 891L: 8, 892L: 8, 923L: 2, 929L: 8, 983L: 8, 985L: 3, 1024L: 5, 1027L: 5, 1029L: 8, 1033L: 5, 1036L: 8, 1039L: 8, 1057L: 5, 1064L: 7, 1108L: 8, 1125L: 8, 1296L: 8, 1365L: 5, 1424L: 5, 1600L: 5, 1601L: 8, 
  }],
  CAR.CRV_5G: [{
    57L: 3, 148L: 8, 199L: 4, 228L: 5, 231L: 5, 232L: 7, 304L: 8, 330L: 8, 340L: 8, 344L: 8, 380L: 8, 399L: 7, 401L: 8, 420L: 8, 423L: 2, 427L: 3, 428L: 8, 432L: 7, 441L: 5, 446L: 3, 450L: 8, 464L: 8, 467L: 2, 469L: 3, 470L: 2, 474L: 8, 476L: 7, 477L: 8, 479L: 8, 490L: 8, 493L: 5, 495L: 8, 507L: 1, 545L: 6, 597L: 8, 661L: 4, 662L: 4, 773L: 7, 777L: 8, 780L: 8, 795L: 8, 800L: 8, 804L: 8, 806L: 8, 808L: 8, 814L: 4, 815L: 8, 817L: 4, 825L: 4, 829L: 5, 862L: 8, 881L: 8, 882L: 4, 884L: 8, 888L: 8, 891L: 8, 927L: 8, 918L: 7, 929L: 8, 983L: 8, 985L: 3, 1024L: 5, 1027L: 5, 1029L: 8, 1036L: 8, 1039L: 8, 1064L: 7, 1108L: 8, 1092L: 1, 1115L: 4, 1125L: 8, 1127L: 2, 1296L: 8, 1302L: 8, 1322L: 5, 1361L: 5, 1365L: 5, 1424L: 5, 1600L: 5, 1601L: 8, 1618L: 5, 1633L: 8, 1670L: 5
  }], 
  CAR.ODYSSEY: [{
    57L: 3, 148L: 8, 228L: 5, 229L: 4, 316L: 8, 342L: 6, 344L: 8, 380L: 8, 399L: 7, 411L: 5, 419L: 8, 420L: 8, 427L: 3, 432L: 7, 450L: 8, 463L: 8, 464L: 8, 476L: 4, 490L: 8, 506L: 8, 512L: 6, 513L: 6, 542L: 7, 545L: 6, 597L: 8, 662L: 4, 773L: 7, 777L: 8, 780L: 8, 795L: 8, 800L: 8, 804L: 8, 806L: 8, 808L: 8, 817L: 4, 819L: 7, 821L: 5, 825L: 4, 829L: 5, 837L: 5, 856L: 7, 862L: 8, 871L: 8, 881L: 8, 882L: 4, 884L: 8, 891L: 8, 892L: 8, 905L: 8, 923L: 2, 927L: 8, 929L: 8, 963L: 8, 965L: 8, 966L: 8, 967L: 8, 983L: 8, 985L: 3, 1029L: 8, 1036L: 8, 1052L: 8, 1064L: 7, 1088L: 8, 1089L: 8, 1092L: 1, 1108L: 8, 1110L: 8, 1125L: 8, 1296L: 8, 1302L: 8, 1600L: 5, 1601L: 8, 1612L: 5, 1613L: 5, 1614L: 5, 1615L: 8, 1616L: 5, 1619L: 5, 1623L: 5, 1668L: 5
  },
  # Odyssey Elite
  {
    57L: 3, 148L: 8, 228L: 5, 229L: 4, 304L: 8, 342L: 6, 344L: 8, 380L: 8, 399L: 7, 411L: 5, 419L: 8, 420L: 8, 427L: 3, 432L: 7, 440L: 8, 450L: 8, 463L: 8, 464L: 8, 476L: 4, 490L: 8, 506L: 8, 507L: 1, 512L: 6, 513L: 6, 542L: 7, 545L: 6, 597L: 8, 662L: 4, 773L: 7, 777L: 8, 780L: 8, 795L: 8, 800L: 8, 804L: 8, 806L: 8, 808L: 8, 817L: 4, 819L: 7, 821L: 5, 825L: 4, 829L: 5, 837L: 5, 856L: 7, 862L: 8, 871L: 8, 881L: 8, 882L: 4, 884L: 8, 891L: 8, 892L: 8, 905L: 8, 923L: 2, 927L: 8, 929L: 8, 963L: 8, 965L: 8, 966L: 8, 967L: 8, 983L: 8, 985L: 3, 1029L: 8, 1036L: 8, 1052L: 8, 1064L: 7, 1088L: 8, 1089L: 8, 1092L: 1, 1108L: 8, 1110L: 8, 1125L: 8, 1296L: 8, 1302L: 8, 1600L: 5, 1601L: 8, 1612L: 5, 1613L: 5, 1614L: 5, 1616L: 5, 1619L: 5, 1623L: 5, 1668L: 5
  }],
  # Includes 2017 Touring and 2016 EX-L messaging.
  CAR.PILOT: [{
    57L: 3, 145L: 8, 228L: 5, 229L: 4, 308L: 5, 316L: 8, 334L: 8, 339L: 7, 342L: 6, 344L: 8, 379L: 8, 380L: 8, 392L: 6, 399L: 7, 419L: 8, 420L: 8, 422L: 8, 425L: 8, 426L: 8, 427L: 3, 432L: 7, 463L: 8, 464L: 8, 476L: 4, 490L: 8, 506L: 8, 507L: 1, 512L: 6, 513L: 6, 538L: 3, 542L: 7, 545L: 5, 546L: 3, 597L: 8, 660L: 8, 773L: 7, 777L: 8, 780L: 8, 795L: 8, 800L: 8, 804L: 8, 808L: 8, 819L: 7, 821L: 5, 829L: 5, 837L: 5, 856L: 7, 871L: 8, 882L: 2, 884L: 7, 891L: 8, 892L: 8, 923L: 2, 929L: 8, 963L: 8, 965L: 8, 966L: 8, 967L: 8, 983L: 8, 985L: 3, 1027L: 5, 1029L: 8, 1036L: 8, 1039L: 8, 1064L: 7, 1088L: 8, 1089L: 8, 1108L: 8, 1125L: 8, 1296L: 8, 1424L: 5, 1600L: 5, 1601L: 8, 1612L: 5, 1613L: 5, 1616L: 5, 1618L: 5, 1668L: 5
  }],
  CAR.RIDGELINE: [{
    57L: 3, 145L: 8, 228L: 5, 229L: 4, 308L: 5, 316L: 8, 339L: 7, 342L: 6, 344L: 8, 380L: 8, 392L: 6, 399L: 7, 419L: 8, 420L: 8, 422L: 8, 425L: 8, 426L: 8, 427L: 3, 432L: 7, 464L: 8, 471L: 3, 476L: 4, 490L: 8, 506L: 8, 512L: 6, 513L: 6, 545L: 5, 546L: 3, 597L: 8, 660L: 8, 773L: 7, 777L: 8, 780L: 8, 795L: 8, 800L: 8, 804L: 8, 808L: 8, 819L: 7, 821L: 5, 829L: 5, 871L: 8, 882L: 2, 884L: 7, 892L: 8, 923L: 2, 927L: 8, 929L: 8, 963L: 8, 965L: 8, 966L: 8, 967L: 8, 983L: 8, 985L: 3, 1027L: 5, 1029L: 8, 1036L: 8, 1039L: 8, 1064L: 7, 1088L: 8, 1089L: 8, 1108L: 8, 1125L: 8, 1296L: 8, 1365L: 5, 1424L: 5, 1600L: 5, 1601L: 8, 1613L: 5, 1616L: 5, 1618L: 5, 1668L: 5, 2015L: 3
  }]
}


DBC = {
  CAR.ACCORD: dbc_dict('honda_accord_s2t_2018_can_generated', None),
  CAR.ACURA_ILX: dbc_dict('acura_ilx_2016_can_generated', 'acura_ilx_2016_nidec'),
  CAR.ACURA_RDX: dbc_dict('acura_rdx_2018_can_generated', 'acura_ilx_2016_nidec'),
  CAR.CIVIC: dbc_dict('honda_civic_touring_2016_can_generated', 'acura_ilx_2016_nidec'),
  CAR.CIVIC_HATCH: dbc_dict('honda_civic_hatchback_ex_2017_can_generated', None),
  CAR.CRV: dbc_dict('honda_crv_touring_2016_can_generated', 'acura_ilx_2016_nidec'),
  CAR.CRV_5G: dbc_dict('honda_crv_ex_2017_can_generated', None),
  CAR.ODYSSEY: dbc_dict('honda_odyssey_exl_2018_generated', 'acura_ilx_2016_nidec'),
  CAR.PILOT: dbc_dict('honda_pilot_touring_2017_can_generated', 'acura_ilx_2016_nidec'),
  CAR.RIDGELINE: dbc_dict('honda_ridgeline_black_edition_2017_can_generated', 'acura_ilx_2016_nidec'),
}

HONDA_BOSCH = [CAR.ACCORD, CAR.CIVIC_HATCH, CAR.CRV_5G]
