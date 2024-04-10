from collections import namedtuple
from dataclasses import dataclass, field
from enum import Enum, IntFlag
import re

from cereal import car
from panda.python import uds
from opendbc.can.can_define import CANDefine
from openpilot.common.conversions import Conversions as CV
from openpilot.selfdrive.car import dbc_dict, CarSpecs, DbcDict, PlatformConfig, Platforms
from openpilot.selfdrive.car.docs_definitions import CarFootnote, CarHarness, CarDocs, CarParts, Column, \
                                                     Device
from openpilot.selfdrive.car.fw_query_definitions import FwQueryConfig, Request, p16

Ecu = car.CarParams.Ecu
NetworkLocation = car.CarParams.NetworkLocation
TransmissionType = car.CarParams.TransmissionType
GearShifter = car.CarState.GearShifter
Button = namedtuple('Button', ['event_type', 'can_addr', 'can_msg', 'values'])


class CarControllerParams:
  STEER_STEP = 2                           # HCA_01/HCA_1 message frequency 50Hz
  ACC_CONTROL_STEP = 2                     # ACC_06/ACC_07/ACC_System frequency 50Hz

  # Documented lateral limits: 3.00 Nm max, rate of change 5.00 Nm/sec.
  # MQB vs PQ maximums are shared, but rate-of-change limited differently
  # based on safety requirements driven by lateral accel testing.

  STEER_MAX = 300                          # Max heading control assist torque 3.00 Nm
  STEER_DRIVER_MULTIPLIER = 3              # weight driver torque heavily
  STEER_DRIVER_FACTOR = 1                  # from dbc

  STEER_TIME_MAX = 360                     # Max time that EPS allows uninterrupted HCA steering control
  STEER_TIME_ALERT = STEER_TIME_MAX - 10   # If mitigation fails, time to soft disengage before EPS timer expires
  STEER_TIME_STUCK_TORQUE = 1.9            # EPS limits same torque to 6 seconds, reset timer 3x within that period

  ACCEL_MAX = 2.0                          # 2.0 m/s max acceleration
  ACCEL_MIN = -3.5                         # 3.5 m/s max deceleration

  def __init__(self, CP):
    can_define = CANDefine(DBC[CP.carFingerprint]["pt"])

    if CP.flags & VolkswagenFlags.PQ:
      self.LDW_STEP = 5                   # LDW_1 message frequency 20Hz
      self.ACC_HUD_STEP = 4               # ACC_GRA_Anzeige frequency 25Hz
      self.STEER_DRIVER_ALLOWANCE = 80    # Driver intervention threshold 0.8 Nm
      self.STEER_DELTA_UP = 6             # Max HCA reached in 1.00s (STEER_MAX / (50Hz * 1.00))
      self.STEER_DELTA_DOWN = 10          # Min HCA reached in 0.60s (STEER_MAX / (50Hz * 0.60))

      if CP.transmissionType == TransmissionType.automatic:
        self.shifter_values = can_define.dv["Getriebe_1"]["Waehlhebelposition__Getriebe_1_"]
      self.hca_status_values = can_define.dv["Lenkhilfe_2"]["LH2_Sta_HCA"]

      self.BUTTONS = [
        Button(car.CarState.ButtonEvent.Type.setCruise, "GRA_Neu", "GRA_Neu_Setzen", [1]),
        Button(car.CarState.ButtonEvent.Type.resumeCruise, "GRA_Neu", "GRA_Recall", [1]),
        Button(car.CarState.ButtonEvent.Type.accelCruise, "GRA_Neu", "GRA_Up_kurz", [1]),
        Button(car.CarState.ButtonEvent.Type.decelCruise, "GRA_Neu", "GRA_Down_kurz", [1]),
        Button(car.CarState.ButtonEvent.Type.cancel, "GRA_Neu", "GRA_Abbrechen", [1]),
        Button(car.CarState.ButtonEvent.Type.gapAdjustCruise, "GRA_Neu", "GRA_Zeitluecke", [1]),
      ]

      self.LDW_MESSAGES = {
        "none": 0,  # Nothing to display
        "laneAssistUnavail": 1,  # "Lane Assist currently not available."
        "laneAssistUnavailSysError": 2,  # "Lane Assist system error"
        "laneAssistUnavailNoSensorView": 3,  # "Lane Assist not available. No sensor view."
        "laneAssistTakeOver": 4,  # "Lane Assist: Please Take Over Steering"
        "laneAssistDeactivTrailer": 5,  # "Lane Assist: no function with trailer"
      }

    else:
      self.LDW_STEP = 10                  # LDW_02 message frequency 10Hz
      self.ACC_HUD_STEP = 6               # ACC_02 message frequency 16Hz
      self.STEER_DRIVER_ALLOWANCE = 80    # Driver intervention threshold 0.8 Nm
      self.STEER_DELTA_UP = 4             # Max HCA reached in 1.50s (STEER_MAX / (50Hz * 1.50))
      self.STEER_DELTA_DOWN = 10          # Min HCA reached in 0.60s (STEER_MAX / (50Hz * 0.60))

      if CP.transmissionType == TransmissionType.automatic:
        self.shifter_values = can_define.dv["Getriebe_11"]["GE_Fahrstufe"]
      elif CP.transmissionType == TransmissionType.direct:
        self.shifter_values = can_define.dv["EV_Gearshift"]["GearPosition"]
      self.hca_status_values = can_define.dv["LH_EPS_03"]["EPS_HCA_Status"]

      self.BUTTONS = [
        Button(car.CarState.ButtonEvent.Type.setCruise, "GRA_ACC_01", "GRA_Tip_Setzen", [1]),
        Button(car.CarState.ButtonEvent.Type.resumeCruise, "GRA_ACC_01", "GRA_Tip_Wiederaufnahme", [1]),
        Button(car.CarState.ButtonEvent.Type.accelCruise, "GRA_ACC_01", "GRA_Tip_Hoch", [1]),
        Button(car.CarState.ButtonEvent.Type.decelCruise, "GRA_ACC_01", "GRA_Tip_Runter", [1]),
        Button(car.CarState.ButtonEvent.Type.cancel, "GRA_ACC_01", "GRA_Abbrechen", [1]),
        Button(car.CarState.ButtonEvent.Type.gapAdjustCruise, "GRA_ACC_01", "GRA_Verstellung_Zeitluecke", [1]),
      ]

      self.LDW_MESSAGES = {
        "none": 0,                            # Nothing to display
        "laneAssistUnavailChime": 1,          # "Lane Assist currently not available." with chime
        "laneAssistUnavailNoSensorChime": 3,  # "Lane Assist not available. No sensor view." with chime
        "laneAssistTakeOverUrgent": 4,        # "Lane Assist: Please Take Over Steering" with urgent beep
        "emergencyAssistUrgent": 6,           # "Emergency Assist: Please Take Over Steering" with urgent beep
        "laneAssistTakeOverChime": 7,         # "Lane Assist: Please Take Over Steering" with chime
        "laneAssistTakeOver": 8,              # "Lane Assist: Please Take Over Steering" silent
        "emergencyAssistChangingLanes": 9,    # "Emergency Assist: Changing lanes..." with urgent beep
        "laneAssistDeactivated": 10,          # "Lane Assist deactivated." silent with persistent icon afterward
      }


class CANBUS:
  pt = 0
  cam = 2


class VolkswagenFlags(IntFlag):
  # Detected flags
  STOCK_HCA_PRESENT = 1

  # Static flags
  PQ = 2


@dataclass
class VolkswagenMQBPlatformConfig(PlatformConfig):
  dbc_dict: DbcDict = field(default_factory=lambda: dbc_dict('vw_mqb_2010', None))
  chassis_codes: set[str] = field(default_factory=set)


@dataclass
class VolkswagenPQPlatformConfig(PlatformConfig):
  dbc_dict: DbcDict = field(default_factory=lambda: dbc_dict('vw_golf_mk4', None))
  chassis_codes: set[str] = field(default_factory=set)

  def init(self):
    self.flags |= VolkswagenFlags.PQ


@dataclass(frozen=True, kw_only=True)
class VolkswagenCarSpecs(CarSpecs):
  centerToFrontRatio: float = 0.45
  steerRatio: float = 15.6


class Footnote(Enum):
  KAMIQ = CarFootnote(
    "Not including the China market Kamiq, which is based on the (currently) unsupported PQ34 platform.",
    Column.MODEL)
  PASSAT = CarFootnote(
    "Refers only to the MQB-based European B8 Passat, not the NMS Passat in the USA/China/Mideast markets.",
    Column.MODEL)
  SKODA_HEATED_WINDSHIELD = CarFootnote(
    "Some Škoda vehicles are equipped with heated windshields, which are known " +
    "to block GPS signal needed for some comma 3X functionality.",
    Column.MODEL)
  VW_EXP_LONG = CarFootnote(
    "Only available for vehicles using a gateway (J533) harness. At this time, vehicles using a camera harness " +
    "are limited to using stock ACC.",
    Column.LONGITUDINAL)
  VW_MQB_A0 = CarFootnote(
    "Model-years 2022 and beyond may have a combined CAN gateway and BCM, which is supported by openpilot " +
    "in software, but doesn't yet have a harness available from the comma store.",
    Column.HARDWARE)


@dataclass
class VWCarDocs(CarDocs):
  package: str = "Adaptive Cruise Control (ACC) & Lane Assist"
  car_parts: CarParts = field(default_factory=CarParts.common([CarHarness.j533]))

  def init_make(self, CP: car.CarParams):
    self.footnotes.append(Footnote.VW_EXP_LONG)
    if "SKODA" in CP.carFingerprint:
      self.footnotes.append(Footnote.SKODA_HEATED_WINDSHIELD)

    if CP.carFingerprint in (CAR.VOLKSWAGEN_CRAFTER_MK2, CAR.VOLKSWAGEN_TRANSPORTER_T61):
      self.car_parts = CarParts([Device.threex_angled_mount, CarHarness.j533])


# Check the 7th and 8th characters of the VIN before adding a new CAR. If the
# chassis code is already listed below, don't add a new CAR, just add to the
# FW_VERSIONS for that existing CAR.

class CAR(Platforms):
  VOLKSWAGEN_ARTEON_MK1 = VolkswagenMQBPlatformConfig(
    [
      VWCarDocs("Volkswagen Arteon 2018-23", video_link="https://youtu.be/FAomFKPFlDA"),
      VWCarDocs("Volkswagen Arteon R 2020-23", video_link="https://youtu.be/FAomFKPFlDA"),
      VWCarDocs("Volkswagen Arteon eHybrid 2020-23", video_link="https://youtu.be/FAomFKPFlDA"),
      VWCarDocs("Volkswagen CC 2018-22", video_link="https://youtu.be/FAomFKPFlDA"),
    ],
    VolkswagenCarSpecs(mass=1733, wheelbase=2.84),
    chassis_codes={"AN"},
  )
  VOLKSWAGEN_ATLAS_MK1 = VolkswagenMQBPlatformConfig(
    [
      VWCarDocs("Volkswagen Atlas 2018-23"),
      VWCarDocs("Volkswagen Atlas Cross Sport 2020-22"),
      VWCarDocs("Volkswagen Teramont 2018-22"),
      VWCarDocs("Volkswagen Teramont Cross Sport 2021-22"),
      VWCarDocs("Volkswagen Teramont X 2021-22"),
    ],
    VolkswagenCarSpecs(mass=2011, wheelbase=2.98),
    chassis_codes={"CA"},
  )
  VOLKSWAGEN_CADDY_MK3 = VolkswagenPQPlatformConfig(
    [
      VWCarDocs("Volkswagen Caddy 2019"),
      VWCarDocs("Volkswagen Caddy Maxi 2019"),
    ],
    VolkswagenCarSpecs(mass=1613, wheelbase=2.6, minSteerSpeed=21 * CV.KPH_TO_MS),
    chassis_codes={"2K"},
  )
  VOLKSWAGEN_CRAFTER_MK2 = VolkswagenMQBPlatformConfig(
    [
      VWCarDocs("Volkswagen Crafter 2017-23", video_link="https://youtu.be/4100gLeabmo"),
      VWCarDocs("Volkswagen e-Crafter 2018-23", video_link="https://youtu.be/4100gLeabmo"),
      VWCarDocs("Volkswagen Grand California 2019-23", video_link="https://youtu.be/4100gLeabmo"),
      VWCarDocs("MAN TGE 2017-23", video_link="https://youtu.be/4100gLeabmo"),
      VWCarDocs("MAN eTGE 2020-23", video_link="https://youtu.be/4100gLeabmo"),
    ],
    VolkswagenCarSpecs(mass=2100, wheelbase=3.64, minSteerSpeed=50 * CV.KPH_TO_MS),
    chassis_codes={"SY", "SZ"},
  )
  VOLKSWAGEN_GOLF_MK7 = VolkswagenMQBPlatformConfig(
    [
      VWCarDocs("Volkswagen e-Golf 2014-20"),
      VWCarDocs("Volkswagen Golf 2015-20", auto_resume=False),
      VWCarDocs("Volkswagen Golf Alltrack 2015-19", auto_resume=False),
      VWCarDocs("Volkswagen Golf GTD 2015-20"),
      VWCarDocs("Volkswagen Golf GTE 2015-20"),
      VWCarDocs("Volkswagen Golf GTI 2015-21", auto_resume=False),
      VWCarDocs("Volkswagen Golf R 2015-19"),
      VWCarDocs("Volkswagen Golf SportsVan 2015-20"),
    ],
    VolkswagenCarSpecs(mass=1397, wheelbase=2.62),
    chassis_codes={"5G", "AU", "BA", "BE"},
  )
  VOLKSWAGEN_JETTA_MK7 = VolkswagenMQBPlatformConfig(
    [
      VWCarDocs("Volkswagen Jetta 2018-24"),
      VWCarDocs("Volkswagen Jetta GLI 2021-24"),
    ],
    VolkswagenCarSpecs(mass=1328, wheelbase=2.71),
    chassis_codes={"BU"},
  )
  VOLKSWAGEN_PASSAT_MK8 = VolkswagenMQBPlatformConfig(
    [
      VWCarDocs("Volkswagen Passat 2015-22", footnotes=[Footnote.PASSAT]),
      VWCarDocs("Volkswagen Passat Alltrack 2015-22"),
      VWCarDocs("Volkswagen Passat GTE 2015-22"),
    ],
    VolkswagenCarSpecs(mass=1551, wheelbase=2.79),
    chassis_codes={"3G"},
  )
  VOLKSWAGEN_PASSAT_NMS = VolkswagenPQPlatformConfig(
    [VWCarDocs("Volkswagen Passat NMS 2017-22")],
    VolkswagenCarSpecs(mass=1503, wheelbase=2.80, minSteerSpeed=50 * CV.KPH_TO_MS, minEnableSpeed=20 * CV.KPH_TO_MS),
    chassis_codes={"A3"},
  )
  VOLKSWAGEN_POLO_MK6 = VolkswagenMQBPlatformConfig(
    [
      VWCarDocs("Volkswagen Polo 2018-23", footnotes=[Footnote.VW_MQB_A0]),
      VWCarDocs("Volkswagen Polo GTI 2018-23", footnotes=[Footnote.VW_MQB_A0]),
    ],
    VolkswagenCarSpecs(mass=1230, wheelbase=2.55),
    chassis_codes={"AW"},
  )
  VOLKSWAGEN_SHARAN_MK2 = VolkswagenPQPlatformConfig(
    [
      VWCarDocs("Volkswagen Sharan 2018-22"),
      VWCarDocs("SEAT Alhambra 2018-20"),
    ],
    VolkswagenCarSpecs(mass=1639, wheelbase=2.92, minSteerSpeed=50 * CV.KPH_TO_MS),
    chassis_codes={"7N"},
  )
  VOLKSWAGEN_TAOS_MK1 = VolkswagenMQBPlatformConfig(
    [VWCarDocs("Volkswagen Taos 2022-23")],
    VolkswagenCarSpecs(mass=1498, wheelbase=2.69),
    chassis_codes={"B2"},
  )
  VOLKSWAGEN_TCROSS_MK1 = VolkswagenMQBPlatformConfig(
    [VWCarDocs("Volkswagen T-Cross 2021", footnotes=[Footnote.VW_MQB_A0])],
    VolkswagenCarSpecs(mass=1150, wheelbase=2.60),
    chassis_codes={"C1"},
  )
  VOLKSWAGEN_TIGUAN_MK2 = VolkswagenMQBPlatformConfig(
    [
      VWCarDocs("Volkswagen Tiguan 2018-24"),
      VWCarDocs("Volkswagen Tiguan eHybrid 2021-23"),
    ],
    VolkswagenCarSpecs(mass=1715, wheelbase=2.74),
    chassis_codes={"AD", "BW"},
  )
  VOLKSWAGEN_TOURAN_MK2 = VolkswagenMQBPlatformConfig(
    [VWCarDocs("Volkswagen Touran 2016-23")],
    VolkswagenCarSpecs(mass=1516, wheelbase=2.79),
    chassis_codes={"1T"},
  )
  VOLKSWAGEN_TRANSPORTER_T61 = VolkswagenMQBPlatformConfig(
    [
      VWCarDocs("Volkswagen Caravelle 2020"),
      VWCarDocs("Volkswagen California 2021-23"),
    ],
    VolkswagenCarSpecs(mass=1926, wheelbase=3.00, minSteerSpeed=14.0),
    chassis_codes={"7H", "7L"},
  )
  VOLKSWAGEN_TROC_MK1 = VolkswagenMQBPlatformConfig(
    [VWCarDocs("Volkswagen T-Roc 2018-22", footnotes=[Footnote.VW_MQB_A0])],
    VolkswagenCarSpecs(mass=1413, wheelbase=2.63),
    chassis_codes={"A1"},
  )
  AUDI_A3_MK3 = VolkswagenMQBPlatformConfig(
    [
      VWCarDocs("Audi A3 2014-19"),
      VWCarDocs("Audi A3 Sportback e-tron 2017-18"),
      VWCarDocs("Audi RS3 2018"),
      VWCarDocs("Audi S3 2015-17"),
    ],
    VolkswagenCarSpecs(mass=1335, wheelbase=2.61),
    chassis_codes={"8V", "FF"},
  )
  AUDI_Q2_MK1 = VolkswagenMQBPlatformConfig(
    [VWCarDocs("Audi Q2 2018")],
    VolkswagenCarSpecs(mass=1205, wheelbase=2.61),
    chassis_codes={"GA"},
  )
  AUDI_Q3_MK2 = VolkswagenMQBPlatformConfig(
    [VWCarDocs("Audi Q3 2019-23")],
    VolkswagenCarSpecs(mass=1623, wheelbase=2.68),
    chassis_codes={"8U", "F3", "FS"},
  )
  SEAT_ATECA_MK1 = VolkswagenMQBPlatformConfig(
    [
      VWCarDocs("SEAT Ateca 2018"),
      VWCarDocs("SEAT Leon 2014-20"),
    ],
    VolkswagenCarSpecs(mass=1300, wheelbase=2.64),
    chassis_codes={"5F"},
  )
  SKODA_FABIA_MK4 = VolkswagenMQBPlatformConfig(
    [VWCarDocs("Škoda Fabia 2022-23", footnotes=[Footnote.VW_MQB_A0])],
    VolkswagenCarSpecs(mass=1266, wheelbase=2.56),
    chassis_codes={"PJ"},
  )
  SKODA_KAMIQ_MK1 = VolkswagenMQBPlatformConfig(
    [
      VWCarDocs("Škoda Kamiq 2021-23", footnotes=[Footnote.VW_MQB_A0, Footnote.KAMIQ]),
      VWCarDocs("Škoda Scala 2020-23", footnotes=[Footnote.VW_MQB_A0]),
    ],
    VolkswagenCarSpecs(mass=1230, wheelbase=2.66),
    chassis_codes={"NW"},
  )
  SKODA_KAROQ_MK1 = VolkswagenMQBPlatformConfig(
    [VWCarDocs("Škoda Karoq 2019-23")],
    VolkswagenCarSpecs(mass=1278, wheelbase=2.66),
    chassis_codes={"NU"},
  )
  SKODA_KODIAQ_MK1 = VolkswagenMQBPlatformConfig(
    [VWCarDocs("Škoda Kodiaq 2017-23")],
    VolkswagenCarSpecs(mass=1569, wheelbase=2.79),
    chassis_codes={"NS"},
  )
  SKODA_OCTAVIA_MK3 = VolkswagenMQBPlatformConfig(
    [
      VWCarDocs("Škoda Octavia 2015-19"),
      VWCarDocs("Škoda Octavia RS 2016"),
      VWCarDocs("Škoda Octavia Scout 2017-19"),
    ],
    VolkswagenCarSpecs(mass=1388, wheelbase=2.68),
    chassis_codes={"NE"},
  )
  SKODA_SUPERB_MK3 = VolkswagenMQBPlatformConfig(
    [VWCarDocs("Škoda Superb 2015-22")],
    VolkswagenCarSpecs(mass=1505, wheelbase=2.84),
    chassis_codes={"3V", "NP"},
  )


def match_fw_to_car_custom(live_fw_versions, vin, offline_fw_versions) -> set[str]:
  candidates = set()

  chassis_code = vin[6:8]
  if CHASSIS_CODE_PATTERN.match(chassis_code) is None:
    return set()

  for platform in CAR:
    if chassis_code in platform.config.chassis_codes:
      candidates.add(platform)

  return {str(c) for c in candidates}


CHASSIS_CODE_PATTERN = re.compile('[A-Z0-9]{2}')

# All supported cars should return FW from the engine, srs, eps, and fwdRadar. Cars
# with a manual trans won't return transmission firmware, but all other cars will.
#
# The 0xF187 SW part number query should return in the form of N[NX][NX] NNN NNN [X[X]],
# where N=number, X=letter, and the trailing two letters are optional. Performance
# tuners sometimes tamper with that field (e.g. 8V0 9C0 BB0 1 from COBB/EQT). Tampered
# ECU SW part numbers are invalid for vehicle ID and compatibility checks. Try to have
# them repaired by the tuner before including them in openpilot.

VOLKSWAGEN_VERSION_REQUEST_MULTI = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER]) + \
  p16(uds.DATA_IDENTIFIER_TYPE.VEHICLE_MANUFACTURER_SPARE_PART_NUMBER) + \
  p16(uds.DATA_IDENTIFIER_TYPE.VEHICLE_MANUFACTURER_ECU_SOFTWARE_VERSION_NUMBER) + \
  p16(uds.DATA_IDENTIFIER_TYPE.APPLICATION_DATA_IDENTIFICATION)
VOLKSWAGEN_VERSION_RESPONSE = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER + 0x40])

VOLKSWAGEN_RX_OFFSET = 0x6a

FW_QUERY_CONFIG = FwQueryConfig(
  # TODO: add back whitelists after we gather enough data
  requests=[request for bus, obd_multiplexing in [(1, True), (1, False), (0, False)] for request in [
    Request(
      [VOLKSWAGEN_VERSION_REQUEST_MULTI],
      [VOLKSWAGEN_VERSION_RESPONSE],
      whitelist_ecus=[Ecu.srs, Ecu.eps, Ecu.fwdRadar, Ecu.fwdCamera],
      rx_offset=VOLKSWAGEN_RX_OFFSET,
      bus=bus,
      logging=(bus != 1 or not obd_multiplexing),
      obd_multiplexing=obd_multiplexing,
    ),
    Request(
      [VOLKSWAGEN_VERSION_REQUEST_MULTI],
      [VOLKSWAGEN_VERSION_RESPONSE],
      whitelist_ecus=[Ecu.engine, Ecu.transmission],
      bus=bus,
      logging=(bus != 1 or not obd_multiplexing),
      obd_multiplexing=obd_multiplexing,
    ),
  ]],
  extra_ecus=[(Ecu.fwdCamera, 0x74f, None)],
  match_fw_to_car_custom=match_fw_to_car_custom,
)

DBC = CAR.create_dbc_map()
