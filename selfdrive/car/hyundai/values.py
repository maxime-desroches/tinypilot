import re
from dataclasses import dataclass, field
from enum import Enum, IntFlag

from cereal import car
from panda.python import uds
from openpilot.common.conversions import Conversions as CV
from openpilot.selfdrive.car import DbcDict, PlatformConfig, Platforms, dbc_dict
from openpilot.selfdrive.car.docs_definitions import CarFootnote, CarHarness, CarInfo, CarParts, Column
from openpilot.selfdrive.car.fw_query_definitions import FwQueryConfig, Request, p16

Ecu = car.CarParams.Ecu


class CarControllerParams:
  ACCEL_MIN = -3.5 # m/s
  ACCEL_MAX = 2.0 # m/s

  def __init__(self, CP):
    self.STEER_DELTA_UP = 3
    self.STEER_DELTA_DOWN = 7
    self.STEER_DRIVER_ALLOWANCE = 50
    self.STEER_DRIVER_MULTIPLIER = 2
    self.STEER_DRIVER_FACTOR = 1
    self.STEER_THRESHOLD = 150
    self.STEER_STEP = 1  # 100 Hz

    if CP.carFingerprint in CANFD_CAR:
      self.STEER_MAX = 270
      self.STEER_DRIVER_ALLOWANCE = 250
      self.STEER_DRIVER_MULTIPLIER = 2
      self.STEER_THRESHOLD = 250
      self.STEER_DELTA_UP = 2
      self.STEER_DELTA_DOWN = 3

    # To determine the limit for your car, find the maximum value that the stock LKAS will request.
    # If the max stock LKAS request is <384, add your car to this list.
    elif CP.carFingerprint in (CAR.GENESIS_G80, CAR.GENESIS_G90, CAR.ELANTRA, CAR.ELANTRA_GT_I30, CAR.IONIQ,
                               CAR.IONIQ_EV_LTD, CAR.SANTA_FE_PHEV_2022, CAR.SONATA_LF, CAR.KIA_FORTE, CAR.KIA_NIRO_PHEV,
                               CAR.KIA_OPTIMA_H, CAR.KIA_OPTIMA_H_G4_FL, CAR.KIA_SORENTO):
      self.STEER_MAX = 255

    # these cars have significantly more torque than most HKG; limit to 70% of max
    elif CP.flags & HyundaiFlags.ALT_LIMITS:
      self.STEER_MAX = 270
      self.STEER_DELTA_UP = 2
      self.STEER_DELTA_DOWN = 3

    # Default for most HKG
    else:
      self.STEER_MAX = 384


class HyundaiFlags(IntFlag):
  # Dynamic Flags
  CANFD_HDA2 = 1
  CANFD_ALT_BUTTONS = 2
  CANFD_ALT_GEARS = 4
  CANFD_CAMERA_SCC = 8

  ALT_LIMITS = 16
  ENABLE_BLINKERS = 32
  CANFD_ALT_GEARS_2 = 64
  SEND_LFA = 128
  USE_FCA = 256
  CANFD_HDA2_ALT_STEERING = 512

  # these cars use a different gas signal
  HYBRID = 1024
  EV = 2048

  # Static Flags

  # If 0x500 is present on bus 1 it probably has a Mando radar outputting radar points.
  # If no points are outputted by default it might be possible to turn it on using  selfdrive/debug/hyundai_enable_radar_points.py
  MANDO_RADAR = 4096
  CANFD = 8192

  # The radar does SCC on these cars when HDA I, rather than the camera
  RADAR_SCC = 16384
  CAMERA_SCC = 32768
  CHECKSUM_CRC8 = 65536
  CHECKSUM_6B = 131072

  # these cars require a special panda safety mode due to missing counters and checksums in the messages
  LEGACY = 262144

  # these cars have not been verified to work with longitudinal yet - radar disable, sending correct messages, etc.
  UNSUPPORTED_LONGITUDINAL = 524288

  CANFD_NO_RADAR_DISABLE = 1048576

  CLUSTER_GEARS = 2097152
  TCU_GEARS = 4194304


@dataclass
class HyundaiCarInfo(CarInfo):
  package: str = "Smart Cruise Control (SCC)"

  def init_make(self, CP: car.CarParams):
    if CP.carFingerprint in CANFD_CAR:
      self.footnotes.insert(0, Footnote.CANFD)


@dataclass
class HyundaiPlatformConfig(PlatformConfig):
  dbc_dict: DbcDict = field(default_factory=lambda: dbc_dict("hyundai_kia_generic", None))

  def init(self):
    if self.flags & HyundaiFlags.CANFD:
      self.dbc_dict = dbc_dict('hyundai_canfd', None)
    if self.flags & HyundaiFlags.MANDO_RADAR:
      self.dbc_dict = dbc_dict('hyundai_kia_generic', 'hyundai_kia_mando_front_radar_generated')


class CAR(Platforms):
  # Hyundai
  AZERA_6TH_GEN = HyundaiPlatformConfig(
    "HYUNDAI AZERA 6TH GEN",
    HyundaiCarInfo("Hyundai Azera 2022", "All", car_parts=CarParts.common([CarHarness.hyundai_k])),
  )
  AZERA_HEV_6TH_GEN = HyundaiPlatformConfig(
    "HYUNDAI AZERA HYBRID 6TH GEN",
    [
      HyundaiCarInfo("Hyundai Azera Hybrid 2019", "All", car_parts=CarParts.common([CarHarness.hyundai_c])),
      HyundaiCarInfo("Hyundai Azera Hybrid 2020", "All", car_parts=CarParts.common([CarHarness.hyundai_k])),
    ],
    flags=HyundaiFlags.HYBRID
  )
  ELANTRA = HyundaiPlatformConfig(
    "HYUNDAI ELANTRA 2017",
    [
      # TODO: 2017-18 could be Hyundai G
      HyundaiCarInfo("Hyundai Elantra 2017-18", min_enable_speed=19 * CV.MPH_TO_MS, car_parts=CarParts.common([CarHarness.hyundai_b])),
      HyundaiCarInfo("Hyundai Elantra 2019", min_enable_speed=19 * CV.MPH_TO_MS, car_parts=CarParts.common([CarHarness.hyundai_g])),
    ],
    flags=HyundaiFlags.LEGACY | HyundaiFlags.CLUSTER_GEARS
  )
  ELANTRA_GT_I30 = HyundaiPlatformConfig(
    "HYUNDAI I30 N LINE 2019 & GT 2018 DCT",
    [
      HyundaiCarInfo("Hyundai Elantra GT 2017-19", car_parts=CarParts.common([CarHarness.hyundai_e])),
      HyundaiCarInfo("Hyundai i30 2017-19", car_parts=CarParts.common([CarHarness.hyundai_e])),
    ],
    flags=HyundaiFlags.LEGACY | HyundaiFlags.CLUSTER_GEARS
  )
  ELANTRA_2021 = HyundaiPlatformConfig(
    "HYUNDAI ELANTRA 2021",
    HyundaiCarInfo("Hyundai Elantra 2021-23", video_link="https://youtu.be/_EdYQtV52-c", car_parts=CarParts.common([CarHarness.hyundai_k])),
    flags=HyundaiFlags.CHECKSUM_CRC8
  )
  ELANTRA_HEV_2021 = HyundaiPlatformConfig(
    "HYUNDAI ELANTRA HYBRID 2021",
    HyundaiCarInfo("Hyundai Elantra Hybrid 2021-23", video_link="https://youtu.be/_EdYQtV52-c",
                                       car_parts=CarParts.common([CarHarness.hyundai_k])),
    flags=HyundaiFlags.CHECKSUM_CRC8 | HyundaiFlags.HYBRID
  )
  HYUNDAI_GENESIS = HyundaiPlatformConfig(
    "HYUNDAI GENESIS 2015-2016",
    [
      # TODO: check 2015 packages
      HyundaiCarInfo("Hyundai Genesis 2015-16", min_enable_speed=19 * CV.MPH_TO_MS, car_parts=CarParts.common([CarHarness.hyundai_j])),
      HyundaiCarInfo("Genesis G80 2017", "All", min_enable_speed=19 * CV.MPH_TO_MS, car_parts=CarParts.common([CarHarness.hyundai_j])),
    ],
    flags=HyundaiFlags.CHECKSUM_6B | HyundaiFlags.LEGACY
  )
  IONIQ = HyundaiPlatformConfig(
    "HYUNDAI IONIQ HYBRID 2017-2019",
    HyundaiCarInfo("Hyundai Ioniq Hybrid 2017-19", car_parts=CarParts.common([CarHarness.hyundai_c])),
    flags=HyundaiFlags.HYBRID
  )
  IONIQ_HEV_2022 = HyundaiPlatformConfig(
    "HYUNDAI IONIQ HYBRID 2020-2022",
    HyundaiCarInfo("Hyundai Ioniq Hybrid 2020-22", car_parts=CarParts.common([CarHarness.hyundai_h])),  # TODO: confirm 2020-21 harness,
    flags=HyundaiFlags.HYBRID | HyundaiFlags.LEGACY
  )
  IONIQ_EV_LTD = HyundaiPlatformConfig(
    "HYUNDAI IONIQ ELECTRIC LIMITED 2019",
    HyundaiCarInfo("Hyundai Ioniq Electric 2019", car_parts=CarParts.common([CarHarness.hyundai_c])),
    flags=HyundaiFlags.MANDO_RADAR | HyundaiFlags.EV | HyundaiFlags.LEGACY
  )
  IONIQ_EV_2020 = HyundaiPlatformConfig(
    "HYUNDAI IONIQ ELECTRIC 2020",
    HyundaiCarInfo("Hyundai Ioniq Electric 2020", "All", car_parts=CarParts.common([CarHarness.hyundai_h])),
    flags=HyundaiFlags.EV
  )
  IONIQ_PHEV_2019 = HyundaiPlatformConfig(
    "HYUNDAI IONIQ PLUG-IN HYBRID 2019",
    HyundaiCarInfo("Hyundai Ioniq Plug-in Hybrid 2019", car_parts=CarParts.common([CarHarness.hyundai_c])),
    flags=HyundaiFlags.HYBRID
  )
  IONIQ_PHEV = HyundaiPlatformConfig(
    "HYUNDAI IONIQ PHEV 2020",
    HyundaiCarInfo("Hyundai Ioniq Plug-in Hybrid 2020-22", "All", car_parts=CarParts.common([CarHarness.hyundai_h])),
    flags=HyundaiFlags.HYBRID
  )
  KONA = HyundaiPlatformConfig(
    "HYUNDAI KONA 2020",
    HyundaiCarInfo("Hyundai Kona 2020", car_parts=CarParts.common([CarHarness.hyundai_b])),
    flags=HyundaiFlags.CLUSTER_GEARS
  )
  KONA_EV = HyundaiPlatformConfig(
    "HYUNDAI KONA ELECTRIC 2019",
    HyundaiCarInfo("Hyundai Kona Electric 2018-21", car_parts=CarParts.common([CarHarness.hyundai_g])),
    flags=HyundaiFlags.EV
  )
  KONA_EV_2022 = HyundaiPlatformConfig(
    "HYUNDAI KONA ELECTRIC 2022",
    HyundaiCarInfo("Hyundai Kona Electric 2022-23", car_parts=CarParts.common([CarHarness.hyundai_o])),
    flags=HyundaiFlags.CAMERA_SCC | HyundaiFlags.EV
  )
  KONA_EV_2ND_GEN = HyundaiPlatformConfig(
    "HYUNDAI KONA ELECTRIC 2ND GEN",
    HyundaiCarInfo("Hyundai Kona Electric (with HDA II, Korea only) 2023", video_link="https://www.youtube.com/watch?v=U2fOCmcQ8hw",
                                      car_parts=CarParts.common([CarHarness.hyundai_r])),
    flags=HyundaiFlags.CANFD | HyundaiFlags.EV | HyundaiFlags.CANFD_NO_RADAR_DISABLE
  )
  KONA_HEV = HyundaiPlatformConfig(
    "HYUNDAI KONA HYBRID 2020",
    HyundaiCarInfo("Hyundai Kona Hybrid 2020", car_parts=CarParts.common([CarHarness.hyundai_i])),  # TODO: check packages,
    flags=HyundaiFlags.HYBRID
  )
  SANTA_FE = HyundaiPlatformConfig(
    "HYUNDAI SANTA FE 2019",
    HyundaiCarInfo("Hyundai Santa Fe 2019-20", "All", video_link="https://youtu.be/bjDR0YjM__s",
                               car_parts=CarParts.common([CarHarness.hyundai_d])),
    flags=HyundaiFlags.MANDO_RADAR | HyundaiFlags.CHECKSUM_CRC8
  )
  SANTA_FE_2022 = HyundaiPlatformConfig(
    "HYUNDAI SANTA FE 2022",
     HyundaiCarInfo("Hyundai Santa Fe 2021-23", "All", video_link="https://youtu.be/VnHzSTygTS4",
                                    car_parts=CarParts.common([CarHarness.hyundai_l])),
    flags=HyundaiFlags.CHECKSUM_CRC8
  )
  SANTA_FE_HEV_2022 = HyundaiPlatformConfig(
    "HYUNDAI SANTA FE HYBRID 2022",
    HyundaiCarInfo("Hyundai Santa Fe Hybrid 2022-23", "All", car_parts=CarParts.common([CarHarness.hyundai_l])),
    flags=HyundaiFlags.CHECKSUM_CRC8 | HyundaiFlags.HYBRID
  )
  SANTA_FE_PHEV_2022 = HyundaiPlatformConfig(
    "HYUNDAI SANTA FE PlUG-IN HYBRID 2022",
    HyundaiCarInfo("Hyundai Santa Fe Plug-in Hybrid 2022-23", "All", car_parts=CarParts.common([CarHarness.hyundai_l])),
    flags=HyundaiFlags.CHECKSUM_CRC8 | HyundaiFlags.HYBRID
  )
  SONATA = HyundaiPlatformConfig(
    "HYUNDAI SONATA 2020",
    HyundaiCarInfo("Hyundai Sonata 2020-23", "All", video_link="https://www.youtube.com/watch?v=ix63r9kE3Fw",
                             car_parts=CarParts.common([CarHarness.hyundai_a])),
    flags=HyundaiFlags.MANDO_RADAR | HyundaiFlags.CHECKSUM_CRC8
  )
  SONATA_LF = HyundaiPlatformConfig(
    "HYUNDAI SONATA 2019",
    HyundaiCarInfo("Hyundai Sonata 2018-19", car_parts=CarParts.common([CarHarness.hyundai_e])),
    flags=HyundaiFlags.UNSUPPORTED_LONGITUDINAL | HyundaiFlags.TCU_GEARS
  )
  STARIA_4TH_GEN = HyundaiPlatformConfig(
    "HYUNDAI STARIA 4TH GEN",
    HyundaiCarInfo("Hyundai Staria 2023", "All", car_parts=CarParts.common([CarHarness.hyundai_k])),
    flags=HyundaiFlags.CANFD
  )
  TUCSON = HyundaiPlatformConfig(
    "HYUNDAI TUCSON 2019",
    [
      HyundaiCarInfo("Hyundai Tucson 2021", min_enable_speed=19 * CV.MPH_TO_MS, car_parts=CarParts.common([CarHarness.hyundai_l])),
      HyundaiCarInfo("Hyundai Tucson Diesel 2019", car_parts=CarParts.common([CarHarness.hyundai_l])),
    ],
    flags=HyundaiFlags.TCU_GEARS
  )
  PALISADE = HyundaiPlatformConfig(
    "HYUNDAI PALISADE 2020",
    [
      HyundaiCarInfo("Hyundai Palisade 2020-22", "All", video_link="https://youtu.be/TAnDqjF4fDY?t=456", car_parts=CarParts.common([CarHarness.hyundai_h])),
      HyundaiCarInfo("Kia Telluride 2020-22", "All", car_parts=CarParts.common([CarHarness.hyundai_h])),
    ],
    flags=HyundaiFlags.MANDO_RADAR | HyundaiFlags.CHECKSUM_CRC8
  )
  VELOSTER = HyundaiPlatformConfig(
    "HYUNDAI VELOSTER 2019",
    HyundaiCarInfo("Hyundai Veloster 2019-20", min_enable_speed=5. * CV.MPH_TO_MS, car_parts=CarParts.common([CarHarness.hyundai_e])),
    flags=HyundaiFlags.LEGACY | HyundaiFlags.TCU_GEARS
  )
  SONATA_HYBRID = HyundaiPlatformConfig(
    "HYUNDAI SONATA HYBRID 2021",
    HyundaiCarInfo("Hyundai Sonata Hybrid 2020-23", "All", car_parts=CarParts.common([CarHarness.hyundai_a])),
    flags=HyundaiFlags.MANDO_RADAR | HyundaiFlags.CHECKSUM_CRC8 | HyundaiFlags.HYBRID
  )
  IONIQ_5 = HyundaiPlatformConfig(
    "HYUNDAI IONIQ 5 2022",
    [
      HyundaiCarInfo("Hyundai Ioniq 5 (Southeast Asia only) 2022-23", "All", car_parts=CarParts.common([CarHarness.hyundai_q])),
      HyundaiCarInfo("Hyundai Ioniq 5 (without HDA II) 2022-23", "Highway Driving Assist", car_parts=CarParts.common([CarHarness.hyundai_k])),
      HyundaiCarInfo("Hyundai Ioniq 5 (with HDA II) 2022-23", "Highway Driving Assist II", car_parts=CarParts.common([CarHarness.hyundai_q])),
    ],
    flags=HyundaiFlags.CANFD | HyundaiFlags.EV
  )
  IONIQ_6 = HyundaiPlatformConfig(
    "HYUNDAI IONIQ 6 2023",
    HyundaiCarInfo("Hyundai Ioniq 6 (with HDA II) 2023", "Highway Driving Assist II", car_parts=CarParts.common([CarHarness.hyundai_p])),
    flags=HyundaiFlags.CANFD | HyundaiFlags.EV | HyundaiFlags.CANFD_NO_RADAR_DISABLE
  )
  TUCSON_4TH_GEN = HyundaiPlatformConfig(
    "HYUNDAI TUCSON 4TH GEN",
    [
      HyundaiCarInfo("Hyundai Tucson 2022", car_parts=CarParts.common([CarHarness.hyundai_n])),
      HyundaiCarInfo("Hyundai Tucson 2023", "All", car_parts=CarParts.common([CarHarness.hyundai_n])),
      HyundaiCarInfo("Hyundai Tucson Hybrid 2022-24", "All", car_parts=CarParts.common([CarHarness.hyundai_n])),
    ],
    flags=HyundaiFlags.CANFD
  )
  SANTA_CRUZ_1ST_GEN = HyundaiPlatformConfig(
    "HYUNDAI SANTA CRUZ 1ST GEN",
    HyundaiCarInfo("Hyundai Santa Cruz 2022-23", car_parts=CarParts.common([CarHarness.hyundai_n])),
    flags=HyundaiFlags.CANFD
  )
  CUSTIN_1ST_GEN = HyundaiPlatformConfig(
    "HYUNDAI CUSTIN 1ST GEN",
    HyundaiCarInfo("Hyundai Custin 2023", "All", car_parts=CarParts.common([CarHarness.hyundai_k])),
    flags=HyundaiFlags.CHECKSUM_CRC8
  )

  # Kia
  KIA_FORTE = HyundaiPlatformConfig(
    "KIA FORTE E 2018 & GT 2021",
    [
    HyundaiCarInfo("Kia Forte 2019-21", car_parts=CarParts.common([CarHarness.hyundai_g])),
    HyundaiCarInfo("Kia Forte 2023", car_parts=CarParts.common([CarHarness.hyundai_e])),
  ]
  )
  KIA_K5_2021 = HyundaiPlatformConfig(
    "KIA K5 2021",
    HyundaiCarInfo("Kia K5 2021-24", car_parts=CarParts.common([CarHarness.hyundai_a])),
    flags=HyundaiFlags.CHECKSUM_CRC8
  )
  KIA_K5_HEV_2020 = HyundaiPlatformConfig(
    "KIA K5 HYBRID 2020",
    HyundaiCarInfo("Kia K5 Hybrid 2020-22", car_parts=CarParts.common([CarHarness.hyundai_a])),
    flags=HyundaiFlags.MANDO_RADAR | HyundaiFlags.CHECKSUM_CRC8 | HyundaiFlags.HYBRID
  )
  KIA_K8_HEV_1ST_GEN = HyundaiPlatformConfig(
    "KIA K8 HYBRID 1ST GEN",
    HyundaiCarInfo("Kia K8 Hybrid (with HDA II) 2023", "Highway Driving Assist II", car_parts=CarParts.common([CarHarness.hyundai_q])),
    flags=HyundaiFlags.CANFD
  )
  KIA_NIRO_EV = HyundaiPlatformConfig(
    "KIA NIRO EV 2020",
    [
      HyundaiCarInfo("Kia Niro EV 2019", "All", video_link="https://www.youtube.com/watch?v=lT7zcG6ZpGo", car_parts=CarParts.common([CarHarness.hyundai_h])),
      HyundaiCarInfo("Kia Niro EV 2020", "All", video_link="https://www.youtube.com/watch?v=lT7zcG6ZpGo", car_parts=CarParts.common([CarHarness.hyundai_f])),
      HyundaiCarInfo("Kia Niro EV 2021", "All", video_link="https://www.youtube.com/watch?v=lT7zcG6ZpGo", car_parts=CarParts.common([CarHarness.hyundai_c])),
      HyundaiCarInfo("Kia Niro EV 2022", "All", video_link="https://www.youtube.com/watch?v=lT7zcG6ZpGo", car_parts=CarParts.common([CarHarness.hyundai_h])),
    ],
    flags=HyundaiFlags.MANDO_RADAR | HyundaiFlags.EV
  )
  KIA_NIRO_EV_2ND_GEN = HyundaiPlatformConfig(
    "KIA NIRO EV 2ND GEN",
    HyundaiCarInfo("Kia Niro EV 2023", "All", car_parts=CarParts.common([CarHarness.hyundai_a])),
    flags=HyundaiFlags.CANFD | HyundaiFlags.EV
  )
  KIA_NIRO_PHEV = HyundaiPlatformConfig(
    "KIA NIRO HYBRID 2019",
    [
      HyundaiCarInfo("Kia Niro Plug-in Hybrid 2018-19", "All", min_enable_speed=10. * CV.MPH_TO_MS, car_parts=CarParts.common([CarHarness.hyundai_c])),
      HyundaiCarInfo("Kia Niro Plug-in Hybrid 2020", "All", car_parts=CarParts.common([CarHarness.hyundai_d])),
    ],
    flags=HyundaiFlags.MANDO_RADAR | HyundaiFlags.HYBRID | HyundaiFlags.UNSUPPORTED_LONGITUDINAL
  )
  KIA_NIRO_PHEV_2022 = HyundaiPlatformConfig(
    "KIA NIRO PLUG-IN HYBRID 2022",
    [
      HyundaiCarInfo("Kia Niro Plug-in Hybrid 2021", "All", car_parts=CarParts.common([CarHarness.hyundai_d])),
      HyundaiCarInfo("Kia Niro Plug-in Hybrid 2022", "All", car_parts=CarParts.common([CarHarness.hyundai_f])),
    ],
    flags=HyundaiFlags.HYBRID
  )
  KIA_NIRO_HEV_2021 = HyundaiPlatformConfig(
    "KIA NIRO HYBRID 2021",
    [
      HyundaiCarInfo("Kia Niro Hybrid 2021", car_parts=CarParts.common([CarHarness.hyundai_d])),
      HyundaiCarInfo("Kia Niro Hybrid 2022", car_parts=CarParts.common([CarHarness.hyundai_f])),
    ],
    flags=HyundaiFlags.HYBRID
  )
  KIA_NIRO_HEV_2ND_GEN = HyundaiPlatformConfig(
    "KIA NIRO HYBRID 2ND GEN",
    HyundaiCarInfo("Kia Niro Hybrid 2023", car_parts=CarParts.common([CarHarness.hyundai_a])),
    flags=HyundaiFlags.CANFD
  )
  KIA_OPTIMA_G4 = HyundaiPlatformConfig(
    "KIA OPTIMA 4TH GEN",
    HyundaiCarInfo("Kia Optima 2017", "Advanced Smart Cruise Control",
                                    car_parts=CarParts.common([CarHarness.hyundai_b])),  # TODO: may support 2016, 2018
    flags=HyundaiFlags.LEGACY | HyundaiFlags.TCU_GEARS
  )
  KIA_OPTIMA_G4_FL = HyundaiPlatformConfig(
    "KIA OPTIMA 4TH GEN FACELIFT",
    HyundaiCarInfo("Kia Optima 2019-20", car_parts=CarParts.common([CarHarness.hyundai_g])),
    flags=HyundaiFlags.UNSUPPORTED_LONGITUDINAL | HyundaiFlags.TCU_GEARS
  )
  # TODO: may support adjacent years. may have a non-zero minimum steering speed
  KIA_OPTIMA_H = HyundaiPlatformConfig(
    "KIA OPTIMA HYBRID 2017 & SPORTS 2019",
    HyundaiCarInfo("Kia Optima Hybrid 2017", "Advanced Smart Cruise Control", car_parts=CarParts.common([CarHarness.hyundai_c])),
    flags=HyundaiFlags.HYBRID | HyundaiFlags.LEGACY
  )
  KIA_OPTIMA_H_G4_FL = HyundaiPlatformConfig(
    "KIA OPTIMA HYBRID 4TH GEN FACELIFT",
    HyundaiCarInfo("Kia Optima Hybrid 2019", car_parts=CarParts.common([CarHarness.hyundai_h])),
    flags=HyundaiFlags.HYBRID | HyundaiFlags.UNSUPPORTED_LONGITUDINAL
  )
  KIA_SELTOS = HyundaiPlatformConfig(
    "KIA SELTOS 2021",
    HyundaiCarInfo("Kia Seltos 2021", car_parts=CarParts.common([CarHarness.hyundai_a])),
    flags=HyundaiFlags.CHECKSUM_CRC8
  )
  KIA_SPORTAGE_5TH_GEN = HyundaiPlatformConfig(
    "KIA SPORTAGE 5TH GEN",
    [
      HyundaiCarInfo("Kia Sportage 2023", car_parts=CarParts.common([CarHarness.hyundai_n])),
      HyundaiCarInfo("Kia Sportage Hybrid 2023", car_parts=CarParts.common([CarHarness.hyundai_n])),
    ],
    flags=HyundaiFlags.CANFD
  )
  KIA_SORENTO = HyundaiPlatformConfig(
    "KIA SORENTO GT LINE 2018",
    [
      HyundaiCarInfo("Kia Sorento 2018", "Advanced Smart Cruise Control & LKAS", video_link="https://www.youtube.com/watch?v=Fkh3s6WHJz8",
                   car_parts=CarParts.common([CarHarness.hyundai_e])),
      HyundaiCarInfo("Kia Sorento 2019", video_link="https://www.youtube.com/watch?v=Fkh3s6WHJz8", car_parts=CarParts.common([CarHarness.hyundai_e])),
    ],
    flags=HyundaiFlags.CHECKSUM_6B | HyundaiFlags.UNSUPPORTED_LONGITUDINAL
  )
  KIA_SORENTO_4TH_GEN = HyundaiPlatformConfig(
    "KIA SORENTO 4TH GEN",
    HyundaiCarInfo("Kia Sorento 2021-23", car_parts=CarParts.common([CarHarness.hyundai_k])),
    flags=HyundaiFlags.CANFD | HyundaiFlags.RADAR_SCC
  )
  KIA_SORENTO_HEV_4TH_GEN = HyundaiPlatformConfig(
    "KIA SORENTO HYBRID 4TH GEN",
    [
      HyundaiCarInfo("Kia Sorento Hybrid 2021-23", "All", car_parts=CarParts.common([CarHarness.hyundai_a])),
      HyundaiCarInfo("Kia Sorento Plug-in Hybrid 2022-23", "All", car_parts=CarParts.common([CarHarness.hyundai_a])),
    ],
    flags=HyundaiFlags.CANFD | HyundaiFlags.RADAR_SCC
  )
  KIA_STINGER = HyundaiPlatformConfig(
    "KIA STINGER GT2 2018",
    HyundaiCarInfo("Kia Stinger 2018-20", video_link="https://www.youtube.com/watch?v=MJ94qoofYw0",
                                  car_parts=CarParts.common([CarHarness.hyundai_c]))
  )
  KIA_STINGER_2022 = HyundaiPlatformConfig(
    "KIA STINGER 2022",
    HyundaiCarInfo("Kia Stinger 2022-23", "All", car_parts=CarParts.common([CarHarness.hyundai_k])),
  )
  KIA_CEED = HyundaiPlatformConfig(
    "KIA CEED INTRO ED 2019",
    HyundaiCarInfo("Kia Ceed 2019", car_parts=CarParts.common([CarHarness.hyundai_e])),
    flags=HyundaiFlags.LEGACY
  )
  KIA_EV6 = HyundaiPlatformConfig(
    "KIA EV6 2022",
    [
      HyundaiCarInfo("Kia EV6 (Southeast Asia only) 2022-23", "All", car_parts=CarParts.common([CarHarness.hyundai_p])),
      HyundaiCarInfo("Kia EV6 (without HDA II) 2022-23", "Highway Driving Assist", car_parts=CarParts.common([CarHarness.hyundai_l])),
      HyundaiCarInfo("Kia EV6 (with HDA II) 2022-23", "Highway Driving Assist II", car_parts=CarParts.common([CarHarness.hyundai_p]))
    ],
    flags=HyundaiFlags.CANFD | HyundaiFlags.EV
  )
  KIA_CARNIVAL_4TH_GEN = HyundaiPlatformConfig(
    "KIA CARNIVAL 4TH GEN",
    [
      HyundaiCarInfo("Kia Carnival 2022-24", car_parts=CarParts.common([CarHarness.hyundai_a])),
      HyundaiCarInfo("Kia Carnival (China only) 2023", car_parts=CarParts.common([CarHarness.hyundai_k]))
    ],
    flags=HyundaiFlags.CANFD | HyundaiFlags.RADAR_SCC
  )

  # Genesis
  GENESIS_GV60_EV_1ST_GEN = HyundaiPlatformConfig(
    "GENESIS GV60 ELECTRIC 1ST GEN",
    [
      HyundaiCarInfo("Genesis GV60 (Advanced Trim) 2023", "All", car_parts=CarParts.common([CarHarness.hyundai_a])),
      HyundaiCarInfo("Genesis GV60 (Performance Trim) 2023", "All", car_parts=CarParts.common([CarHarness.hyundai_k])),
    ],
    flags=HyundaiFlags.CANFD | HyundaiFlags.EV
  )
  GENESIS_G70 = HyundaiPlatformConfig(
    "GENESIS G70 2018",
    HyundaiCarInfo("Genesis G70 2018-19", "All", car_parts=CarParts.common([CarHarness.hyundai_f])),
    flags=HyundaiFlags.LEGACY
  )
  GENESIS_G70_2020 = HyundaiPlatformConfig(
    "GENESIS G70 2020",
    HyundaiCarInfo("Genesis G70 2020", "All", car_parts=CarParts.common([CarHarness.hyundai_f])),
    flags=HyundaiFlags.MANDO_RADAR
  )
  GENESIS_GV70_1ST_GEN = HyundaiPlatformConfig(
    "GENESIS GV70 1ST GEN",
    [
      HyundaiCarInfo("Genesis GV70 (2.5T Trim) 2022-23", "All", car_parts=CarParts.common([CarHarness.hyundai_l])),
      HyundaiCarInfo("Genesis GV70 (3.5T Trim) 2022-23", "All", car_parts=CarParts.common([CarHarness.hyundai_m])),
    ],
    flags=HyundaiFlags.CANFD | HyundaiFlags.RADAR_SCC
  )
  GENESIS_G80 = HyundaiPlatformConfig(
    "GENESIS G80 2017",
    HyundaiCarInfo("Genesis G80 2018-19", "All", car_parts=CarParts.common([CarHarness.hyundai_h])),
    flags=HyundaiFlags.LEGACY
  )
  GENESIS_G90 = HyundaiPlatformConfig(
    "GENESIS G90 2017",
    HyundaiCarInfo("Genesis G90 2017-18", "All", car_parts=CarParts.common([CarHarness.hyundai_c])),
  )
  GENESIS_GV80 = HyundaiPlatformConfig(
    "GENESIS GV80 2023",
    HyundaiCarInfo("Genesis GV80 2023", "All", car_parts=CarParts.common([CarHarness.hyundai_m])),
    flags=HyundaiFlags.CANFD | HyundaiFlags.RADAR_SCC
  )


class Footnote(Enum):
  CANFD = CarFootnote(
    "Requires a <a href=\"https://comma.ai/shop/can-fd-panda-kit\" target=\"_blank\">CAN FD panda kit</a> if not using " +
    "comma 3X for this <a href=\"https://en.wikipedia.org/wiki/CAN_FD\" target=\"_blank\">CAN FD car</a>.",
    Column.MODEL, shop_footnote=False)


class Buttons:
  NONE = 0
  RES_ACCEL = 1
  SET_DECEL = 2
  GAP_DIST = 3
  CANCEL = 4  # on newer models, this is a pause/resume button


def get_platform_codes(fw_versions: list[bytes]) -> set[tuple[bytes, bytes | None]]:
  # Returns unique, platform-specific identification codes for a set of versions
  codes = set()  # (code-Optional[part], date)
  for fw in fw_versions:
    code_match = PLATFORM_CODE_FW_PATTERN.search(fw)
    part_match = PART_NUMBER_FW_PATTERN.search(fw)
    date_match = DATE_FW_PATTERN.search(fw)
    if code_match is not None:
      code: bytes = code_match.group()
      part = part_match.group() if part_match else None
      date = date_match.group() if date_match else None
      if part is not None:
        # part number starts with generic ECU part type, add what is specific to platform
        code += b"-" + part[-5:]

      codes.add((code, date))
  return codes


def match_fw_to_car_fuzzy(live_fw_versions, offline_fw_versions) -> set[str]:
  # Non-electric CAN FD platforms often do not have platform code specifiers needed
  # to distinguish between hybrid and ICE. All EVs so far are either exclusively
  # electric or specify electric in the platform code.
  fuzzy_platform_blacklist = {str(c) for c in (CANFD_CAR - EV_CAR - CANFD_FUZZY_WHITELIST)}
  candidates: set[str] = set()

  for candidate, fws in offline_fw_versions.items():
    # Keep track of ECUs which pass all checks (platform codes, within date range)
    valid_found_ecus = set()
    valid_expected_ecus = {ecu[1:] for ecu in fws if ecu[0] in PLATFORM_CODE_ECUS}
    for ecu, expected_versions in fws.items():
      addr = ecu[1:]
      # Only check ECUs expected to have platform codes
      if ecu[0] not in PLATFORM_CODE_ECUS:
        continue

      # Expected platform codes & dates
      codes = get_platform_codes(expected_versions)
      expected_platform_codes = {code for code, _ in codes}
      expected_dates = {date for _, date in codes if date is not None}

      # Found platform codes & dates
      codes = get_platform_codes(live_fw_versions.get(addr, set()))
      found_platform_codes = {code for code, _ in codes}
      found_dates = {date for _, date in codes if date is not None}

      # Check platform code + part number matches for any found versions
      if not any(found_platform_code in expected_platform_codes for found_platform_code in found_platform_codes):
        break

      if ecu[0] in DATE_FW_ECUS:
        # If ECU can have a FW date, require it to exist
        # (this excludes candidates in the database without dates)
        if not len(expected_dates) or not len(found_dates):
          break

        # Check any date within range in the database, format is %y%m%d
        if not any(min(expected_dates) <= found_date <= max(expected_dates) for found_date in found_dates):
          break

      valid_found_ecus.add(addr)

    # If all live ECUs pass all checks for candidate, add it as a match
    if valid_expected_ecus.issubset(valid_found_ecus):
      candidates.add(candidate)

  return candidates - fuzzy_platform_blacklist


HYUNDAI_VERSION_REQUEST_LONG = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER]) + \
  p16(0xf100)  # Long description

HYUNDAI_VERSION_REQUEST_ALT = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER]) + \
  p16(0xf110)  # Alt long description

HYUNDAI_VERSION_REQUEST_MULTI = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER]) + \
  p16(uds.DATA_IDENTIFIER_TYPE.VEHICLE_MANUFACTURER_SPARE_PART_NUMBER) + \
  p16(uds.DATA_IDENTIFIER_TYPE.APPLICATION_SOFTWARE_IDENTIFICATION) + \
  p16(0xf100)

HYUNDAI_ECU_MANUFACTURING_DATE = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER]) + \
  p16(uds.DATA_IDENTIFIER_TYPE.ECU_MANUFACTURING_DATE)

HYUNDAI_VERSION_RESPONSE = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER + 0x40])

# Regex patterns for parsing platform code, FW date, and part number from FW versions
PLATFORM_CODE_FW_PATTERN = re.compile(b'((?<=' + HYUNDAI_VERSION_REQUEST_LONG[1:] +
                                      b')[A-Z]{2}[A-Za-z0-9]{0,2})')
DATE_FW_PATTERN = re.compile(b'(?<=[ -])([0-9]{6}$)')
PART_NUMBER_FW_PATTERN = re.compile(b'(?<=[0-9][.,][0-9]{2} )([0-9]{5}[-/]?[A-Z][A-Z0-9]{3}[0-9])')

# We've seen both ICE and hybrid for these platforms, and they have hybrid descriptors (e.g. MQ4 vs MQ4H)
CANFD_FUZZY_WHITELIST = {CAR.KIA_SORENTO_4TH_GEN, CAR.KIA_SORENTO_HEV_4TH_GEN}

# List of ECUs expected to have platform codes, camera and radar should exist on all cars
# TODO: use abs, it has the platform code and part number on many platforms
PLATFORM_CODE_ECUS = [Ecu.fwdRadar, Ecu.fwdCamera, Ecu.eps]
# So far we've only seen dates in fwdCamera
# TODO: there are date codes in the ABS firmware versions in hex
DATE_FW_ECUS = [Ecu.fwdCamera]

ALL_HYUNDAI_ECUS = [Ecu.eps, Ecu.abs, Ecu.fwdRadar, Ecu.fwdCamera, Ecu.engine, Ecu.parkingAdas, Ecu.transmission, Ecu.adas, Ecu.hvac, Ecu.cornerRadar]

FW_QUERY_CONFIG = FwQueryConfig(
  requests=[
    # TODO: minimize shared whitelists for CAN and cornerRadar for CAN-FD
    # CAN queries (OBD-II port)
    Request(
      [HYUNDAI_VERSION_REQUEST_LONG],
      [HYUNDAI_VERSION_RESPONSE],
      whitelist_ecus=[Ecu.transmission, Ecu.eps, Ecu.abs, Ecu.fwdRadar, Ecu.fwdCamera],
    ),
    Request(
      [HYUNDAI_VERSION_REQUEST_MULTI],
      [HYUNDAI_VERSION_RESPONSE],
      whitelist_ecus=[Ecu.engine, Ecu.transmission, Ecu.eps, Ecu.abs, Ecu.fwdRadar],
    ),

    # CAN-FD queries (from camera)
    # TODO: combine shared whitelists with CAN requests
    Request(
      [HYUNDAI_VERSION_REQUEST_LONG],
      [HYUNDAI_VERSION_RESPONSE],
      whitelist_ecus=[Ecu.fwdCamera, Ecu.fwdRadar, Ecu.cornerRadar, Ecu.hvac],
      bus=0,
      auxiliary=True,
    ),
    Request(
      [HYUNDAI_VERSION_REQUEST_LONG],
      [HYUNDAI_VERSION_RESPONSE],
      whitelist_ecus=[Ecu.fwdCamera, Ecu.adas, Ecu.cornerRadar, Ecu.hvac],
      bus=1,
      auxiliary=True,
      obd_multiplexing=False,
    ),

    # CAN & CAN FD query to understand the three digit date code
    # HDA2 cars usually use 6 digit date codes, so skip bus 1
    Request(
      [HYUNDAI_ECU_MANUFACTURING_DATE],
      [HYUNDAI_VERSION_RESPONSE],
      whitelist_ecus=[Ecu.fwdCamera],
      bus=0,
      auxiliary=True,
      logging=True,
    ),

    # CAN & CAN FD logging queries (from camera)
    Request(
      [HYUNDAI_VERSION_REQUEST_LONG],
      [HYUNDAI_VERSION_RESPONSE],
      whitelist_ecus=ALL_HYUNDAI_ECUS,
      bus=0,
      auxiliary=True,
      logging=True,
    ),
    Request(
      [HYUNDAI_VERSION_REQUEST_MULTI],
      [HYUNDAI_VERSION_RESPONSE],
      whitelist_ecus=ALL_HYUNDAI_ECUS,
      bus=0,
      auxiliary=True,
      logging=True,
    ),
    Request(
      [HYUNDAI_VERSION_REQUEST_LONG],
      [HYUNDAI_VERSION_RESPONSE],
      whitelist_ecus=ALL_HYUNDAI_ECUS,
      bus=1,
      auxiliary=True,
      obd_multiplexing=False,
      logging=True,
    ),

    # CAN-FD alt request logging queries
    Request(
      [HYUNDAI_VERSION_REQUEST_ALT],
      [HYUNDAI_VERSION_RESPONSE],
      whitelist_ecus=[Ecu.parkingAdas, Ecu.hvac],
      bus=0,
      auxiliary=True,
      logging=True,
    ),
    Request(
      [HYUNDAI_VERSION_REQUEST_ALT],
      [HYUNDAI_VERSION_RESPONSE],
      whitelist_ecus=[Ecu.parkingAdas, Ecu.hvac],
      bus=1,
      auxiliary=True,
      logging=True,
      obd_multiplexing=False,
    ),
  ],
  extra_ecus=[
    (Ecu.adas, 0x730, None),         # ADAS Driving ECU on HDA2 platforms
    (Ecu.parkingAdas, 0x7b1, None),  # ADAS Parking ECU (may exist on all platforms)
    (Ecu.hvac, 0x7b3, None),         # HVAC Control Assembly
    (Ecu.cornerRadar, 0x7b7, None),
  ],
  # Custom fuzzy fingerprinting function using platform codes, part numbers + FW dates:
  match_fw_to_car_fuzzy=match_fw_to_car_fuzzy,
)

CHECKSUM = {
  "crc8": CAR.with_flags(HyundaiFlags.CHECKSUM_CRC8),
  "6b": CAR.with_flags(HyundaiFlags.CHECKSUM_6B),
}

CAN_GEARS = {
  # which message has the gear. hybrid and EV use ELECT_GEAR
  "use_cluster_gears": CAR.with_flags(HyundaiFlags.CLUSTER_GEARS),
  "use_tcu_gears": CAR.with_flags(HyundaiFlags.TCU_GEARS),
}

CANFD_CAR = CAR.with_flags(HyundaiFlags.CANFD)
CANFD_RADAR_SCC_CAR = CAR.with_flags(HyundaiFlags.RADAR_SCC)

# These CAN FD cars do not accept communication control to disable the ADAS ECU,
# responds with 0x7F2822 - 'conditions not correct'
CANFD_UNSUPPORTED_LONGITUDINAL_CAR = CAR.with_flags(HyundaiFlags.CANFD_NO_RADAR_DISABLE)

# The camera does SCC on these cars, rather than the radar
CAMERA_SCC_CAR = CAR.with_flags(HyundaiFlags.CAMERA_SCC)

HYBRID_CAR = CAR.with_flags(HyundaiFlags.HYBRID)

EV_CAR = CAR.with_flags(HyundaiFlags.EV)

LEGACY_SAFETY_MODE_CAR = CAR.with_flags(HyundaiFlags.LEGACY)

UNSUPPORTED_LONGITUDINAL_CAR = CAR.with_flags(HyundaiFlags.LEGACY) | CAR.with_flags(HyundaiFlags.UNSUPPORTED_LONGITUDINAL)

CAR_INFO = CAR.create_carinfo_map()
DBC = CAR.create_dbc_map()

if __name__ == "__main__":
  CAR.print_debug(HyundaiFlags)
