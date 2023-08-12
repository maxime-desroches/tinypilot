import copy
from cereal import car
from opendbc.can.can_define import CANDefine
from common.conversions import Conversions as CV
from selfdrive.car.interfaces import CarStateBase
from opendbc.can.parser import CANParser
from selfdrive.car.subaru.values import DBC, CAR, GLOBAL_GEN2, PREGLOBAL_CARS, CanBus, SubaruFlags


class CarState(CarStateBase):
  def __init__(self, CP):
    super().__init__(CP)
    can_define = CANDefine(DBC[CP.carFingerprint]["pt"])
    self.shifter_values = can_define.dv["Transmission"]["Gear"]

  def update(self, cp, cp_cam, cp_body):
    ret = car.CarState.new_message()

    ret.gas = cp.vl["Throttle"]["Throttle_Pedal"] / 255.
    ret.gasPressed = ret.gas > 1e-5
    if self.car_fingerprint in PREGLOBAL_CARS:
      ret.brakePressed = cp.vl["Brake_Pedal"]["Brake_Pedal"] > 2
    else:
      cp_brakes = cp_body if self.car_fingerprint in GLOBAL_GEN2 else cp
      ret.brakePressed = cp_brakes.vl["Brake_Status"]["Brake"] == 1

    cp_wheels = cp_body if self.car_fingerprint in GLOBAL_GEN2 else cp
    ret.wheelSpeeds = self.get_wheel_speeds(
      cp_wheels.vl["Wheel_Speeds"]["FL"],
      cp_wheels.vl["Wheel_Speeds"]["FR"],
      cp_wheels.vl["Wheel_Speeds"]["RL"],
      cp_wheels.vl["Wheel_Speeds"]["RR"],
    )
    ret.vEgoRaw = (ret.wheelSpeeds.fl + ret.wheelSpeeds.fr + ret.wheelSpeeds.rl + ret.wheelSpeeds.rr) / 4.
    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)
    ret.standstill = ret.vEgoRaw == 0

    # continuous blinker signals for assisted lane change
    ret.leftBlinker, ret.rightBlinker = self.update_blinker_from_lamp(50, cp.vl["Dashlights"]["LEFT_BLINKER"],
                                                                          cp.vl["Dashlights"]["RIGHT_BLINKER"])

    if self.CP.enableBsm:
      ret.leftBlindspot = (cp.vl["BSD_RCTA"]["L_ADJACENT"] == 1) or (cp.vl["BSD_RCTA"]["L_APPROACHING"] == 1)
      ret.rightBlindspot = (cp.vl["BSD_RCTA"]["R_ADJACENT"] == 1) or (cp.vl["BSD_RCTA"]["R_APPROACHING"] == 1)

    can_gear = int(cp.vl["Transmission"]["Gear"])
    ret.gearShifter = self.parse_gear_shifter(self.shifter_values.get(can_gear, None))

    ret.steeringAngleDeg = cp.vl["Steering_Torque"]["Steering_Angle"]
    ret.steeringTorque = cp.vl["Steering_Torque"]["Steer_Torque_Sensor"]
    ret.steeringTorqueEps = cp.vl["Steering_Torque"]["Steer_Torque_Output"]

    steer_threshold = 75 if self.CP.carFingerprint in PREGLOBAL_CARS else 80
    ret.steeringPressed = abs(ret.steeringTorque) > steer_threshold

    cp_cruise = cp_body if self.car_fingerprint in GLOBAL_GEN2 else cp
    ret.cruiseState.enabled = cp_cruise.vl["CruiseControl"]["Cruise_Activated"] != 0
    ret.cruiseState.available = cp_cruise.vl["CruiseControl"]["Cruise_On"] != 0
    ret.cruiseState.speed = cp_cam.vl["ES_DashStatus"]["Cruise_Set_Speed"] * CV.KPH_TO_MS

    if (self.car_fingerprint in PREGLOBAL_CARS and cp.vl["Dash_State2"]["UNITS"] == 1) or \
       (self.car_fingerprint not in PREGLOBAL_CARS and cp.vl["Dashlights"]["UNITS"] == 1):
      ret.cruiseState.speed *= CV.MPH_TO_KPH

    ret.seatbeltUnlatched = cp.vl["Dashlights"]["SEATBELT_FL"] == 1
    ret.doorOpen = any([cp.vl["BodyInfo"]["DOOR_OPEN_RR"],
                        cp.vl["BodyInfo"]["DOOR_OPEN_RL"],
                        cp.vl["BodyInfo"]["DOOR_OPEN_FR"],
                        cp.vl["BodyInfo"]["DOOR_OPEN_FL"]])
    ret.steerFaultPermanent = cp.vl["Steering_Torque"]["Steer_Error_1"] == 1

    cp_es_distance = cp_body if self.car_fingerprint in GLOBAL_GEN2 else cp_cam
    if self.car_fingerprint in PREGLOBAL_CARS:
      self.cruise_button = cp_cam.vl["ES_Distance"]["Cruise_Button"]
      self.ready = not cp_cam.vl["ES_DashStatus"]["Not_Ready_Startup"]
    else:
      ret.steerFaultTemporary = cp.vl["Steering_Torque"]["Steer_Warning"] == 1
      ret.cruiseState.nonAdaptive = cp_cam.vl["ES_DashStatus"]["Conventional_Cruise"] == 1
      ret.cruiseState.standstill = cp_cam.vl["ES_DashStatus"]["Cruise_State"] == 3
      ret.stockFcw = (cp_cam.vl["ES_LKAS_State"]["LKAS_Alert"] == 1) or \
                     (cp_cam.vl["ES_LKAS_State"]["LKAS_Alert"] == 2)
      # 8 is known AEB, there are a few other values related to AEB we ignore
      ret.stockAeb = (cp_es_distance.vl["ES_Brake"]["AEB_Status"] == 8) and \
                     (cp_es_distance.vl["ES_Brake"]["Brake_Pressure"] != 0)
      self.es_lkas_state_msg = copy.copy(cp_cam.vl["ES_LKAS_State"])

    self.es_distance_msg = copy.copy(cp_es_distance.vl["ES_Distance"])
    self.es_dashstatus_msg = copy.copy(cp_cam.vl["ES_DashStatus"])
    if self.CP.flags & SubaruFlags.SEND_INFOTAINMENT:
      self.es_infotainment_msg = copy.copy(cp_cam.vl["ES_Infotainment"])

    return ret

  @staticmethod
  def get_common_global_body_messages():
    messages = [
      ("CruiseControl", 20),
      ("Wheel_Speeds", 50),
      ("Brake_Status", 50),
    ]

    return messages

  @staticmethod
  def get_common_global_es_messages():
    messages = [
      ("ES_Brake", 20),
      ("ES_Distance", 20),
    ]

    return messages

  @staticmethod
  def get_can_parser(CP):
    messages = [
      # sig_address, frequency
      ("Throttle", 100),
      ("Dashlights", 10),
      ("Brake_Pedal", 50),
      ("Transmission", 100),
      ("Steering_Torque", 50),
      ("BodyInfo", 1),
    ]

    if CP.enableBsm:
      messages.append(("BSD_RCTA", 17))

    if CP.carFingerprint not in PREGLOBAL_CARS:
      if CP.carFingerprint not in GLOBAL_GEN2:
        messages += CarState.get_common_global_body_messages()

      messages += [
        ("Dashlights", 10),
        ("BodyInfo", 10),
      ]
    else:
      messages += [
        ("Wheel_Speeds", 50),
        ("Dash_State2", 1),
      ]

      if CP.carFingerprint == CAR.FORESTER_PREGLOBAL:
        messages += [
          ("Dashlights", 20),
          ("BodyInfo", 1),
          ("CruiseControl", 50),
        ]

      if CP.carFingerprint in (CAR.LEGACY_PREGLOBAL, CAR.OUTBACK_PREGLOBAL, CAR.OUTBACK_PREGLOBAL_2018):
        messages += [
          ("Dashlights", 10),
          ("CruiseControl", 50),
        ]

    return CANParser(DBC[CP.carFingerprint]["pt"], messages, CanBus.main)

  @staticmethod
  def get_cam_can_parser(CP):
    if CP.carFingerprint in PREGLOBAL_CARS:
      messages = [
        ("ES_DashStatus", 20),
        ("ES_Distance", 20),
      ]
    else:
      messages = [
        ("ES_DashStatus", 10),
        ("ES_LKAS_State", 10),
      ]

      if CP.carFingerprint not in GLOBAL_GEN2:
        messages += CarState.get_common_global_es_messages()

      if CP.flags & SubaruFlags.SEND_INFOTAINMENT:
        messages.append(("ES_Infotainment", 10))

    return CANParser(DBC[CP.carFingerprint]["pt"], messages, CanBus.camera)

  @staticmethod
  def get_body_can_parser(CP):
    if CP.carFingerprint in GLOBAL_GEN2:
      messages = CarState.get_common_global_body_messages()
      messages += CarState.get_common_global_es_messages()
      return CANParser(DBC[CP.carFingerprint]["pt"], messages, CanBus.alt)

    return None
