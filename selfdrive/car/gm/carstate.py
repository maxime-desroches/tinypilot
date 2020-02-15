from cereal import car
from common.numpy_fast import mean
from selfdrive.config import Conversions as CV
from opendbc.can.parser import CANParser
from selfdrive.car.interfaces import CarStateBase
from selfdrive.car.gm.values import DBC, CAR, parse_gear_shifter, \
                                    CruiseButtons, is_eps_status_ok, \
                                    STEER_THRESHOLD, SUPERCRUISE_CARS

def get_powertrain_can_parser(CP, canbus):
  # this function generates lists for signal, messages and initial values
  signals = [
    # sig_name, sig_address, default
    ("BrakePedalPosition", "EBCMBrakePedalPosition", 0),
    ("FrontLeftDoor", "BCMDoorBeltStatus", 0),
    ("FrontRightDoor", "BCMDoorBeltStatus", 0),
    ("RearLeftDoor", "BCMDoorBeltStatus", 0),
    ("RearRightDoor", "BCMDoorBeltStatus", 0),
    ("LeftSeatBelt", "BCMDoorBeltStatus", 0),
    ("RightSeatBelt", "BCMDoorBeltStatus", 0),
    ("TurnSignals", "BCMTurnSignals", 0),
    ("AcceleratorPedal", "AcceleratorPedal", 0),
    ("ACCButtons", "ASCMSteeringButton", CruiseButtons.UNPRESS),
    ("SteeringWheelAngle", "PSCMSteeringAngle", 0),
    ("FLWheelSpd", "EBCMWheelSpdFront", 0),
    ("FRWheelSpd", "EBCMWheelSpdFront", 0),
    ("RLWheelSpd", "EBCMWheelSpdRear", 0),
    ("RRWheelSpd", "EBCMWheelSpdRear", 0),
    ("PRNDL", "ECMPRDNL", 0),
    ("LKADriverAppldTrq", "PSCMStatus", 0),
    ("LKATorqueDeliveredStatus", "PSCMStatus", 0),
  ]

  if CP.carFingerprint == CAR.VOLT:
    signals += [
      ("RegenPaddle", "EBCMRegenPaddle", 0),
    ]
  if CP.carFingerprint in SUPERCRUISE_CARS:
    signals += [
      ("ACCCmdActive", "ASCMActiveCruiseControlStatus", 0)
    ]
  else:
    signals += [
      ("TractionControlOn", "ESPStatus", 0),
      ("EPBClosed", "EPBStatus", 0),
      ("CruiseMainOn", "ECMEngineStatus", 0),
      ("CruiseState", "AcceleratorPedal2", 0),
    ]

  return CANParser(DBC[CP.carFingerprint]['pt'], signals, [], canbus.powertrain)


class CarState(CarStateBase):
  def __init__(self, CP, canbus):
    super().__init__()
    self.CP = CP
    # initialize can parser

    self.car_fingerprint = CP.carFingerprint
    self.cruise_buttons = CruiseButtons.UNPRESS
    self.left_blinker_on = False
    self.prev_left_blinker_on = False
    self.right_blinker_on = False
    self.prev_right_blinker_on = False

  def update(self, pt_cp):
    self.prev_cruise_buttons = self.cruise_buttons
    self.cruise_buttons = pt_cp.vl["ASCMSteeringButton"]['ACCButtons']

    self.v_wheel_fl = pt_cp.vl["EBCMWheelSpdFront"]['FLWheelSpd'] * CV.KPH_TO_MS
    self.v_wheel_fr = pt_cp.vl["EBCMWheelSpdFront"]['FRWheelSpd'] * CV.KPH_TO_MS
    self.v_wheel_rl = pt_cp.vl["EBCMWheelSpdRear"]['RLWheelSpd'] * CV.KPH_TO_MS
    self.v_wheel_rr = pt_cp.vl["EBCMWheelSpdRear"]['RRWheelSpd'] * CV.KPH_TO_MS
    self.v_ego_raw = mean([self.v_wheel_fl, self.v_wheel_fr, self.v_wheel_rl, self.v_wheel_rr])
    self.v_ego, self.a_ego = self.update_speed_kf(self.v_ego_raw)
    self.standstill = self.v_ego_raw < 0.01

    self.angle_steers = pt_cp.vl["PSCMSteeringAngle"]['SteeringWheelAngle']
    self.gear_shifter = parse_gear_shifter(pt_cp.vl["ECMPRDNL"]['PRNDL'])
    self.user_brake = pt_cp.vl["EBCMBrakePedalPosition"]['BrakePedalPosition']

    self.pedal_gas = pt_cp.vl["AcceleratorPedal"]['AcceleratorPedal']
    self.user_gas_pressed = self.pedal_gas > 0

    self.steer_torque_driver = pt_cp.vl["PSCMStatus"]['LKADriverAppldTrq']
    self.steer_override = abs(self.steer_torque_driver) > STEER_THRESHOLD

    # 0 - inactive, 1 - active, 2 - temporary limited, 3 - failed
    self.lkas_status = pt_cp.vl["PSCMStatus"]['LKATorqueDeliveredStatus']
    self.steer_not_allowed = not is_eps_status_ok(self.lkas_status, self.car_fingerprint)

    # 1 - open, 0 - closed
    self.door_all_closed = (pt_cp.vl["BCMDoorBeltStatus"]['FrontLeftDoor'] == 0 and
      pt_cp.vl["BCMDoorBeltStatus"]['FrontRightDoor'] == 0 and
      pt_cp.vl["BCMDoorBeltStatus"]['RearLeftDoor'] == 0 and
      pt_cp.vl["BCMDoorBeltStatus"]['RearRightDoor'] == 0)

    # 1 - latched
    self.seatbelt = pt_cp.vl["BCMDoorBeltStatus"]['LeftSeatBelt'] == 1

    self.steer_error = False

    self.brake_error = False

    self.prev_left_blinker_on = self.left_blinker_on
    self.prev_right_blinker_on = self.right_blinker_on
    self.left_blinker_on = pt_cp.vl["BCMTurnSignals"]['TurnSignals'] == 1
    self.right_blinker_on = pt_cp.vl["BCMTurnSignals"]['TurnSignals'] == 2

    if self.car_fingerprint in SUPERCRUISE_CARS:
      self.park_brake = False
      self.main_on = False
      self.acc_active = pt_cp.vl["ASCMActiveCruiseControlStatus"]['ACCCmdActive']
      self.esp_disabled = False
      self.regen_pressed = False
      self.pcm_acc_status = int(self.acc_active)
    else:
      self.park_brake = pt_cp.vl["EPBStatus"]['EPBClosed']
      self.main_on = pt_cp.vl["ECMEngineStatus"]['CruiseMainOn']
      self.acc_active = False
      self.esp_disabled = pt_cp.vl["ESPStatus"]['TractionControlOn'] != 1
      self.pcm_acc_status = pt_cp.vl["AcceleratorPedal2"]['CruiseState']
      if self.car_fingerprint == CAR.VOLT:
        self.regen_pressed = bool(pt_cp.vl["EBCMRegenPaddle"]['RegenPaddle'])
      else:
        self.regen_pressed = False

    # Brake pedal's potentiometer returns near-zero reading
    # even when pedal is not pressed.
    if self.user_brake < 10:
      self.user_brake = 0

    # Regen braking is braking
    self.brake_pressed = self.user_brake > 10 or self.regen_pressed

    self.gear_shifter_valid = self.gear_shifter == car.CarState.GearShifter.drive
