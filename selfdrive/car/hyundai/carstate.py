from selfdrive.car.hyundai.values import CAR, DBC
from selfdrive.can.parser import CANParser
from selfdrive.config import Conversions as CV
from common.kalman.simple_kalman import KF1D
import numpy as np


def get_can_parser(CP):

  signals = [
    # sig_name, sig_address, default
    ("WHL_SPD_FL", "WHL_SPD11", 0),
    ("WHL_SPD_FR", "WHL_SPD11", 0),
    ("WHL_SPD_RL", "WHL_SPD11", 0),
    ("WHL_SPD_RR", "WHL_SPD11", 0),

    ("YAW_RATE", "ESP12", 0),
    
    ("CF_Gway_DrvSeatBeltInd", "CGW4", 1),
    ("CF_Gway_DrvSeatBeltSw", "CGW1", 0),
    ("CF_Gway_TSigLHSw", "CGW1", 0),
    ("CF_Gway_TurnSigLh", "CGW1", 0),
    ("CF_Gway_TSigRHSw", "CGW1", 0),
    ("CF_Gway_TurnSigRh", "CGW1", 0),

    ("BRAKE_ACT", "EMS12", 0),
    ("PV_AV_CAN", "EMS12", 0),
    ("TPS", "EMS12", 0),

    ("CYL_PRES", "ESP12", 0),

    ("CF_Clu_CruiseSwState", "CLU11", 0),
    ("CF_Clu_CruiseSwMain", "CLU11", 0),

    ("CF_Clu_InhibitD", "CLU15", 0),
    ("CF_Clu_InhibitP", "CLU15", 0),
    ("CF_Clu_InhibitN", "CLU15", 0),
    ("CF_Clu_InhibitR", "CLU15", 0),

    ("ACCEnable", "TCS13", 0),
    ("ACC_REQ", "TCS13", 0),
    ("DriverBraking", "TCS13", 0),
    ("DriverOverride", "TCS13", 0),

    ("ESC_Off_Step", "TCS15", 0),

    ("Gear", "AT01", 0),        #Transmission Gear (0 = N or P, 1-8 = Fwd, 14 = Rev)

    ("CR_Mdps_DrvTq", "MDPS11", 0),

    ("CR_Mdps_StrColTq", "MDPS12", 0),
    ("CF_Mdps_ToiActive", "MDPS12", 0),
    ("CF_Mdps_ToiUnavail", "MDPS12", 0),
    ("CF_Mdps_FailStat", "MDPS12", 0),
    ("CR_Mdps_OutTq", "MDPS12", 0),

    ("VSetDis", "SCC11", 0),
    ("ACCMode", "SCC12", 1),

    ("SAS_Angle", "SAS11", 0),
    ("SAS_Speed", "SAS11", 0),

  ]
  checks = [
    # address, frequency
    ("MDPS12", 50),
    ("MDPS11", 100),
    ("TCS15", 10),
    ("TCS13", 50),
    ("CLU11", 50),
    ("CLU15", 10),
    ("ESP12", 100), 
    ("EMS12", 100),
    ("CGW1", 10),
    ("CGW4", 5),
    ("WHL_SPD11", 50),
    ("SCC11", 50),
    ("SCC12", 50),
    ("SAS11", 100)
  ]

  return CANParser(DBC[CP.carFingerprint]['pt'], signals, checks, 0)


class CarState(object):
  def __init__(self, CP):

    self.CP = CP

    # initialize can parser
    self.car_fingerprint = CP.carFingerprint

    # vEgo kalman filter
    dt = 0.01
    # Q = np.matrix([[10.0, 0.0], [0.0, 100.0]])
    # R = 1e3
    self.v_ego_kf = KF1D(x0=np.matrix([[0.0], [0.0]]),
                         A=np.matrix([[1.0, dt], [0.0, 1.0]]),
                         C=np.matrix([1.0, 0.0]),
                         K=np.matrix([[0.12287673], [0.29666309]]))
    self.v_ego = 0.0
    self.left_blinker_on = 0
    self.left_blinker_flash = 0
    self.right_blinker_on = 0
    self.right_blinker_flash = 0

  def update(self, cp):
    # copy can_valid
    self.can_valid = cp.can_valid

    # update prevs, update must run once per Loop
    self.prev_left_blinker_on = self.left_blinker_on
    self.prev_right_blinker_on = self.right_blinker_on

    self.door_all_closed = True
    self.seatbelt = cp.vl["CGW1"]['CF_Gway_DrvSeatBeltSw']
    
    self.brake_pressed = cp.vl["TCS13"]['DriverBraking']
    self.esp_disabled = cp.vl["TCS15"]['ESC_Off_Step']

    self.park_brake = False
    self.main_on = True
    self.acc_active = cp.vl["SCC12"]['ACCMode'] == 1
    self.pcm_acc_status = int(self.acc_active)

    # calc best v_ego estimate, by averaging two opposite corners
    self.v_wheel_fl = cp.vl["WHL_SPD11"]['WHL_SPD_FL'] * CV.KPH_TO_MS
    self.v_wheel_fr = cp.vl["WHL_SPD11"]['WHL_SPD_FR'] * CV.KPH_TO_MS
    self.v_wheel_rl = cp.vl["WHL_SPD11"]['WHL_SPD_RL'] * CV.KPH_TO_MS
    self.v_wheel_rr = cp.vl["WHL_SPD11"]['WHL_SPD_RR'] * CV.KPH_TO_MS
    self.v_wheel = (self.v_wheel_fl + self.v_wheel_fr + self.v_wheel_rl + self.v_wheel_rr) / 4.

    # There is a 2% error on Sorento GT due to slightly larger wheel diameter compared to poverty packs.
    #   Dash assumes about 4% which is excessive. Using 3% to be safe
    if self.car_fingerprint == CAR.SORENTO:
      self.v_wheel = self.v_wheel * 1.03 
    self.low_speed_lockout = self.v_wheel < 1.0

    # Kalman filter
    if abs(self.v_wheel - self.v_ego) > 2.0:  # Prevent large accelerations when car starts at non zero speed
      self.v_ego_x = np.matrix([[self.v_wheel], [0.0]])

    self.v_ego_raw = self.v_wheel
    v_ego_x = self.v_ego_kf.update(self.v_wheel)
    self.v_ego = float(v_ego_x[0])
    self.a_ego = float(v_ego_x[1])
    self.cruise_set_speed = cp.vl["SCC11"]['VSetDis']
    self.standstill = not self.v_wheel > 0.001

    self.angle_steers = cp.vl["SAS11"]['SAS_Angle']
    self.angle_steers_rate = cp.vl["SAS11"]['SAS_Speed']
    self.yaw_rate = cp.vl["ESP12"]['YAW_RATE']
    self.main_on = True
    self.left_blinker_on = cp.vl["CGW1"]['CF_Gway_TSigLHSw']
    self.left_blinker_flash = cp.vl["CGW1"]['CF_Gway_TurnSigLh']
    self.right_blinker_on = cp.vl["CGW1"]['CF_Gway_TSigRHSw']
    self.right_blinker_flash = cp.vl["CGW1"]['CF_Gway_TurnSigRh']
    self.steer_override = abs(cp.vl["MDPS11"]['CR_Mdps_DrvTq']) > 1.0
    self.steer_state = cp.vl["MDPS12"]['CF_Mdps_ToiActive'] #0 NOT ACTIVE, 1 ACTIVE
    self.steer_error = cp.vl["MDPS12"]['CF_Mdps_ToiUnavail']
    self.brake_error = 0
    self.steer_torque_driver = cp.vl["MDPS11"]['CR_Mdps_DrvTq'] 
    self.steer_torque_motor = cp.vl["MDPS12"]['CR_Mdps_OutTq']

    self.user_brake = 0

    self.brake_pressed = cp.vl["TCS13"]['DriverBraking']
    self.brake_lights = bool(self.brake_pressed)
    if (cp.vl["TCS13"]["DriverOverride"] == 0 and cp.vl["TCS13"]['ACC_REQ'] == 1):
      self.pedal_gas = 0
    else: 
      self.pedal_gas = cp.vl["EMS12"]['TPS']
    self.car_gas = cp.vl["EMS12"]['TPS']

    # Gear Selecton - This should be compatible with all Kia/Hyundai with Auto's
    if cp.vl["CLU15"]["CF_Clu_InhibitD"] == 1:
      self.gear_shifter = "drive"
    elif cp.vl["CLU15"]["CF_Clu_InhibitN"] == 1:
      self.gear_shifter = "neutral"
    elif cp.vl["CLU15"]["CF_Clu_InhibitP"] == 1:
      self.gear_shifter = "park"
    elif cp.vl["CLU15"]["CF_Clu_InhibitR"] == 1:
      self.gear_shifter = "reverse"
