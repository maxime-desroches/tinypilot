import numpy as np
from cereal import car
from selfdrive.config import Conversions as CV
from selfdrive.car.interfaces import CarStateBase
from opendbc.can.parser import CANParser
from opendbc.can.can_define import CANDefine
from selfdrive.car.volkswagen.values import DBC, CANBUS, NWL, TRANS, GEAR, BUTTON_STATES, CarControllerParams

class CarState(CarStateBase):
  def __init__(self, CP):
    super().__init__(CP)
    can_define = CANDefine(DBC[CP.carFingerprint]['pt'])
    if CP.transmissionType == TRANS.automatic:
      self.shifter_values = can_define.dv["Getriebe_11"]['GE_Fahrstufe']
    elif CP.transmissionType == TRANS.direct:
      self.shifter_values = can_define.dv["EV_Gearshift"]['GearPosition']
    self.buttonStates = BUTTON_STATES.copy()

  def update(self, pt_cp, cam_cp, acc_cp, trans_type):
    ret = car.CarState.new_message()
    # Update vehicle speed and acceleration from ABS wheel speeds.
    ret.wheelSpeeds.fl = pt_cp.vl["ESP_19"]['ESP_VL_Radgeschw_02'] * CV.KPH_TO_MS
    ret.wheelSpeeds.fr = pt_cp.vl["ESP_19"]['ESP_VR_Radgeschw_02'] * CV.KPH_TO_MS
    ret.wheelSpeeds.rl = pt_cp.vl["ESP_19"]['ESP_HL_Radgeschw_02'] * CV.KPH_TO_MS
    ret.wheelSpeeds.rr = pt_cp.vl["ESP_19"]['ESP_HR_Radgeschw_02'] * CV.KPH_TO_MS

    ret.vEgoRaw = float(np.mean([ret.wheelSpeeds.fl, ret.wheelSpeeds.fr, ret.wheelSpeeds.rl, ret.wheelSpeeds.rr]))
    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)

    ret.standstill = ret.vEgoRaw < 0.1

    # Update steering angle, rate, yaw rate, and driver input torque. VW send
    # the sign/direction in a separate signal so they must be recombined.
    ret.steeringAngle = pt_cp.vl["LWI_01"]['LWI_Lenkradwinkel'] * (1, -1)[int(pt_cp.vl["LWI_01"]['LWI_VZ_Lenkradwinkel'])]
    ret.steeringRate = pt_cp.vl["LWI_01"]['LWI_Lenkradw_Geschw'] * (1, -1)[int(pt_cp.vl["LWI_01"]['LWI_VZ_Lenkradw_Geschw'])]
    ret.steeringTorque = pt_cp.vl["LH_EPS_03"]['EPS_Lenkmoment'] * (1, -1)[int(pt_cp.vl["LH_EPS_03"]['EPS_VZ_Lenkmoment'])] * 100.0
    ret.steeringPressed = abs(ret.steeringTorque) > CarControllerParams.STEER_DRIVER_ALLOWANCE
    ret.yawRate = pt_cp.vl["ESP_02"]['ESP_Gierrate'] * (1, -1)[int(pt_cp.vl["ESP_02"]['ESP_VZ_Gierrate'])] * CV.DEG_TO_RAD

    # Update gas, brakes, and gearshift.
    ret.gas = pt_cp.vl["Motor_20"]['MO_Fahrpedalrohwert_01'] / 100.0
    ret.gasPressed = ret.gas > 0
    ret.brake = pt_cp.vl["ESP_05"]['ESP_Bremsdruck'] / 250.0  # FIXME: this is pressure in Bar, not sure what OP expects
    ret.brakePressed = bool(pt_cp.vl["ESP_05"]['ESP_Fahrer_bremst'])
    ret.brakeLights = bool(pt_cp.vl["ESP_05"]['ESP_Status_Bremsdruck'])

    # Additional safety checks performed in CarInterface.
    self.parkingBrakeSet = bool(pt_cp.vl["Kombi_01"]['KBI_Handbremse'])  # FIXME: need to include an EPB check as well
    ret.espDisabled = pt_cp.vl["ESP_21"]['ESP_Tastung_passiv'] != 0

    # Update gear and/or clutch position data.
    if trans_type == TRANS.automatic:
      ret.gearShifter = self.parse_gear_shifter(self.shifter_values.get(pt_cp.vl["Getriebe_11"]['GE_Fahrstufe'], None))
    elif trans_type == TRANS.direct:
      ret.gearShifter = self.parse_gear_shifter(self.shifter_values.get(pt_cp.vl["EV_Gearshift"]['GearPosition'], None))
    elif trans_type == TRANS.manual:
      ret.clutchPressed = not pt_cp.vl["Motor_14"]['MO_Kuppl_schalter']
      reverse_light = bool(pt_cp.vl["Gateway_72"]['BCM1_Rueckfahrlicht_Schalter'])
      # TODO: consider gating an OP minimum engage speed on whether the clutch is pressed, to prevent stalling
      if reverse_light:
        ret.gearShifter = GEAR.reverse
      else:
        ret.gearShifter = GEAR.drive

    # Update door and trunk/hatch lid open status.
    ret.doorOpen = any([pt_cp.vl["Gateway_72"]['ZV_FT_offen'],
                        pt_cp.vl["Gateway_72"]['ZV_BT_offen'],
                        pt_cp.vl["Gateway_72"]['ZV_HFS_offen'],
                        pt_cp.vl["Gateway_72"]['ZV_HBFS_offen'],
                        pt_cp.vl["Gateway_72"]['ZV_HD_offen']])

    # Update seatbelt fastened status.
    ret.seatbeltUnlatched = pt_cp.vl["Airbag_02"]["AB_Gurtschloss_FA"] != 3

    # Update driver preference for metric. VW stores many different unit
    # preferences, including separate units for for distance vs. speed.
    # We use the speed preference for OP.
    self.displayMetricUnits = not pt_cp.vl["Einheiten_01"]["KBI_MFA_v_Einheit_02"]

    # Stock FCW is considered active if a warning is displayed to the driver
    # or the release bit for brake-jerk warning is set. Stock AEB considered
    # active if the partial braking or target braking release bits are set.
    # Ref: VW SSP 890253 "Volkswagen Driver Assistance Systems V2", "Front
    # Assist with Braking: Golf Family" (applies to all MQB)
    ret.stockFcw = any([bool(acc_cp.vl["ACC_10"]["AWV2_Priowarnung"]),
                        bool(acc_cp.vl["ACC_10"]["AWV2_Freigabe"])])
    ret.stockAeb = any([bool(acc_cp.vl["ACC_10"]["ANB_Teilbremsung_Freigabe"]),
                        bool(acc_cp.vl["ACC_10"]["ANB_Zielbremsung_Freigabe"])])

    # Consume blind-spot radar info/warning LED states, if available
    ret.leftBlindspot = any([bool(acc_cp.vl["SWA_01"]["SWA_Infostufe_SWA_li"]),
                             bool(acc_cp.vl["SWA_01"]["SWA_Warnung_SWA_li"])])
    ret.rightBlindspot = any([bool(acc_cp.vl["SWA_01"]["SWA_Infostufe_SWA_re"]),
                             bool(acc_cp.vl["SWA_01"]["SWA_Warnung_SWA_re"])])

    # Consume SWA (Lane Change Assist) relevant info from factory LDW message
    # to pass along to the blind spot radar controller
    self.ldw_lane_warning_left = bool(cam_cp.vl["LDW_02"]["LDW_SW_Warnung_links"])
    self.ldw_lane_warning_right = bool(cam_cp.vl["LDW_02"]["LDW_SW_Warnung_rechts"])
    self.ldw_side_dlc_tlc = bool(cam_cp.vl["LDW_02"]["LDW_Seite_DLCTLC"])
    self.ldw_dlc = cam_cp.vl["LDW_02"]["LDW_DLC"]
    self.ldw_tlc = cam_cp.vl["LDW_02"]["LDW_TLC"]

    # Update ACC radar status.
    accStatus = pt_cp.vl["TSK_06"]['TSK_Status']
    if accStatus == 2:
      # ACC okay and enabled, but not currently engaged
      ret.cruiseState.available = True
      ret.cruiseState.enabled = False
    elif accStatus in [3, 4, 5]:
      # ACC okay and enabled, currently engaged and regulating speed (3) or engaged with driver accelerating (4) or overrun (5)
      ret.cruiseState.available = True
      ret.cruiseState.enabled = True
    else:
      # ACC okay but disabled (1), or a radar visibility or other fault/disruption (6 or 7)
      ret.cruiseState.available = False
      ret.cruiseState.enabled = False

    # Update ACC setpoint. When the setpoint is zero or there's an error, the
    # radar sends a set-speed of ~90.69 m/s / 203mph.
    ret.cruiseState.speed = acc_cp.vl["ACC_02"]["ACC_Wunschgeschw"] * CV.KPH_TO_MS
    if ret.cruiseState.speed > 90:
      ret.cruiseState.speed = 0

    # Update control button states for turn signals and ACC controls.
    self.buttonStates["accelCruise"] = bool(pt_cp.vl["GRA_ACC_01"]['GRA_Tip_Hoch'])
    self.buttonStates["decelCruise"] = bool(pt_cp.vl["GRA_ACC_01"]['GRA_Tip_Runter'])
    self.buttonStates["cancel"] = bool(pt_cp.vl["GRA_ACC_01"]['GRA_Abbrechen'])
    self.buttonStates["setCruise"] = bool(pt_cp.vl["GRA_ACC_01"]['GRA_Tip_Setzen'])
    self.buttonStates["resumeCruise"] = bool(pt_cp.vl["GRA_ACC_01"]['GRA_Tip_Wiederaufnahme'])
    self.buttonStates["gapAdjustCruise"] = bool(pt_cp.vl["GRA_ACC_01"]['GRA_Verstellung_Zeitluecke'])
    ret.leftBlinker = bool(pt_cp.vl["Gateway_72"]['BH_Blinker_li'])
    ret.rightBlinker = bool(pt_cp.vl["Gateway_72"]['BH_Blinker_re'])

    # Read ACC hardware button type configuration info that has to pass thru
    # to the radar. Ends up being different for steering wheel buttons vs
    # third stalk type controls.
    self.graHauptschalter = pt_cp.vl["GRA_ACC_01"]['GRA_Hauptschalter']
    self.graTypHauptschalter = pt_cp.vl["GRA_ACC_01"]['GRA_Typ_Hauptschalter']
    self.graButtonTypeInfo = pt_cp.vl["GRA_ACC_01"]['GRA_ButtonTypeInfo']
    self.graTipStufe2 = pt_cp.vl["GRA_ACC_01"]['GRA_Tip_Stufe_2']
    self.graTyp468 = pt_cp.vl["GRA_ACC_01"]['GRA_Typ468']
    # Pick up the GRA_ACC_01 CAN message counter so we can sync to it for
    # later cruise-control button spamming.
    self.graMsgBusCounter = pt_cp.vl["GRA_ACC_01"]['COUNTER']

    # Check to make sure the electric power steering rack is configured to
    # accept and respond to HCA_01 messages and has not encountered a fault.
    self.steeringFault = not pt_cp.vl["LH_EPS_03"]["EPS_HCA_Status"]

    return ret

  @staticmethod
  def get_can_parser(CP):
    # this function generates lists for signal, messages and initial values
    signals = [
      # sig_name, sig_address, default
      ("LWI_Lenkradwinkel", "LWI_01", 0),           # Absolute steering angle
      ("LWI_VZ_Lenkradwinkel", "LWI_01", 0),        # Steering angle sign
      ("LWI_Lenkradw_Geschw", "LWI_01", 0),         # Absolute steering rate
      ("LWI_VZ_Lenkradw_Geschw", "LWI_01", 0),      # Steering rate sign
      ("ESP_VL_Radgeschw_02", "ESP_19", 0),         # ABS wheel speed, front left
      ("ESP_VR_Radgeschw_02", "ESP_19", 0),         # ABS wheel speed, front right
      ("ESP_HL_Radgeschw_02", "ESP_19", 0),         # ABS wheel speed, rear left
      ("ESP_HR_Radgeschw_02", "ESP_19", 0),         # ABS wheel speed, rear right
      ("ESP_Gierrate", "ESP_02", 0),                # Absolute yaw rate
      ("ESP_VZ_Gierrate", "ESP_02", 0),             # Yaw rate sign
      ("ZV_FT_offen", "Gateway_72", 0),             # Door open, driver
      ("ZV_BT_offen", "Gateway_72", 0),             # Door open, passenger
      ("ZV_HFS_offen", "Gateway_72", 0),            # Door open, rear left
      ("ZV_HBFS_offen", "Gateway_72", 0),           # Door open, rear right
      ("ZV_HD_offen", "Gateway_72", 0),             # Trunk or hatch open
      ("BH_Blinker_li", "Gateway_72", 0),           # Left turn signal on
      ("BH_Blinker_re", "Gateway_72", 0),           # Right turn signal on
      ("AB_Gurtschloss_FA", "Airbag_02", 0),        # Seatbelt status, driver
      ("AB_Gurtschloss_BF", "Airbag_02", 0),        # Seatbelt status, passenger
      ("ESP_Fahrer_bremst", "ESP_05", 0),           # Brake pedal pressed
      ("ESP_Status_Bremsdruck", "ESP_05", 0),       # Brakes applied
      ("ESP_Bremsdruck", "ESP_05", 0),              # Brake pressure applied
      ("MO_Fahrpedalrohwert_01", "Motor_20", 0),    # Accelerator pedal value
      ("EPS_Lenkmoment", "LH_EPS_03", 0),           # Absolute driver torque input
      ("EPS_VZ_Lenkmoment", "LH_EPS_03", 0),        # Driver torque input sign
      ("EPS_HCA_Status", "LH_EPS_03", 0),           # Steering rack ready to process HCA commands
      ("ESP_Tastung_passiv", "ESP_21", 0),          # Stability control disabled
      ("KBI_MFA_v_Einheit_02", "Einheiten_01", 0),  # MPH vs KMH speed display
      ("KBI_Handbremse", "Kombi_01", 0),            # Manual handbrake applied
      ("TSK_Status", "TSK_06", 0),                  # ACC engagement status from drivetrain coordinator
      ("GRA_Hauptschalter", "GRA_ACC_01", 0),       # ACC button, on/off
      ("GRA_Abbrechen", "GRA_ACC_01", 0),           # ACC button, cancel
      ("GRA_Tip_Setzen", "GRA_ACC_01", 0),          # ACC button, set
      ("GRA_Tip_Hoch", "GRA_ACC_01", 0),            # ACC button, increase or accel
      ("GRA_Tip_Runter", "GRA_ACC_01", 0),          # ACC button, decrease or decel
      ("GRA_Tip_Wiederaufnahme", "GRA_ACC_01", 0),  # ACC button, resume
      ("GRA_Verstellung_Zeitluecke", "GRA_ACC_01", 0),  # ACC button, time gap adj
      ("GRA_Typ_Hauptschalter", "GRA_ACC_01", 0),   # ACC main button type
      ("GRA_Tip_Stufe_2", "GRA_ACC_01", 0),         # unknown related to stalk type
      ("GRA_Typ468", "GRA_ACC_01", 0),              # Set/Resume button behavior as overloaded coast/accel??
      ("GRA_ButtonTypeInfo", "GRA_ACC_01", 0),      # unknown related to stalk type
      ("COUNTER", "GRA_ACC_01", 0),                 # GRA_ACC_01 CAN message counter
    ]

    checks = [
      # sig_address, frequency
      ("LWI_01", 100),            # From J500 Steering Assist with integrated sensors
      ("LH_EPS_03", 100),         # From J500 Steering Assist with integrated sensors
      ("ESP_19", 100),            # From J104 ABS/ESP controller
      ("ESP_05", 50),             # From J104 ABS/ESP controller
      ("ESP_21", 50),             # From J104 ABS/ESP controller
      ("Motor_20", 50),           # From J623 Engine control module
      ("TSK_06", 50),             # From J623 Engine control module
      ("GRA_ACC_01", 33),         # From J??? steering wheel control buttons
      ("Gateway_72", 10),         # From J533 CAN gateway (aggregated data)
      ("Airbag_02", 5),           # From J234 Airbag control module
      ("Kombi_01", 2),            # From J285 Instrument cluster
      ("Einheiten_01", 1),        # From J??? not known if gateway, cluster, or BCM
    ]

    if CP.transmissionType == TRANS.automatic:
      signals += [("GE_Fahrstufe", "Getriebe_11", 0)]  # Auto trans gear selector position
      checks += [("Getriebe_11", 20)]  # From J743 Auto transmission control module
    elif CP.transmissionType == TRANS.direct:
      signals += [("GearPosition", "EV_Gearshift", 0)]  # EV gear selector position
      checks += [("EV_Gearshift", 10)]  # From J??? unknown EV control module
    elif CP.transmissionType == TRANS.manual:
      signals += [("MO_Kuppl_schalter", "Motor_14", 0),  # Clutch switch
                  ("BCM1_Rueckfahrlicht_Schalter", "Gateway_72", 0)]  # Reverse light from BCM
      checks += [("Motor_14", 10)]  # From J623 Engine control module

    if CP.networkLocation == NWL.fwdCamera:
      # Extended CAN devices other than the camera are here on CANBUS.pt
      # FIXME: gate SWA_01 checks on module being detected, and reduce duplicate network location code
      signals += [("AWV2_Priowarnung", "ACC_10", 0),      # FCW related
                  ("AWV2_Freigabe", "ACC_10", 0),         # FCW related
                  ("ANB_Teilbremsung_Freigabe", "ACC_10", 0),  # AEB related
                  ("ANB_Zielbremsung_Freigabe", "ACC_10", 0),  # AEB related
                  ("SWA_Infostufe_SWA_li", "SWA_01", 0),  # Blindspot object info, left
                  ("SWA_Warnung_SWA_li", "SWA_01", 0),    # Blindspot object warning, left
                  ("SWA_Infostufe_SWA_re", "SWA_01", 0),  # Blindspot object info, right
                  ("SWA_Warnung_SWA_re", "SWA_01", 0),    # Blindspot object warning, right
                  ("ACC_Wunschgeschw", "ACC_02", 0)]      # ACC set speed
      checks += [("ACC_10", 50),  # From J428 ACC radar control module
                 # FIXME: SWA_01 should be checked when we have better detection of installed hardware
                 #("SWA_01", 20),  # From J1086 Lane Change Assist module
                 ("ACC_02", 17)]  # From J428 ACC radar control module

    return CANParser(DBC[CP.carFingerprint]['pt'], signals, checks, CANBUS.pt)

  @staticmethod
  def get_cam_can_parser(CP):

    # FIXME: gate LDW_02 checks on module being detected
    signals = [
      # sig_name, sig_address, default
      ("LDW_SW_Warnung_links", "LDW_02", 0),    # Blind spot in warning mode on left side due to lane departure
      ("LDW_SW_Warnung_rechts", "LDW_02", 0),   # Blind spot in warning mode on right side due to lane departure
      ("LDW_Seite_DLCTLC", "LDW_02", 0),        # Direction of most likely lane departure (left or right)
      ("LDW_DLC", "LDW_02", 0),                 # Lane departure, distance to line crossing
      ("LDW_TLC", "LDW_02", 0),                 # Lane departure, time to line crossing
    ]

    checks = [
      # sig_address, frequency
      # FIXME: LDW_02 should be checked when we have better detection of installed hardware
      #("LDW_02", 10),  # From R242 Driver assistance camera
    ]

    if CP.networkLocation == NWL.gateway:
      # All Extended CAN devices are here on CANBUS.cam
      # FIXME: gate SWA_01 checks on module being detected, and reduce duplicate network location code
      signals += [("AWV2_Priowarnung", "ACC_10", 0),      # FCW related
                  ("AWV2_Freigabe", "ACC_10", 0),         # FCW related
                  ("ANB_Teilbremsung_Freigabe", "ACC_10", 0),  # AEB related
                  ("ANB_Zielbremsung_Freigabe", "ACC_10", 0),  # AEB related
                  ("SWA_Infostufe_SWA_li", "SWA_01", 0),  # Blindspot object info, left
                  ("SWA_Warnung_SWA_li", "SWA_01", 0),    # Blindspot object warning, left
                  ("SWA_Infostufe_SWA_re", "SWA_01", 0),  # Blindspot object info, right
                  ("SWA_Warnung_SWA_re", "SWA_01", 0),    # Blindspot object warning, right
                  ("ACC_Wunschgeschw", "ACC_02", 0)]              # ACC set speed
      checks += [("ACC_10", 50),  # From J428 ACC radar control module
                 # FIXME: SWA_01 should be checked when we have better detection of installed hardware
                 #("SWA_01", 20),  # From J1086 Lane Change Assist module
                 ("ACC_02", 17)]  # From J428 ACC radar control module

    return CANParser(DBC[CP.carFingerprint]['pt'], signals, checks, CANBUS.cam)
