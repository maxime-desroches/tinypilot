from cereal import car
from panda import Panda
from common.conversions import Conversions as CV
from common.params import Params
from selfdrive.car.volkswagen.values import CAR, PQ_CARS, CANBUS, NetworkLocation, TransmissionType, GearShifter
from selfdrive.car import STD_CARGO_KG, scale_rot_inertia, scale_tire_stiffness, gen_empty_fingerprint, get_safety_config
from selfdrive.car.interfaces import CarInterfaceBase

ButtonType = car.CarState.ButtonEvent.Type
EventName = car.CarEvent.EventName


class CarInterface(CarInterfaceBase):
  def __init__(self, CP, CarController, CarState):
    super().__init__(CP, CarController, CarState)

    if CP.networkLocation == NetworkLocation.fwdCamera:
      self.ext_bus = CANBUS.pt
      self.cp_ext = self.cp
    else:
      self.ext_bus = CANBUS.cam
      self.cp_ext = self.cp_cam

  @staticmethod
  def get_params(candidate, fingerprint=gen_empty_fingerprint(), car_fw=None, disable_radar=False):
    ret = CarInterfaceBase.get_std_params(candidate, fingerprint)
    ret.carName = "volkswagen"
    ret.radarOffCan = True

    if candidate in PQ_CARS:
      # Set global PQ35/PQ46/NMS parameters
      ret.safetyConfigs = [get_safety_config(car.CarParams.SafetyModel.volkswagenPq)]
      # ret.dashcamOnly = True  # Enable Passat NMS footnote before removing from dashcamOnly
      ret.enableBsm = 0x3BA in fingerprint[0]  # SWA_1

      if 0x440 in fingerprint[0]:  # Getriebe_1
        ret.transmissionType = TransmissionType.automatic
      else:
        ret.transmissionType = TransmissionType.manual

      if any(msg in fingerprint[1] for msg in (0x1A0, 0xC2)):  # Bremse_1, Lenkwinkel_1
        ret.networkLocation = NetworkLocation.gateway
      else:
        ret.networkLocation = NetworkLocation.fwdCamera

      if Params().get_bool("DisableRadar") and ret.networkLocation == NetworkLocation.gateway:
        ret.openpilotLongitudinalControl = True
        ret.safetyConfigs[0].safetyParam |= Panda.FLAG_VOLKSWAGEN_LONG_CONTROL

    else:
      # Set global MQB parameters
      ret.safetyConfigs = [get_safety_config(car.CarParams.SafetyModel.volkswagen)]
      ret.enableBsm = 0x30F in fingerprint[0]  # SWA_01

      if 0xAD in fingerprint[0]:  # Getriebe_11
        ret.transmissionType = TransmissionType.automatic
      elif 0x187 in fingerprint[0]:  # EV_Gearshift
        ret.transmissionType = TransmissionType.direct
      else:
        ret.transmissionType = TransmissionType.manual

      if any(msg in fingerprint[1] for msg in (0x40, 0x86, 0xB2, 0xFD)):  # Airbag_01, LWI_01, ESP_19, ESP_21
        ret.networkLocation = NetworkLocation.gateway
      else:
        ret.networkLocation = NetworkLocation.fwdCamera

    # Global lateral tuning defaults, can be overridden per-vehicle

    ret.steerActuatorDelay = 0.1
    ret.steerLimitTimer = 0.4
    ret.steerRatio = 15.6  # Let the params learner figure this out
    tire_stiffness_factor = 1.0  # Let the params learner figure this out
    ret.lateralTuning.pid.kpBP = [0.]
    ret.lateralTuning.pid.kiBP = [0.]
    ret.lateralTuning.pid.kf = 0.00006
    ret.lateralTuning.pid.kpV = [0.6]
    ret.lateralTuning.pid.kiV = [0.2]

    # Global longitudinal tuning defaults, can be overridden per-vehicle

    ret.pcmCruise = not ret.openpilotLongitudinalControl
    ret.longitudinalActuatorDelayUpperBound = 0.5  # s
    ret.longitudinalTuning.kpV = [0.1]
    ret.longitudinalTuning.kiV = [0.0]

    # Per-chassis tuning values, override tuning defaults here if desired

    if candidate == CAR.ARTEON_MK1:
      ret.mass = 1733 + STD_CARGO_KG
      ret.wheelbase = 2.84

    elif candidate == CAR.ATLAS_MK1:
      ret.mass = 2011 + STD_CARGO_KG
      ret.wheelbase = 2.98

    elif candidate == CAR.GOLF_MK7:
      ret.mass = 1397 + STD_CARGO_KG
      ret.wheelbase = 2.62

    elif candidate == CAR.JETTA_MK7:
      ret.mass = 1328 + STD_CARGO_KG
      ret.wheelbase = 2.71

    elif candidate == CAR.PASSAT_MK8:
      ret.mass = 1551 + STD_CARGO_KG
      ret.wheelbase = 2.79

    elif candidate == CAR.PASSAT_NMS:
      ret.mass = 1503 + STD_CARGO_KG
      ret.wheelbase = 2.80
      ret.minEnableSpeed = 20 * CV.KPH_TO_MS  # ACC "basic", no FtS
      ret.minSteerSpeed = 50 * CV.KPH_TO_MS
      ret.steerActuatorDelay = 0.2
      CarInterfaceBase.configure_torque_tune(candidate, ret.lateralTuning)

    elif candidate == CAR.POLO_MK6:
      ret.mass = 1230 + STD_CARGO_KG
      ret.wheelbase = 2.55

    elif candidate == CAR.TAOS_MK1:
      ret.mass = 1498 + STD_CARGO_KG
      ret.wheelbase = 2.69

    elif candidate == CAR.TCROSS_MK1:
      ret.mass = 1150 + STD_CARGO_KG
      ret.wheelbase = 2.60

    elif candidate == CAR.TIGUAN_MK2:
      ret.mass = 1715 + STD_CARGO_KG
      ret.wheelbase = 2.74

    elif candidate == CAR.TOURAN_MK2:
      ret.mass = 1516 + STD_CARGO_KG
      ret.wheelbase = 2.79

    elif candidate == CAR.TRANSPORTER_T61:
      ret.mass = 1926 + STD_CARGO_KG
      ret.wheelbase = 3.00  # SWB, LWB is 3.40, TBD how to detect difference
      ret.minSteerSpeed = 14.0

    elif candidate == CAR.TROC_MK1:
      ret.mass = 1413 + STD_CARGO_KG
      ret.wheelbase = 2.63

    elif candidate == CAR.AUDI_A3_MK3:
      ret.mass = 1335 + STD_CARGO_KG
      ret.wheelbase = 2.61

    elif candidate == CAR.AUDI_Q2_MK1:
      ret.mass = 1205 + STD_CARGO_KG
      ret.wheelbase = 2.61

    elif candidate == CAR.AUDI_Q3_MK2:
      ret.mass = 1623 + STD_CARGO_KG
      ret.wheelbase = 2.68

    elif candidate == CAR.SEAT_ATECA_MK1:
      ret.mass = 1900 + STD_CARGO_KG
      ret.wheelbase = 2.64

    elif candidate == CAR.SEAT_LEON_MK3:
      ret.mass = 1227 + STD_CARGO_KG
      ret.wheelbase = 2.64

    elif candidate == CAR.SKODA_KAMIQ_MK1:
      ret.mass = 1265 + STD_CARGO_KG
      ret.wheelbase = 2.66

    elif candidate == CAR.SKODA_KAROQ_MK1:
      ret.mass = 1278 + STD_CARGO_KG
      ret.wheelbase = 2.66

    elif candidate == CAR.SKODA_KODIAQ_MK1:
      ret.mass = 1569 + STD_CARGO_KG
      ret.wheelbase = 2.79

    elif candidate == CAR.SKODA_OCTAVIA_MK3:
      ret.mass = 1388 + STD_CARGO_KG
      ret.wheelbase = 2.68

    elif candidate == CAR.SKODA_SCALA_MK1:
      ret.mass = 1192 + STD_CARGO_KG
      ret.wheelbase = 2.65

    elif candidate == CAR.SKODA_SUPERB_MK3:
      ret.mass = 1505 + STD_CARGO_KG
      ret.wheelbase = 2.84

    else:
      raise ValueError(f"unsupported car {candidate}")

    ret.rotationalInertia = scale_rot_inertia(ret.mass, ret.wheelbase)
    ret.centerToFront = ret.wheelbase * 0.45
    ret.tireStiffnessFront, ret.tireStiffnessRear = scale_tire_stiffness(ret.mass, ret.wheelbase, ret.centerToFront,
                                                                         tire_stiffness_factor=tire_stiffness_factor)
    return ret

  # returns a car.CarState
  def _update(self, c):
    ret = self.CS.update(self.cp, self.cp_cam, self.cp_ext, self.CP.transmissionType)

    events = self.create_common_events(ret, extra_gears=[GearShifter.eco, GearShifter.sport, GearShifter.manumatic],
                                       pcm_enable=not self.CS.CP.openpilotLongitudinalControl)

    # Low speed steer alert hysteresis logic
    if self.CP.minSteerSpeed > 0. and ret.vEgo < (self.CP.minSteerSpeed + 1.):
      self.low_speed_alert = True
    elif ret.vEgo > (self.CP.minSteerSpeed + 2.):
      self.low_speed_alert = False
    if self.low_speed_alert:
      events.add(EventName.belowSteerSpeed)

    if self.CS.CP.openpilotLongitudinalControl:
      if ret.vEgo < self.CP.minEnableSpeed + 2.:
        events.add(EventName.belowEngageSpeed)
      if c.enabled and ret.vEgo < self.CP.minEnableSpeed:
        events.add(EventName.speedTooLow)

      for b in ret.buttonEvents:
        # Enable on falling edge of both set and resume
        if b.type in (ButtonType.setCruise, ButtonType.resumeCruise) and not b.pressed:
          events.add(EventName.buttonEnable)
        # Disable on rising edge of cancel
        if b.type == ButtonType.cancel and b.pressed:
          events.add(EventName.buttonCancel)

    ret.events = events.to_msg()

    return ret

  def apply(self, c):
    return self.CC.update(c, self.CS, self.ext_bus)
