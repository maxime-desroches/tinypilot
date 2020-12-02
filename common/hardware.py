import os
import random
from typing import cast

from cereal import log
from common.hardware_android import Android
from common.hardware_tici import Tici
from common.hardware_base import HardwareBase

EON = os.path.isfile('/EON')
TICI = os.path.isfile('/TICI')
PC = not (EON or TICI)
ANDROID = EON


NetworkType = log.ThermalData.NetworkType
NetworkStrength = log.ThermalData.NetworkStrength


class Pc(HardwareBase):
  def get_sound_card_online(self):
    return True

  def get_imei(self, slot):
    return "%015d" % random.randint(0, 1 << 32)

  def get_serial(self):
    return "cccccccc"

  def get_subscriber_info(self):
    return ""

  def reboot(self, reason=None):
    print("REBOOT!")

  def get_network_type(self):
    return NetworkType.wifi

  def get_sim_info(self):
    return {
      'sim_id': '',
      'mcc_mnc': None,
      'network_type': ["Unknown"],
      'sim_state': ["ABSENT"],
      'data_connected': False
    }

  def get_network_strength(self, network_type):
    return NetworkStrength.unknown

  def get_battery_capacity(self):
    return 100

  def get_battery_status(self):
    return ""

  def get_battery_current(self):
    return 0

  def get_battery_voltage(self):
    return 0

  def get_battery_charging(self):
    return True

  def set_battery_charging(self, on):
    pass

  def get_usb_present(self):
    return False

  def get_current_power_draw(self):
    return 0


if EON:
  HARDWARE = cast(HardwareBase, Android())
elif TICI:
  HARDWARE = cast(HardwareBase, Tici())
else:
  HARDWARE = cast(HardwareBase, Pc())
