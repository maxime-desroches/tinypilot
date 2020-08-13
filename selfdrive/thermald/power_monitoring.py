import datetime
import random
import threading
import time
from statistics import mean

from cereal import log
from common.realtime import sec_since_boot
from selfdrive.swaglog import cloudlog

PANDA_OUTPUT_VOLTAGE = 5.28
CAR_VOLTAGE_LOW_PASS_K = 0.091 # LPF gain for 5s tau (dt/tau / (dt/tau + 1))

# A C2 uses about 1W while idling, and 30h seens like a good shutoff for most cars
# While driving, a battery charges completely in about 30-60 minutes
CAR_BATTERY_CAPACITY_uWh = 30e6
CAR_CHARGING_RATE_W = 45

VBATT_START_CHARGING = 11.5
VBATT_PAUSE_CHARGING = 11.0
MAX_TIME_OFFROAD_S = 30*3600

# Parameters
def get_battery_capacity():
  return _read_param("/sys/class/power_supply/battery/capacity", int)


def get_battery_status():
  # This does not correspond with actual charging or not.
  # If a USB cable is plugged in, it responds with 'Charging', even when charging is disabled
  return _read_param("/sys/class/power_supply/battery/status", lambda x: x.strip(), '')


def get_battery_current():
  return _read_param("/sys/class/power_supply/battery/current_now", int)


def get_battery_voltage():
  return _read_param("/sys/class/power_supply/battery/voltage_now", int)


def get_usb_present():
  return _read_param("/sys/class/power_supply/usb/present", lambda x: bool(int(x)), False)


def get_battery_charging():
  # This does correspond with actually charging
  return _read_param("/sys/class/power_supply/battery/charge_type", lambda x: x.strip() != "N/A", True)


def set_battery_charging(on):
  with open('/sys/class/power_supply/battery/charging_enabled', 'w') as f:
    f.write(f"{1 if on else 0}\n")


# Helpers
def _read_param(path, parser, default=0):
  try:
    with open(path) as f:
      return parser(f.read())
  except Exception:
    return default


def panda_current_to_actual_current(panda_current):
  # From white/grey panda schematic
  return (3.3 - (panda_current * 3.3 / 4096)) / 8.25


class PowerMonitoring:
  def __init__(self, params):
    self.params = params
    self.last_measurement_time = None           # Used for integration delta
    self.power_used_uWh = 0                     # Integrated power usage in uWh since going into offroad
    self.next_pulsed_measurement_time = None
    self.car_voltage_mV = 12e3                  # Low-passed version of health voltage
    self.integration_lock = threading.Lock()

    car_battery_power_uWh = params.get("CarBatteryPower")
    if car_battery_power_uWh == None:
      # If unknown, we assume the car battery is almost dead
      self.car_battery_power_uWh = (CAR_BATTERY_CAPACITY_uWh / 10)
    else:
      self.car_battery_power_uWh = int(car_battery_power_uWh)
    

  # Calculation tick
  def calculate(self, health):
    try:
      now = sec_since_boot()

      # Check that time is valid
      if datetime.datetime.fromtimestamp(now).year < 2019:
        return

      # If health is None, we're probably not in a car, so we don't care
      if health is None or health.health.hwType == log.HealthData.HwType.unknown:
        with self.integration_lock:
          self.last_measurement_time = None
          self.next_pulsed_measurement_time = None
          self.power_used_uWh = 0
        return

      # Low-pass battery voltage
      self.car_voltage_mV = ((health.health.voltage * CAR_VOLTAGE_LOW_PASS_K) + (self.car_voltage_mV * (1 -  CAR_VOLTAGE_LOW_PASS_K)))

      # Cap the car battery power and save it in a param
      self.car_battery_power_uWh = max(self.car_battery_power_uWh, 0)
      self.car_battery_power_uWh = min(self.car_battery_power_uWh, CAR_BATTERY_CAPACITY_uWh)
      self.params.put("CarBatteryPower", int(self.car_battery_power_uWh))

      # First measurement, set integration time
      with self.integration_lock:
        if self.last_measurement_time is None:
          self.last_measurement_time = now
          return

      if (health.health.ignitionLine or health.health.ignitionCan):
        # If there is ignition, we integrate the charging rate of the car
        with self.integration_lock:
          self.power_used_uWh = 0
          integration_time_h = (now - self.last_measurement_time) / 3600
          if integration_time_h < 0:
            raise ValueError(f"Negative integration time: {integration_time_h}h")
          self.car_battery_power_uWh += (CAR_CHARGING_RATE_W * 1e6 * integration_time_h)
          self.last_measurement_time = now
      else:
        # No ignition, we integrate the offroad power used by the device
        is_uno = health.health.hwType == log.HealthData.HwType.uno
        # Get current power draw somehow
        current_power = 0
        if get_battery_status() == 'Discharging':
          # If the battery is discharging, we can use this measurement
          # On C2: this is low by about 10-15%, probably mostly due to UNO draw not being factored in
          current_power = ((get_battery_voltage() / 1000000) * (get_battery_current() / 1000000))
        elif (health.health.hwType in [log.HealthData.HwType.whitePanda, log.HealthData.HwType.greyPanda]) and (health.health.current > 1):
          # If white/grey panda, use the integrated current measurements if the measurement is not 0
          # If the measurement is 0, the current is 400mA or greater, and out of the measurement range of the panda
          # This seems to be accurate to about 5%
          current_power = (PANDA_OUTPUT_VOLTAGE * panda_current_to_actual_current(health.health.current))
        elif (self.next_pulsed_measurement_time is not None) and (self.next_pulsed_measurement_time <= now):
          # TODO: Figure out why this is off by a factor of 3/4???
          FUDGE_FACTOR = 1.33

          # Turn off charging for about 10 sec in a thread that does not get killed on SIGINT, and perform measurement here to avoid blocking thermal
          def perform_pulse_measurement(now):
            try:
              set_battery_charging(False)
              time.sleep(5)

              # Measure for a few sec to get a good average
              voltages = []
              currents = []
              for _ in range(6):
                voltages.append(get_battery_voltage())
                currents.append(get_battery_current())
                time.sleep(1)
              current_power = ((mean(voltages) / 1000000) * (mean(currents) / 1000000))

              self._perform_integration(now, current_power * FUDGE_FACTOR)

              # Enable charging again
              set_battery_charging(True)
            except Exception:
              cloudlog.exception("Pulsed power measurement failed")

          # Start pulsed measurement and return
          threading.Thread(target=perform_pulse_measurement, args=(now,)).start()
          self.next_pulsed_measurement_time = None
          return

        elif self.next_pulsed_measurement_time is None and not is_uno:
          # On a charging EON with black panda, or drawing more than 400mA out of a white/grey one
          # Only way to get the power draw is to turn off charging for a few sec and check what the discharging rate is
          # We shouldn't do this very often, so make sure it has been some long-ish random time interval
          self.next_pulsed_measurement_time = now + random.randint(120, 180)
          return
        else:
          # Do nothing
          return

        # Do the integration
        self._perform_integration(now, current_power)
    except Exception:
      cloudlog.exception("Power monitoring calculation failed")

  def _perform_integration(self, t, current_power):
    with self.integration_lock:
      try:
        if self.last_measurement_time:
          integration_time_h = (t - self.last_measurement_time) / 3600
          power_used = (current_power * 1000000) * integration_time_h
          if power_used < 0:
            raise ValueError(f"Negative power used! Integration time: {integration_time_h} h Current Power: {power_used} uWh")
          self.power_used_uWh += power_used
          self.car_battery_power_uWh -= power_used
          self.last_measurement_time = t
      except Exception:
        cloudlog.exception("Integration failed")

  # Get the power usage
  def get_power_used(self):
    return int(self.power_used_uWh)

  # See if we need to disable charging
  def should_disable_charging(self, health, offroad_timestamp):
    if health == None or offroad_timestamp == None:
      return False

    now = sec_since_boot()
    disable_charging = False
    disable_charging |= (now - offroad_timestamp) > MAX_TIME_OFFROAD_S
    disable_charging |= (self.car_voltage_mV < (VBATT_PAUSE_CHARGING * 1e3))
    disable_charging |= (self.car_battery_power_uWh <= 0)
    disable_charging &= (self.car_voltage_mV < (VBATT_START_CHARGING * 1e3))
    disable_charging &= (not health.health.ignitionLine and not health.health.ignitionCan)
    disable_charging &= (self.params.get("DisablePowerDown") != b"1")
    return disable_charging

  # See if we need to shutdown
  def should_shutdown(self, health, offroad_timestamp, started_seen, LEON):
    if health == None or offroad_timestamp == None:
      return False

    panda_charging = (health.health.usbPowerMode != log.HealthData.UsbPowerMode.client)
    BATT_PERC_OFF = 10 if LEON else 3

    should_shutdown = False
    # Wait until we have shut down charging before powering down
    should_shutdown |= (not panda_charging and self.should_disable_charging(health, offroad_timestamp))
    should_shutdown |= ((self.get_battery_capacity() < BATT_PERC_OFF) and (not self.get_battery_charging()) and started_seen and ((now - offroad_timestamp) > 60))

    return should_shutdown

