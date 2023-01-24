import time
from datetime import datetime
from collections import defaultdict
import cereal.messaging as messaging

MAX_SATS = 150

def print_stats(start_time, ephem_stats, sat_meas, meas_cnt, locktime_per_sat):
  print(f"----- {datetime.now().strftime('%H:%M:%S.%f')}, Runtime: {datetime.now() - start_time}")
  for sat in range(len(ephem_stats)):
    if ephem_stats[sat] != 0:
      print(f"{sat}: {round(ephem_stats[sat]/meas_cnt, 2)} ephem/min")

  print("Measurements (in last minute):")
  m6 = []
  for sat in sat_meas:
    if sat_meas[sat] in [60, 61, 600, 601]:
      m6.append(sat)
    else:
      print(f"{sat}: {sat_meas[sat]}")
  print(f"60club msgs sats: {m6}")

  print("Locktime per sat (in last minute):")
  for sat in locktime_per_sat:
    print(f"  {sat}: {locktime_per_sat[sat]}")


def print_data(ephem_data):
  print(f"----- {datetime.now().strftime('%H:%M:%S.%f')}")
  for sat in ephem_data:
    print(f"{sat}: {ephem_data[sat]}")


def main():
  ephem_stats = [0]*MAX_SATS
  sat_meas = defaultdict(int)
  meas_cnt = 0

  last_log = time.monotonic()
  last_log_10s = time.monotonic()
  start_time = datetime.now()

  ephem_data = defaultdict(int)
  total_ephems_per_min = []
  diff_ephems_per_min = []

  locktime_per_sat = {}

  sm = messaging.SubMaster(["ubloxGnss", "ubloxRaw"])
  while True:
    sm.update()

    #if sm.updated["ubloxRaw"]:
    #  continue

    if not sm.updated["ubloxGnss"]:
      continue

    # check ubluxGnss message
    msg = sm['ubloxGnss']

    if msg.which() == "measurementReport":
      for m in msg.measurementReport.measurements:
        if m.gnssId == 6 and m.svId == 255:
          continue

        svId = m.svId + 100 if m.gnssId == 6 else m.svId
        sat_meas[svId] += 1
        locktime_per_sat[svId] = m.locktime

    if msg.which() == "ephemeris":
      ephem_data[msg.ephemeris.svId] += 1

    if (time.monotonic() - last_log_10s) > 10:
      print_data(ephem_data)
      last_log_10s = time.monotonic()

    if (time.monotonic() - last_log) > 60:
      for sat in ephem_data:
        ephem_stats[sat] += ephem_data[sat]
      diff_ephems_per_min.append(len(ephem_data))
      total_ephems_per_min.append(sum(n for n in ephem_data.values()))
      ephem_data = defaultdict(int)

      meas_cnt += 1
      print_stats(start_time, ephem_stats, sat_meas, meas_cnt, locktime_per_sat)
      print(f"Total     e/min: {sum(total_ephems_per_min)/meas_cnt}")
      print(f"Different e/min: {sum(diff_ephems_per_min)/meas_cnt}")

      sat_meas = defaultdict(int)
      locktime_per_sat = {}
      last_log = time.monotonic()

if __name__ == "__main__":
  main()

