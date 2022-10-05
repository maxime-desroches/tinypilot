#!/usr/bin/env python3
import sys
import time
from typing import List

import cereal.messaging as messaging
from selfdrive.manager.process_config import managed_processes

DELTA = 0.001
# assume running openpilot for now
procs: List[str] = []#"ubloxd", "pigeond"]


def main():
  if len(sys.argv) != 4:
    print("args: <q|u> <latitude> <longitude>")
    return

  quectel_mod = sys.argv[1] == 'q'
  sol_lat = float(sys.argv[2])
  sol_lon = float(sys.argv[3])

  for p in procs:
    managed_processes[p].start()
    time.sleep(0.5) # give time to startup

  socket = 'gpsLocationExternal'
  if quectel_mod:
    socket = 'gpsLocation'

  gps_sock = messaging.sub_sock(socket, timeout=0.1)

  # analyze until the location changed
  while True:
    events = messaging.drain_sock(gps_sock)
    for e in events:

      if quectel_mod:
        lat = e.gpsLocation.latitude
        lon = e.gpsLocation.longitude
      else:
        lat = e.gpsLocationExternal.latitude
        lon = e.gpsLocationExternal.longitude

      if abs(lat - sol_lat) < DELTA and abs(lon - sol_lon) < DELTA:
        print("MATCH")
        return

    time.sleep(0.1)

    for p in procs:
      if not managed_processes[p].proc.is_alive():
        print(f"ERROR: '{p}' died")
        return


if __name__ == "__main__":
  main()
  for p in procs:
    managed_processes[p].stop()
