#!/usr/bin/env python3
import sys
from tools.lib.logreader import LogReader
from selfdrive.car.fw_versions import match_fw_to_car


def fw_versions_to_dict(car_fw):
  fw_versions = {}
  for f in car_fw:
    addr = f.address
    subaddr = f.subAddress
    if subaddr == 0:
      subaddr = None
    fw_versions[(addr, subaddr)] = f.fwVersion

  return fw_versions


if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Usage: ./test_fw_query_on_routes.py <route_list>")
    sys.exit(1)

  i = 0
  for route in open(sys.argv[1]):
    route = route.rstrip()

    dongle_id, time = route.split('|')
    qlog_path = f"cd:/{dongle_id}/{time}/0/qlog.bz2"
    lr = LogReader(qlog_path)

    for msg in lr:
      if msg.which() == "carParams":
        car_fw = msg.carParams.carFw
        if len(car_fw) == 0:
          break

        live_fingerprint = msg.carParams.carFingerprint

        fw_versions = fw_versions_to_dict(car_fw)
        candidates = match_fw_to_car(fw_versions)
        if (len(candidates) == 1) and (list(candidates)[0] == live_fingerprint):
          print("Correct", live_fingerprint)
          break

        print("Old style:", live_fingerprint)
        print("New style:", candidates)
        print(msg.carParams.carFw)

        i += 1
        break

  print(f"Unfingerprinted cars {i}")
