#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Any
from collections import namedtuple

from selfdrive.car.car_helpers import interface_names
from selfdrive.test.process_replay.compare_logs import compare_logs
from selfdrive.test.process_replay.process_replay import CONFIGS, replay_process
from tools.lib.logreader import LogReader

INJECT_MODEL = 0
Segment = namedtuple('Segment', ['car_brand', 'route', 'whitelist_procs', 'blacklist_procs'])

segments = [
  Segment(
    car_brand="HONDA", # HONDA.ACCORD
    route="0375fdf7b1ce594d|2019-06-13--08-32-25--3",
    whitelist_procs=[],
    blacklist_procs=[]
  ),
  Segment(
    car_brand="HONDA", # HONDA.CIVIC
    route="99c94dc769b5d96e|2019-08-03--14-19-59--2",
    whitelist_procs=[],
    blacklist_procs=[]
  ),
  Segment(
    car_brand="TOYOTA", # TOYOTA.COROLLA_TSS2
    route="77611a1fac303767|2020-02-29--13-29-33--3",
    whitelist_procs=[],
    blacklist_procs=[]
  ),
  Segment(
    car_brand="TOYOTA", # TOYOTA.RAV4  (LQR)
    route="b14c5b4742e6fc85|2020-10-14--11-04-47--4",
    whitelist_procs=["controlsd"],
    blacklist_procs=[]
  ),
  Segment(
    car_brand="TOYOTA", # TOYOTA.PRIUS (INDI)
    route="0982d79ebb0de295|2020-10-18--19-11-36--5",
    whitelist_procs=["controlsd"],
    blacklist_procs=[]
  ),
  Segment(
    car_brand="GM", # GM.ACADIA
    route="7cc2a8365b4dd8a9|2018-12-02--12-10-44--2",
    whitelist_procs=[],
    blacklist_procs=["ubloxd", "locationd"]
  ),
  Segment(
    car_brand="CHRYSLER", # CHRYSLER.PACIFICA
    route="b6849f5cf2c926b1|2020-02-28--07-29-48--13",
    whitelist_procs=[],
    blacklist_procs=[]
  ),
  Segment(
    car_brand="HYUNDAI", # HYUNDAI.SONATA
    route="5b7c365c50084530|2020-04-15--16-13-24--3",
    whitelist_procs=[],
    blacklist_procs=[]
  ),
  #Segment(
  #  car_brand="CHRYSLER", # CHRYSLER.JEEP_CHEROKEE
  #  route="b6e1317e1bfbefa6|2020-03-04--13-11-40",
  #  whitelist_procs=[],
  #  blacklist_procs=[]
  #),
  Segment(
    car_brand="SUBARU", # SUBARU.IMPREZA
    route="7873afaf022d36e2|2019-07-03--18-46-44--0",
    whitelist_procs=[],
    blacklist_procs=[]
  ),
  Segment(
    car_brand="VOLKSWAGEN", # VW.GOLF
    route="76b83eb0245de90e|2020-03-05--19-16-05--3",
    whitelist_procs=[],
    blacklist_procs=["ubloxd", "locationd"]
  ),
  Segment(
    car_brand="NISSAN", # NISSAN.XTRAIL
    route="fbbfa6af821552b9|2020-03-03--08-09-43--0",
    whitelist_procs=[],
    blacklist_procs=["ubloxd", "locationd"]
  ),

  # Enable when port is tested and dascamOnly is no longer set
  #Segment(
  #  car_brand="MAZDA", # MAZDA.CX5
  #  route="32a319f057902bb3|2020-04-27--15-18-58--2",
  #  whitelist_procs=[],
  #  blacklist_procs=[]
  #),
]

# ford doesn't need to be tested until a full port is done
excluded_interfaces = ["mock", "ford", "mazda"]

BASE_URL = "https://commadataci.blob.core.windows.net/openpilotci/"

# run the full test (including checks) when no args given
FULL_TEST = len(sys.argv) <= 1


def get_segment(segment_name, original=True):
  route_name, segment_num = segment_name.rsplit("--", 1)
  if original:
    rlog_url = BASE_URL + "%s/%s/rlog.bz2" % (route_name.replace("|", "/"), segment_num)
  else:
    process_replay_dir = os.path.dirname(os.path.abspath(__file__))
    model_ref_commit = open(os.path.join(process_replay_dir, "model_ref_commit")).read().strip()
    rlog_url = BASE_URL + "%s/%s/rlog_%s.bz2" % (route_name.replace("|", "/"), segment_num, model_ref_commit)

  return rlog_url


def test_process(cfg, lr, cmp_log_fn, ignore_fields=None, ignore_msgs=None):
  if ignore_fields is None:
    ignore_fields = []
  if ignore_msgs is None:
    ignore_msgs = []
  url = BASE_URL + os.path.basename(cmp_log_fn)
  cmp_log_msgs = list(LogReader(url))

  log_msgs = replay_process(cfg, lr)

  # check to make sure openpilot is engaged in the route
  # TODO: update routes so enable check can run
  #       failed enable check: honda bosch, hyundai, chrysler, and subaru
  if cfg.proc_name == "controlsd" and FULL_TEST and False:
    for msg in log_msgs:
      if msg.which() == "controlsState":
        if msg.controlsState.active:
          break
    else:
      segment = cmp_log_fn.split("/")[-1].split("_")[0]
      raise Exception("Route never enabled: %s" % segment)

  try:
    return compare_logs(cmp_log_msgs, log_msgs, ignore_fields+cfg.ignore, ignore_msgs, cfg.tolerance)
  except Exception as e:
    return str(e)

def format_diff(results, ref_commit):
  diff1, diff2 = "", ""
  diff2 += "***** tested against commit %s *****\n" % ref_commit

  failed = False
  for segment, result in list(results.items()):
    diff1 += "***** results for segment %s *****\n" % segment
    diff2 += "***** differences for segment %s *****\n" % segment

    for proc, diff in list(result.items()):
      diff1 += "\t%s\n" % proc
      diff2 += "*** process: %s ***\n" % proc

      if isinstance(diff, str):
        diff1 += "\t\t%s\n" % diff
        failed = True
      elif len(diff):
        cnt = {}
        for d in diff:
          diff2 += "\t%s\n" % str(d)

          k = str(d[1])
          cnt[k] = 1 if k not in cnt else cnt[k] + 1

        for k, v in sorted(cnt.items()):
          diff1 += "\t\t%s: %s\n" % (k, v)
        failed = True
  return diff1, diff2, failed

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="Regression test to identify changes in a process's output")

  # whitelist has precedence over blacklist in case both are defined
  parser.add_argument("--whitelist-procs", type=str, nargs="*", default=[],
                        help="Whitelist given processes from the test (e.g. controlsd)")
  parser.add_argument("--whitelist-cars", type=str, nargs="*", default=[],
                        help="Whitelist given cars from the test (e.g. HONDA)")
  parser.add_argument("--blacklist-procs", type=str, nargs="*", default=[],
                        help="Blacklist given processes from the test (e.g. controlsd)")
  parser.add_argument("--blacklist-cars", type=str, nargs="*", default=[],
                        help="Blacklist given cars from the test (e.g. HONDA)")
  parser.add_argument("--ignore-fields", type=str, nargs="*", default=[],
                        help="Extra fields or msgs to ignore (e.g. carState.events)")
  parser.add_argument("--ignore-msgs", type=str, nargs="*", default=[],
                        help="Msgs to ignore (e.g. carEvents)")
  args = parser.parse_args()

  cars_whitelisted = len(args.whitelist_cars) > 0
  procs_whitelisted = len(args.whitelist_procs) > 0

  process_replay_dir = os.path.dirname(os.path.abspath(__file__))
  try:
    ref_commit = open(os.path.join(process_replay_dir, "ref_commit")).read().strip()
  except FileNotFoundError:
    print("couldn't find reference commit")
    sys.exit(1)

  print("***** testing against commit %s *****" % ref_commit)

  # check to make sure all car brands are tested
  if FULL_TEST:
    tested_cars = set(s.car_brand.lower() for s in segments)
    untested = (set(interface_names) - set(excluded_interfaces)) - tested_cars
    assert len(untested) == 0, "Cars missing routes: %s" % (str(untested))

  results: Any = {}
  for s in segments:
    segment = s.route
    car_brand = s.car_brand
    print(car_brand)
    if (cars_whitelisted and car_brand.upper() not in args.whitelist_cars) or \
       (not cars_whitelisted and car_brand.upper() in args.blacklist_cars):
      continue
    
    print("***** testing route segment %s *****\n" % segment)

    results[segment] = {}

    rlog_fn = get_segment(segment)
    lr = LogReader(rlog_fn)

    for cfg in CONFIGS:
      if (procs_whitelisted and cfg.proc_name not in args.whitelist_procs) or \
         (not procs_whitelisted and cfg.proc_name in args.blacklist_procs):
        continue
      #In case of full test, use white/blacklists from Segment
      if len(args.whitelist_procs)==0 and len(args.blacklist_procs)==0:
        if (len(s.whitelist_procs)>0 and cfg.proc_name not in s.whitelist_procs) or \
          cfg.proc_name in s.blacklist_procs:
          continue
      
      cmp_log_fn = os.path.join(process_replay_dir, "%s_%s_%s.bz2" % (segment, cfg.proc_name, ref_commit))
      results[segment][cfg.proc_name] = test_process(cfg, lr, cmp_log_fn, args.ignore_fields, args.ignore_msgs)

  diff1, diff2, failed = format_diff(results, ref_commit)
  with open(os.path.join(process_replay_dir, "diff.txt"), "w") as f:
    f.write(diff2)
  print(diff1)

  print("TEST", "FAILED" if failed else "SUCCEEDED")

  print("\n\nTo update the reference logs for this test run:")
  print("./update_refs.py")

  sys.exit(int(failed))
