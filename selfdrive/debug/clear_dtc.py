#!/usr/bin/env python3
import argparse
import sys
from subprocess import CalledProcessError, check_output

from panda import Panda
from panda.python.uds import DTC_GROUP_TYPE, SESSION_TYPE, MessageTimeoutError, UdsClient

parser = argparse.ArgumentParser(description="clear DTC status")
parser.add_argument("addr", type=lambda x: int(x,0), nargs="?", default=0x7DF) # default is functional (broadcast) address
parser.add_argument("--bus", type=int, default=0)
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

try:
  check_output(["pidof", "boardd"])
  print("boardd is running, please kill openpilot before running this script! (aborted)")
  sys.exit(1)
except CalledProcessError as e:
  if e.returncode != 1: # 1 == no process found (boardd not running)
    raise e

panda = Panda()
panda.set_safety_mode(Panda.SAFETY_ELM327)
uds_client = UdsClient(panda, args.addr, bus=args.bus, debug=args.debug)
print("extended diagnostic session ...")
try:
  uds_client.diagnostic_session_control(SESSION_TYPE.EXTENDED_DIAGNOSTIC)
except MessageTimeoutError:
  # functional address isn't properly handled so a timeout occurs
  if args.addr != 0x7DF:
    raise
print("clear diagnostic info ...")
try:
  uds_client.clear_diagnostic_information(DTC_GROUP_TYPE.ALL)
except MessageTimeoutError:
  # functional address isn't properly handled so a timeout occurs
  if args.addr != 0x7DF:
    pass
print("")
print("you may need to power cycle your vehicle now")
