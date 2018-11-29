#!/usr/bin/env python
import zmq
import math
import time
from collections import namedtuple
from cereal import car
from selfdrive.can.parser import CANParser
from selfdrive.car.gm.interface import CanBus
from selfdrive.car.gm.values import DBC, CAR
from common.realtime import sec_since_boot
from selfdrive.services import service_list
from selfdrive.config import Conversions as CV
import selfdrive.messaging as messaging
import selfdrive.rcp_helpers as rcp


RADAR_HEADER_MSG = 1120
SLOT_1_MSG = RADAR_HEADER_MSG + 1
NUM_SLOTS = 20

# Actually it's 0x47f, but can parser only reports
# messages that are present in DBC
LAST_RADAR_MSG = RADAR_HEADER_MSG + NUM_SLOTS

def create_radard_can_parser(canbus, car_fingerprint):
  dbc_f = DBC[car_fingerprint]['radar']
  if car_fingerprint in (CAR.VOLT, CAR.MALIBU, CAR.HOLDEN_ASTRA):
    # C1A-ARS3-A by Continental
    radar_trackers = range(SLOT_1_MSG, SLOT_1_MSG + NUM_SLOTS)
    radar_messages = [RADAR_HEADER_MSG] + radar_trackers

    sig = namedtuple('sig', 'name_value msg')
    headers = [('FLRRNumValidTargets', 0),
               ('FLRRSnsrBlckd', 0),
               ('FLRRYawRtPlsblityFlt', 0),
               ('FLRRHWFltPrsntInt', 0),
               ('FLRRAntTngFltPrsnt', 0),
               ('FLRRAlgnFltPrsnt', 0),
               ('FLRRSnstvFltPrsntInt', 0)]
    header_sig = sig(headers, [RADAR_HEADER_MSG])

    trackers = [('TrkRange', 0.0),
                ('TrkRangeRate', 0.0),
                ('TrkRangeAccel', 0.0),
                ('TrkAzimuth', 0.0),
                ('TrkWidth', 0.0),
                ('TrkObjectID', 0)]
    tracker_sig = sig(trackers, radar_trackers)

    signals = rcp.create_radar_signals(header_sig, tracker_sig)
    checks = rcp.create_radar_checks(radar_messages, select = "none")

    return CANParser(dbc_f, signals, checks, canbus.obstacle)
  else:
    return None


class RadarInterface(object):
  def __init__(self, CP):
    # radar
    self.pts = {}

    self.delay = 0.0  # Delay of radar

    canbus = CanBus()
    print "Using %d as obstacle CAN bus ID" % canbus.obstacle
    self.rcp = create_radard_can_parser(canbus, CP.carFingerprint)

    context = zmq.Context()
    self.logcan = messaging.sub_sock(context, service_list['can'].port)

  def update(self):
    updated_messages = set()
    ret = car.RadarState.new_message()
    while 1:

      if self.rcp is None:
        time.sleep(0.05)   # nothing to do
        return ret

      tm = int(sec_since_boot() * 1e9)
      updated_messages.update(self.rcp.update(tm, True))
      if LAST_RADAR_MSG in updated_messages:
        break

    header = self.rcp.vl[RADAR_HEADER_MSG]
    fault = header['FLRRSnsrBlckd'] or header['FLRRSnstvFltPrsntInt'] or \
      header['FLRRYawRtPlsblityFlt'] or header['FLRRHWFltPrsntInt'] or \
      header['FLRRAntTngFltPrsnt'] or header['FLRRAlgnFltPrsnt']
    errors = []
    if not self.rcp.can_valid:
      errors.append("commIssue")
    if fault:
      errors.append("fault")
    ret.errors = errors

    currentTargets = set()
    num_targets = header['FLRRNumValidTargets']

    # Not all radar messages describe targets,
    # no need to monitor all of the self.rcp.msgs_upd
    for ii in updated_messages:
      if ii == RADAR_HEADER_MSG:
        continue

      if num_targets == 0:
        break

      cpt = self.rcp.vl[ii]
      # Zero distance means it's an empty target slot
      if cpt['TrkRange'] > 0.0:
        targetId = cpt['TrkObjectID']
        currentTargets.add(targetId)
        if targetId not in self.pts:
          self.pts[targetId] = car.RadarState.RadarPoint.new_message()
          self.pts[targetId].trackId = targetId
        distance = cpt['TrkRange']
        self.pts[targetId].dRel = distance # from front of car
        # From driver's pov, left is positive
        self.pts[targetId].yRel = math.sin(cpt['TrkAzimuth'] * CV.DEG_TO_RAD) * distance
        self.pts[targetId].vRel = cpt['TrkRangeRate']
        self.pts[targetId].aRel = float('nan')
        self.pts[targetId].yvRel = float('nan')

    for oldTarget in self.pts.keys():
      if not oldTarget in currentTargets:
        del self.pts[oldTarget]

    ret.points = self.pts.values()
    return ret

if __name__ == "__main__":
  RI = RadarInterface(None)
  while 1:
    ret = RI.update()
    print(chr(27) + "[2J")
    print ret
