#!/usr/bin/env python3
from cereal import car
from opendbc.can.parser import CANParser
from selfdrive.car.tesla.values import DBC, CANBUS
from selfdrive.car.interfaces import RadarInterfaceBase

RADAR_MSGS_A = list(range(0x371, 0x37E, 3))
RADAR_MSGS_B = list(range(0x372, 0x37F, 3))
NUM_POINTS = len(RADAR_MSGS_A)
SGU_INDEX_VALUES = [f"RADC_ACCTargObj{i+1}_sguIndex" for i in range(NUM_POINTS)]
SGU_INDEX_NONE = 63

def get_radar_can_parser(CP):
  # Status messages
  signals = [
    ('RADC_HWFail', 'TeslaRadarSguInfo', 0),
    ('RADC_SGUFail', 'TeslaRadarSguInfo', 0),
    ('RADC_SensorDirty', 'TeslaRadarSguInfo', 0),
  ]

  checks = [
    ('TeslaRadarSguInfo', 10),
    ('TeslaRadarTguInfo', 10),
  ]

  # Radar tracks. There are also raw point clouds available,
  # we don't use those.
  for i in range(NUM_POINTS):
    msg_id_a = RADAR_MSGS_A[i]
    msg_id_b = RADAR_MSGS_B[i]

    # There is a bunch more info in the messages,
    # but these are the only things actually used in openpilot
    signals.extend([
      ('LongDist', msg_id_a, 255),
      ('LongSpeed', msg_id_a, 0),
      ('LatDist', msg_id_a, 0),
      ('LongAccel', msg_id_a, 0),
      ('Meas', msg_id_a, 0),
      ('Tracked', msg_id_a, 0),
      ('Index', msg_id_a, 0),

      ('LatSpeed', msg_id_b, 0),
      ('Index2', msg_id_b, 0),

      (SGU_INDEX_VALUES[i], "TeslaRadarTguInfo", 63),
    ])

    checks.extend([
      (msg_id_a, 8),
      (msg_id_b, 8),
    ])

  return CANParser(DBC[CP.carFingerprint]['radar'], signals, checks, CANBUS.radar)

class RadarInterface(RadarInterfaceBase):
  def __init__(self, CP):
    super().__init__(CP)
    self.rcp = get_radar_can_parser(CP)
    self.updated_messages = set()

  def update(self, can_strings):
    if self.rcp is None:
      return super().update(None)

    values = self.rcp.update_strings(can_strings)
    self.updated_messages.update(values)

    # Trigger update only on new first track message
    if RADAR_MSGS_B[-1] not in self.updated_messages:
      return None

    ret = car.RadarData.new_message()

    # Errors
    errors = []
    sgu_info = self.rcp.vl['TeslaRadarSguInfo']
    if not self.rcp.can_valid:
      errors.append('canError')
    if sgu_info['RADC_HWFail'] or sgu_info['RADC_SGUFail'] or sgu_info['RADC_SensorDirty']:
      errors.append('fault')
    ret.errors = errors

    # Radar tracks
    for i in range(NUM_POINTS):
      msg_a = self.rcp.vl[RADAR_MSGS_A[i]]
      msg_b = self.rcp.vl[RADAR_MSGS_B[i]]

      # Make sure msg A and B are together
      if msg_a['Index'] != msg_b['Index2']:
        continue

      # Check if it's a valid track
      track_id = self.rcp.vl['TeslaRadarTguInfo'][SGU_INDEX_VALUES[i]]
      if track_id == SGU_INDEX_NONE or not msg_a['Tracked']:
        if i in self.pts:
          del self.pts[i]
        continue

      if i not in self.pts:
        self.pts[i] = car.RadarData.RadarPoint.new_message()

      # Parse track data
      self.pts[i].trackId = track_id
      self.pts[i].dRel = msg_a['LongDist']
      self.pts[i].yRel = msg_a['LatDist']
      self.pts[i].vRel = msg_a['LongSpeed']
      self.pts[i].aRel = msg_a['LongAccel']
      self.pts[i].yvRel = msg_b['LatSpeed']
      self.pts[i].measured = bool(msg_a['Meas'])

    ret.points = list(self.pts.values())
    self.updated_messages.clear()
    return ret
