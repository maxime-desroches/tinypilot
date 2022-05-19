#!/usr/bin/env python3
from typing import List

import numpy as np
from collections import defaultdict

from cereal import log, messaging
from laika import AstroDog
from laika.helpers import ConstellationId
from laika.raw_gnss import GNSSMeasurement, calc_pos_fix, correct_measurements, process_measurements, read_raw_ublox
from selfdrive.locationd.models.constants import GENERATED_DIR, ObservationKind
from selfdrive.locationd.models.gnss_kf import GNSSKalman
from selfdrive.locationd.models.gnss_kf import States as GStates
import common.transformations.coordinates as coord


class Laikad:

  def __init__(self):
    self._gnss_kf = GNSSKalman(GENERATED_DIR)

  def process_ublox_msg(self, ublox_msg, dog: AstroDog, ublox_mono_time: int):
    if ublox_msg.which == 'measurementReport':
      report = ublox_msg.measurementReport
      if len(report.measurements) == 0:
        return None

      new_meas = read_raw_ublox(report)
      measurements = process_measurements(new_meas, dog)
      if len(measurements) == 0:
        return None

      pos_fix = calc_pos_fix(measurements)
      # pos fix needs more than 5 processed_measurements
      if len(pos_fix) > 0:
        measurements = correct_measurements(measurements, pos_fix[0][:3], dog)
      if len(measurements) == 0:
        return None
      meas_msgs = [create_measurement_msg(m) for m in measurements]

      t = ublox_mono_time * 1e-9

      localizer_valid = self.update_localizer(pos_fix, t, measurements)
      if not localizer_valid:
        return None

      ecef_pos = self._gnss_kf.x[GStates.ECEF_POS].tolist()
      ecef_vel = self._gnss_kf.x[GStates.ECEF_VELOCITY].tolist()
      bearing_deg, bearing_std = get_bearing_from_gnss(self._gnss_kf)
      dat = messaging.new_message('gnssMeasurements')

      dat.gnssMeasurements = {
        "positionECEF": ecef_pos,
        "velocityECEF": ecef_vel,
        "bearingDeg": float(bearing_deg),
        "bearingAccuracyDeg": float(bearing_std),
        "ubloxMonoTime": ublox_mono_time,
        "correctedMeasurements": meas_msgs
      }
      return dat

  def update_localizer(self, pos_fix, t, measurements):
    if not self.localizer_valid(t):
      # A position fix is needed when resetting the kalman filter.
      if len(pos_fix) == 0:
        return False
      post_est = pos_fix[0][:3].tolist()
      if self._gnss_kf.filter.filter_time is None:
        print("Init gnss kalman filter")
      else:
        print("Time gap of over 10s detected, gnss kalman reset")
      self.init_gnss_localizer(post_est)
    kf_add_observations(self._gnss_kf, t, measurements)
    return True

  def localizer_valid(self, t):
    return self._gnss_kf.filter.filter_time is not None and (self._gnss_kf.filter.filter_time - t) < 10


  def init_gnss_localizer(self, est_pos):
    x_initial, p_initial_diag = np.copy(GNSSKalman.x_initial), np.copy(np.diagonal(GNSSKalman.P_initial))
    x_initial[GStates.ECEF_POS] = est_pos
    p_initial_diag[GStates.ECEF_POS] = 1000 ** 2

    self._gnss_kf.init_state(x_initial, covs_diag=p_initial_diag)


def create_measurement_msg(meas: GNSSMeasurement):
  c = log.GnssMeasurements.CorrectedMeasurement.new_message()
  c.constellationId = meas.constellation_id.value
  c.svId = meas.sv_id
  observables = meas.observables_final
  c.glonassFrequency = meas.glonass_freq if meas.constellation_id == ConstellationId.GLONASS else 0
  c.pseudorange = float(observables['C1C'])
  c.pseudorangeStd = float(meas.observables_std['C1C'])
  c.pseudorangeRate = float(observables['D1C'])
  c.pseudorangeRateStd = float(meas.observables_std['D1C'])
  c.satPos = meas.sat_pos_final.tolist()
  c.satVel = meas.sat_vel.tolist()
  return c


def kf_add_observations(gnss_kf: GNSSKalman, t: float, measurements: List[GNSSMeasurement]):
  ekf_data = defaultdict(list)
  for m in measurements:
    m_arr = m.as_array()
    if m.constellation_id == ConstellationId.GPS:
      ekf_data[ObservationKind.PSEUDORANGE_GPS].append(m_arr)
      ekf_data[ObservationKind.PSEUDORANGE_RATE_GPS].append(m_arr)
    elif m.constellation_id == ConstellationId.GLONASS:
      ekf_data[ObservationKind.PSEUDORANGE_GLONASS].append(m_arr)
      ekf_data[ObservationKind.PSEUDORANGE_RATE_GLONASS].append(m_arr)

  for kind, data in ekf_data.items():
    gnss_kf.predict_and_observe(t, kind, data)


def get_bearing_from_gnss(gnss_kf: GNSSKalman):
  ecef_pos = gnss_kf.x[GStates.ECEF_POS]
  ecef_vel = gnss_kf.x[GStates.ECEF_VELOCITY]
  ecef_vel_std = gnss_kf.P[GStates.ECEF_VELOCITY]
  vel_std = np.linalg.norm(ecef_vel_std)

  # init orientation with direction of velocity
  converter = coord.LocalCoord.from_ecef(ecef_pos)

  ned_vel = np.einsum('ij,j ->i', converter.ned_from_ecef_matrix, ecef_vel)
  bearing = np.arctan2(ned_vel[1], ned_vel[0])
  bearing_std = np.arctan2(vel_std, np.linalg.norm(ned_vel))
  return np.rad2deg(bearing), bearing_std


def main():
  dog = AstroDog(use_internet=True)
  sm = messaging.SubMaster(['ubloxGnss'])
  pm = messaging.PubMaster(['gnssMeasurements'])

  laikad = Laikad()

  while True:
    sm.update()

    # Todo if no internet available use latest ephemeris
    if sm.updated['ubloxGnss']:
      ublox_msg = sm['ubloxGnss']
      msg = laikad.process_ublox_msg(ublox_msg, dog, sm.logMonoTime['ubloxGnss'])
      if msg is None:
        msg = messaging.new_message('gnssMeasurements')
        msg.valid = False
      pm.send('gnssMeasurements', msg)


if __name__ == "__main__":
  main()
