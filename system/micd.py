#!/usr/bin/env python3
import sounddevice as sd
import numpy as np

from cereal import messaging
from common.filter_simple import FirstOrderFilter
from common.realtime import Ratekeeper
from system.swaglog import cloudlog

RATE = 10
DT_MIC = 1. / RATE


class Mic:
  def __init__(self, pm, sm):
    self.pm = pm
    self.sm = sm
    self.rk = Ratekeeper(RATE)

    self.measurements = np.array([])
    self.filter = FirstOrderFilter(1, 3, DT_MIC)

  def update(self):
    self.sm.update(0)

    noise_level_raw = min(float(np.linalg.norm(self.measurements)), 5.)
    if len(self.measurements) > 0:
      self.filter.update(noise_level_raw)
    self.measurements = np.array([])

    msg = messaging.new_message('microphone')
    microphone = msg.microphone
    microphone.ambientNoiseLevelRaw = noise_level_raw
    microphone.filteredAmbientNoiseLevel = self.filter.x

    self.pm.send('microphone', msg)
    self.rk.keep_time()

  def callback(self, indata, frames, time, status):
    self.measurements = np.concatenate((self.measurements, indata[:, 0]))

  def micd_thread(self, device=None):
    if device is None:
      device = "sysdefault"

    with sd.InputStream(device=device, channels=1, samplerate=44100, callback=self.callback) as stream:
      cloudlog.info(f"micd stream started: {stream.samplerate=} {stream.channels=} {stream.dtype=} {stream.device=}")
      while True:
        self.update()


def main(pm=None, sm=None):
  if pm is None:
    pm = messaging.PubMaster(['microphone'])
  if sm is None:
    sm = messaging.SubMaster(['controlsState'])

  mic = Mic(pm, sm)
  mic.micd_thread()


if __name__ == "__main__":
  main()
