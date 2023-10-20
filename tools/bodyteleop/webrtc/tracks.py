#!/usr/bin/env python3

import argparse
import json
import time
from typing import Awaitable, Callable, Any

import aiortc
from aiortc.mediastreams import VIDEO_CLOCK_RATE, VIDEO_TIME_BASE
import av
import asyncio
import numpy as np

import pyaudio
from pydub import AudioSegment

from cereal import messaging
from openpilot.common.realtime import DT_MDL, DT_DMON
from openpilot.tools.lib.framereader import FrameReader


class TiciVideoStreamTrack(aiortc.MediaStreamTrack):
  def __init__(self, camera_type, dt, time_base=VIDEO_TIME_BASE, clock_rate=VIDEO_CLOCK_RATE):
    assert camera_type in ["driver", "wideRoad", "road"]
    super().__init__()
    # override track id to include camera type - client needs that for identification
    self._id = f"{camera_type}:{self._id}"
    self._dt = dt
    self._time_base = time_base
    self._clock_rate = clock_rate
    self._start = None

  async def next_pts(self, current_pts):
    pts = current_pts + self._dt * self._clock_rate

    data_time = pts * self._time_base
    if self._start is None:
      self._start = time.time() - data_time
    else:
      wait_time = self._start + data_time - time.time()
      await asyncio.sleep(wait_time)

    return pts


class LiveStreamVideoStreamTrack(TiciVideoStreamTrack):
  kind = "video"
  camera_to_sock_mapping = {
    "driver": "livestreamDriverEncodeData",
    "wideRoad": "livestreamWideRoadEncodeData",
    "road": "livestreamRoadEncodeData",
  }

  def __init__(self, camera_type):
    super().__init__(camera_type, DT_MDL)

    self._sock = messaging.sub_sock(self.camera_to_sock_mapping[camera_type], conflate=True)
    self._pts = 0

  async def recv(self):
    while True:
      msg = messaging.recv_one_or_none(self._sock)
      if msg is not None:
        break
      await asyncio.sleep(0.005)

    evta = getattr(msg, msg.which())

    packet = av.Packet(evta.header + evta.data)
    packet.time_base = self._time_base
    packet.pts = self._pts

    self._pts += self._dt * self._clock_rate

    return packet


class DummyVideoStreamTrack(TiciVideoStreamTrack):
  kind = "video"

  def __init__(self, color=0, dt=DT_MDL, camera_type="driver"):
    super().__init__(camera_type, dt)
    self._color = color
    self._pts = 0

  async def recv(self):
    print("-- sending frame", self._pts)
    img = np.full((1920, 1080, 3), self._color, dtype=np.uint8)

    new_frame = av.VideoFrame.from_ndarray(img, format="rgb24")
    new_frame.pts = self._pts
    new_frame.time_base = self._time_base

    self._pts = await self.next_pts(self._pts)

    return new_frame


class FrameReaderVideoStreamTrack(TiciVideoStreamTrack):
  kind = "video"

  def __init__(self, input_file, dt=DT_MDL, camera_type="driver"):
    super().__init__(camera_type, dt)

    frame_reader = FrameReader(input_file)
    self._frames = [frame_reader.get(i, pix_fmt="rgb24") for i in range(frame_reader.frame_count)]
    self._frame_count = len(self.frames)
    self._frame_index = 0
    self._pts = 0

  async def recv(self):
    print("-- sending frame", self._pts)
    img = self._frames[self._frame_index]

    new_frame = av.VideoFrame.from_ndarray(img, format="rgb24")
    new_frame.pts = self._pts
    new_frame.time_base = self._time_base

    self._frame_index = (self._frame_index + 1) % self._frame_count
    self._pts = await self.next_pts(self._pts)

    return new_frame


class AudioInputStreamTrack(aiortc.mediastreams.AudioStreamTrack):
  def __init__(self, format=pyaudio.paInt16, rate=16000, channels=1, packet_time=0.020, device_index=None):
    super().__init__()

    self.p = pyaudio.PyAudio()
    frame_per_buffer = packet_time * rate
    self.stream = self.p.open(format=format, 
                              channels=channels, 
                              rate=rate, 
                              frames_per_buffer=frame_per_buffer, 
                              input=True, 
                              input_device_index=device_index)
    self.format = format
    self.rate = rate
    self.channels = channels
    self.packet_time = packet_time
    self.chunk_size = int(rate * packet_time)
    self.chunk_index = 0
    self.pts = 0

  async def recv(self):
    mic_data = self.stream.read(self.chunk_size)
    mic_sound = AudioSegment(mic_data, sample_width=pyaudio.get_sample_size(self.format), channels=self.channels, frame_rate=self.rate)
    # create stereo sound?
    mic_sound = AudioSegment.from_mono_audiosegments(mic_sound, mic_sound)
    packet = av.Packet(mic_sound.raw_data)
    # TODO
    return None
