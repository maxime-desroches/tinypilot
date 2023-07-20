#!/usr/bin/env python3
import os
from typing import Any
import unittest
import time

import cereal.messaging as messaging
from cereal.visionipc import VisionIpcClient, VisionStreamType
from selfdrive.manager.process_config import managed_processes

LLK_DECIMATION = 10
CACHE_PATH = "/data/mbgl-cache-navd.db"

def gen_llk(location):
  msg = messaging.new_message('liveLocationKalman')
  msg.liveLocationKalman.positionGeodetic = {'value': [*location, 0], 'std': [0., 0., 0.], 'valid': True}
  msg.liveLocationKalman.calibratedOrientationNED = {'value': [0., 0., 0.], 'std': [0., 0., 0.], 'valid': True}
  msg.liveLocationKalman.status = 'valid'
  return msg

import requests
import threading
import http.server


class MapBoxInternetDisabledRequestHandler(http.server.BaseHTTPRequestHandler):
  INTERNET_ACTIVE = True

  def setup(self):
    if self.INTERNET_ACTIVE:
      super().setup()
  
  def handle(self):
    if self.INTERNET_ACTIVE:
      super().handle()

  def finish(self):
    if self.INTERNET_ACTIVE:
      super().finish()
    
  def do_GET(self):
    url = f'https://api.mapbox.com{self.path}'

    #token_regex = r'access_token=([\d\w.]*)'
    #url = re.sub(token_regex, "access_token=invalid_token", url)

    headers = dict(self.headers)
    headers["Host"] = "api.mapbox.com"

    r = requests.get(url, headers=headers, timeout=5)

    self.send_response(r.status_code)
    
    self.end_headers()

    self.wfile.write(r.content)
  
  def log_message(self, *args: Any) -> None:
    return

  def log_error(self, *args: Any) -> None:
    return


class MapBoxInternetDisabledServer(threading.Thread):
  def run(self):
    self.server = http.server.HTTPServer(("127.0.0.1", 5000), MapBoxInternetDisabledRequestHandler)
    self.server.serve_forever()
  
  def stop(self):
    self.server.shutdown()

LOCATION1 = (32.7174, -117.16277)
LOCATION2 = (32.7558, -117.2037)


class TestMapRenderer(unittest.TestCase):
  server = MapBoxInternetDisabledServer()

  @classmethod
  def setUpClass(cls):
    assert "MAPBOX_TOKEN" in os.environ
    cls.original_token = os.environ["MAPBOX_TOKEN"]
    cls.server.start()
    time.sleep(2) # wait for server to be fully started

  @classmethod
  def tearDownClass(cls) -> None:
    cls.server.stop()

  def setUp(self):
    self.sm = messaging.SubMaster(['mapRenderState'])
    self.pm = messaging.PubMaster(['liveLocationKalman'])
    self.vipc = VisionIpcClient("navd", VisionStreamType.VISION_STREAM_MAP, True)

    if os.path.exists(CACHE_PATH):
      os.remove(CACHE_PATH)
    
    self.location = LOCATION1
  
  def tearDown(self):
    managed_processes['mapsd'].stop()
  
  def _setup_test(self):
    os.environ['MAPS_HOST'] = 'http://localhost:5000'
    # start + sync up
    managed_processes['mapsd'].start()

    assert self.pm.wait_for_readers_to_update("liveLocationKalman", 10)

    assert VisionIpcClient.available_streams("navd", False) == {VisionStreamType.VISION_STREAM_MAP, }
    assert self.vipc.connect(False)
    self.vipc.recv()
  
  def _run_test(self, expect_valid):
    starting_frame_id = None

    # run test
    for i in range(30*LLK_DECIMATION):
      frame_expected = (i+1) % LLK_DECIMATION == 0

      if self.sm.logMonoTime['mapRenderState'] == 0:
        prev_frame_id = -1
      else:
        prev_frame_id = self.sm['mapRenderState'].frameId
      
      if starting_frame_id is None:
        starting_frame_id = prev_frame_id

      llk = gen_llk(self.location)
      self.pm.send("liveLocationKalman", llk)
      self.pm.wait_for_readers_to_update("liveLocationKalman", 10)
      self.sm.update(1000 if frame_expected else 0)
      assert self.sm.updated['mapRenderState'] == frame_expected, "renderer running at wrong frequency"

      if not frame_expected:
        continue

      frames_since_test_start = self.sm['mapRenderState'].frameId - starting_frame_id

      # give a few frames to go valid
      if frames_since_test_start < 5:
        continue
        
      # check output
      assert self.sm.valid['mapRenderState'] == expect_valid
      assert self.sm['mapRenderState'].frameId == (prev_frame_id + 1)
      assert self.sm['mapRenderState'].locationMonoTime == llk.logMonoTime
      if not expect_valid:
        assert self.sm['mapRenderState'].renderTime == 0.
      else:
        assert 0. < self.sm['mapRenderState'].renderTime < 0.1

      # check vision ipc output
      assert self.vipc.recv() is not None
      assert self.vipc.valid == expect_valid
      assert self.vipc.timestamp_sof == llk.logMonoTime
      assert self.vipc.frame_id == self.sm['mapRenderState'].frameId

  def disable_internet(self):
    MapBoxInternetDisabledRequestHandler.INTERNET_ACTIVE = False
    print("internet disabled")
  
  def enable_internet(self):
    MapBoxInternetDisabledRequestHandler.INTERNET_ACTIVE = True
    print("internet enabled")

  def test_with_internet(self):
    self.location = LOCATION1
    self.enable_internet()
    self._setup_test()
    self._run_test(True)

  def test_with_no_internet(self):
    self.location = LOCATION1
    self.disable_internet()
    self._setup_test()
    self._run_test(False)
    
  def test_recover_from_no_internet(self):
    self.location = LOCATION1
    self.enable_internet()
    self._setup_test()
    
    self._run_test(True)

    self.disable_internet()

    # change locations to force mapsd to refetch
    self.location = LOCATION2
    self._run_test(False)

    self.enable_internet()
    self._run_test(True)

    self.location = LOCATION1
    self._run_test(True)

if __name__ == "__main__":
  unittest.main()
