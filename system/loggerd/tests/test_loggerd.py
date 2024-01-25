#!/usr/bin/env python3
import numpy as np
import os
import random
import string
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import cereal.messaging as messaging
from cereal import log
from cereal.services import SERVICE_LIST
from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params
from openpilot.common.timeout import Timeout
from openpilot.system.hardware.hw import Paths
from openpilot.system.loggerd.xattr_cache import getxattr
from openpilot.system.loggerd.deleter import PRESERVE_ATTR_NAME, PRESERVE_ATTR_VALUE
from openpilot.selfdrive.manager.process_config import managed_processes
from openpilot.system.version import get_version
from openpilot.tools.lib.logreader import LogReader
from cereal.visionipc import VisionIpcServer, VisionStreamType
from openpilot.common.transformations.camera import tici_f_frame_size, tici_d_frame_size, tici_e_frame_size

SentinelType = log.Sentinel.SentinelType

CEREAL_SERVICES = [f for f in log.Event.schema.union_fields if f in SERVICE_LIST
                   and SERVICE_LIST[f].should_log and "encode" not in f.lower()]


class TestLoggerd:
  def _get_latest_log_dir(self):
    log_dirs = sorted(Path(Paths.log_root()).iterdir(), key=lambda f: f.stat().st_mtime)
    return log_dirs[-1]

  def _get_log_dir(self, x):
    for l in x.splitlines():
      for p in l.split(' '):
        path = Path(p.strip())
        if path.is_dir():
          return path
    return None

  def _get_log_fn(self, x):
    for l in x.splitlines():
      for p in l.split(' '):
        path = Path(p.strip())
        if path.is_file():
          return path
    return None

  def _gen_bootlog(self):
    with Timeout(5):
      out = subprocess.check_output("./bootlog", cwd=os.path.join(BASEDIR, "system/loggerd"), encoding='utf-8')

    log_fn = self._get_log_fn(out)

    # check existence
    assert log_fn is not None

    return log_fn

  def _check_init_data(self, msgs):
    msg = msgs[0]
    assert msg.which() == 'initData'

  def _check_sentinel(self, msgs, route):
    start_type = SentinelType.startOfRoute if route else SentinelType.startOfSegment
    assert msgs[1].sentinel.type == start_type

    end_type = SentinelType.endOfRoute if route else SentinelType.endOfSegment
    assert msgs[-1].sentinel.type == end_type

  def _publish_random_messages(self, services: List[str]) -> Dict[str, list]:
    pm = messaging.PubMaster(services)

    managed_processes["loggerd"].start()
    for s in services:
      assert pm.wait_for_readers_to_update(s, timeout=5)

    sent_msgs = defaultdict(list)
    for _ in range(random.randint(2, 10) * 100):
      for s in services:
        try:
          m = messaging.new_message(s)
        except Exception:
          m = messaging.new_message(s, random.randint(2, 10))
        pm.send(s, m)
        sent_msgs[s].append(m)

    for s in services:
      assert pm.wait_for_readers_to_update(s, timeout=5)
    managed_processes["loggerd"].stop()

    return sent_msgs

  def test_init_data_values(self):
    os.environ["CLEAN"] = random.choice(["0", "1"])

    dongle  = ''.join(random.choice(string.printable) for n in range(random.randint(1, 100)))
    fake_params = [
      # param, initData field, value
      ("DongleId", "dongleId", dongle),
      ("GitCommit", "gitCommit", "commit"),
      ("GitBranch", "gitBranch", "branch"),
      ("GitRemote", "gitRemote", "remote"),
    ]
    params = Params()
    params.clear_all()
    for k, _, v in fake_params:
      params.put(k, v)
    params.put("AccessToken", "abc")

    lr = list(LogReader(str(self._gen_bootlog())))
    initData = lr[0].initData

    assert initData.dirty != bool(os.environ["CLEAN"])
    assert initData.version == get_version()

    if os.path.isfile("/proc/cmdline"):
      with open("/proc/cmdline") as f:
        assert list(initData.kernelArgs) == f.read().strip().split(" ")

      with open("/proc/version") as f:
        assert initData.kernelVersion == f.read()

    # check params
    logged_params = {entry.key: entry.value for entry in initData.params.entries}
    expected_params = {k for k, _, __ in fake_params} | {'AccessToken'}
    assert set(logged_params.keys()) == expected_params, set(logged_params.keys()) ^ expected_params
    assert logged_params['AccessToken'] == b'', f"DONT_LOG param value was logged: {repr(logged_params['AccessToken'])}"
    for param_key, initData_key, v in fake_params:
      assert getattr(initData, initData_key) == v
      assert logged_params[param_key].decode() == v

    params.put("AccessToken", "")

  def test_rotation(self):
    os.environ["LOGGERD_TEST"] = "1"
    Params().put("RecordFront", "1")

    expected_files = {"rlog", "qlog", "qcamera.ts", "fcamera.hevc", "dcamera.hevc", "ecamera.hevc"}
    streams = [(VisionStreamType.VISION_STREAM_ROAD, (*tici_f_frame_size, 2048*2346, 2048, 2048*1216), "roadCameraState"),
               (VisionStreamType.VISION_STREAM_DRIVER, (*tici_d_frame_size, 2048*2346, 2048, 2048*1216), "driverCameraState"),
               (VisionStreamType.VISION_STREAM_WIDE_ROAD, (*tici_e_frame_size, 2048*2346, 2048, 2048*1216), "wideRoadCameraState")]

    pm = messaging.PubMaster(["roadCameraState", "driverCameraState", "wideRoadCameraState"])
    vipc_server = VisionIpcServer("camerad")
    for stream_type, frame_spec, _ in streams:
      vipc_server.create_buffers_with_sizes(stream_type, 40, False, *(frame_spec))
    vipc_server.start_listener()

    num_segs = 3
    length = 2
    os.environ["LOGGERD_SEGMENT_LENGTH"] = str(length)
    managed_processes["loggerd"].start()
    managed_processes["encoderd"].start()
    assert pm.wait_for_readers_to_update("roadCameraState", timeout=5)

    fps = 20.0
    for n in range(1, int(num_segs*length*fps)+1):
      for stream_type, frame_spec, state in streams:
        dat = np.empty(frame_spec[2], dtype=np.uint8)
        vipc_server.send(stream_type, dat[:].flatten().tobytes(), n, n/fps, n/fps)

        camera_state = messaging.new_message(state)
        frame = getattr(camera_state, state)
        frame.frameId = n
        pm.send(state, camera_state)

      for _, _, state in streams:
        assert pm.wait_for_readers_to_update(state, timeout=5, dt=0.001)

    managed_processes["loggerd"].stop()
    managed_processes["encoderd"].stop()

    route_path = str(self._get_latest_log_dir()).rsplit("--", 1)[0]
    for n in range(num_segs):
      p = Path(f"{route_path}--{n}")
      logged = {f.name for f in p.iterdir() if f.is_file()}
      diff = logged ^ expected_files
      assert len(diff) == 0, f"didn't get all expected files. run={_} seg={n} {route_path=}, {diff=}\n{logged=} {expected_files=}"

  def test_bootlog(self):
    # generate bootlog with fake launch log
    launch_log = ''.join(str(random.choice(string.printable)) for _ in range(100))
    with open("/tmp/launch_log", "w") as f:
      f.write(launch_log)

    bootlog_path = self._gen_bootlog()
    lr = list(LogReader(str(bootlog_path)))

    # check length
    assert len(lr) == 2  # boot + initData

    self._check_init_data(lr)

    # check msgs
    bootlog_msgs = [m for m in lr if m.which() == 'boot']
    assert len(bootlog_msgs) == 1

    # sanity check values
    boot = bootlog_msgs.pop().boot
    assert abs(boot.wallTimeNanos - time.time_ns()) < 5*1e9 # within 5s
    assert boot.launchLog == launch_log

    for fn in ["console-ramoops", "pmsg-ramoops-0"]:
      path = Path(os.path.join("/sys/fs/pstore/", fn))
      if path.is_file():
        with open(path, "rb") as f:
          expected_val = f.read()
        bootlog_val = [e.value for e in boot.pstore.entries if e.key == fn][0]
        assert expected_val == bootlog_val

  def test_qlog(self):
    qlog_services = [s for s in CEREAL_SERVICES if SERVICE_LIST[s].decimation is not None]
    no_qlog_services = [s for s in CEREAL_SERVICES if SERVICE_LIST[s].decimation is None]

    services = random.sample(qlog_services, random.randint(2, min(10, len(qlog_services)))) + \
               random.sample(no_qlog_services, random.randint(2, min(10, len(no_qlog_services))))
    sent_msgs = self._publish_random_messages(services)

    qlog_path = os.path.join(self._get_latest_log_dir(), "qlog")
    lr = list(LogReader(qlog_path))

    # check initData and sentinel
    self._check_init_data(lr)
    self._check_sentinel(lr, True)

    recv_msgs = defaultdict(list)
    for m in lr:
      recv_msgs[m.which()].append(m)

    for s, msgs in sent_msgs.items():
      recv_cnt = len(recv_msgs[s])

      if s in no_qlog_services:
        # check services with no specific decimation aren't in qlog
        assert recv_cnt == 0, f"got {recv_cnt} {s} msgs in qlog"
      else:
        # check logged message count matches decimation
        expected_cnt = (len(msgs) - 1) // SERVICE_LIST[s].decimation + 1
        assert recv_cnt == expected_cnt, f"expected {expected_cnt} msgs for {s}, got {recv_cnt}"

  def test_rlog(self):
    services = random.sample(CEREAL_SERVICES, random.randint(5, 10))
    sent_msgs = self._publish_random_messages(services)

    lr = list(LogReader(os.path.join(self._get_latest_log_dir(), "rlog")))

    # check initData and sentinel
    self._check_init_data(lr)
    self._check_sentinel(lr, True)

    # check all messages were logged and in order
    lr = lr[2:-1] # slice off initData and both sentinels
    for m in lr:
      sent = sent_msgs[m.which()].pop(0)
      sent.clear_write_flag()
      assert sent.to_bytes() == m.as_builder().to_bytes()

  def test_preserving_flagged_segments(self):
    services = set(random.sample(CEREAL_SERVICES, random.randint(5, 10))) | {"userFlag"}
    self._publish_random_messages(services)

    segment_dir = self._get_latest_log_dir()
    assert getxattr(segment_dir, PRESERVE_ATTR_NAME) == PRESERVE_ATTR_VALUE

  def test_not_preserving_unflagged_segments(self):
    services = set(random.sample(CEREAL_SERVICES, random.randint(5, 10))) - {"userFlag"}
    self._publish_random_messages(services)

    segment_dir = self._get_latest_log_dir()
    assert getxattr(segment_dir, PRESERVE_ATTR_NAME) is None

