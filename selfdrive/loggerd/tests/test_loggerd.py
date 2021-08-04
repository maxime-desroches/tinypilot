#!/usr/bin/env python3
import os
import random
import string
import subprocess
import time
import unittest
from pathlib import Path

from cereal import log
from cereal.services import service_list
from common.basedir import BASEDIR
from common.params import Params
from common.timeout import Timeout
from selfdrive.hardware import PC, TICI
from selfdrive.loggerd.config import ROOT
from selfdrive.manager.process_config import managed_processes
from selfdrive.test.helpers import with_processes
from selfdrive.version import version as VERSION
from tools.lib.logreader import LogReader

SentinelType = log.Sentinel.SentinelType

CEREAL_SERVICES = [f for f in log.Event.schema.union_fields if f in service_list
                   and service_list[f].should_log and "encode" not in f.lower()]


class TestLoggerd(unittest.TestCase):
  # TODO: all tests should work on PC
  @classmethod
  def setUpClass(cls):
    if PC:
      raise unittest.SkipTest

  def _get_latest_log_dir(self):
    log_dirs = sorted(Path(ROOT).iterdir(), key=lambda f: f.stat().st_mtime)
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
      out = subprocess.check_output("./bootlog", cwd=os.path.join(BASEDIR, "selfdrive/loggerd"), encoding='utf-8')

    log_fn = self._get_log_fn(out)

    # check existence
    assert log_fn is not None

    return log_fn

  def _check_init_data(self, msgs):
    msg = msgs[0]
    self.assertEqual(msg.which(), 'initData')

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
    for k, _, v in fake_params:
      params.put(k, v)

    lr = list(LogReader(str(self._gen_bootlog())))
    initData = lr[0].initData

    self.assertTrue(initData.dirty != bool(os.environ["CLEAN"]))
    self.assertEqual(initData.version, VERSION)

    if os.path.isfile("/proc/cmdline"):
      with open("/proc/cmdline") as f:
        self.assertEqual(list(initData.kernelArgs), f.read().strip().split(" "))

      with open("/proc/version") as f:
        self.assertEqual(initData.kernelVersion, f.read())

    for _, k, v in fake_params:
      self.assertEqual(getattr(initData, k), v)

  # TODO: this shouldn't need camerad
  @with_processes(['camerad'])
  def test_rotation(self):
    os.environ["LOGGERD_TEST"] = "1"
    Params().put("RecordFront", "1")
    expected_files = {"rlog.bz2", "qlog.bz2", "qcamera.ts", "fcamera.hevc", "dcamera.hevc"}
    if TICI:
      expected_files.add("ecamera.hevc")

    # give camerad time to start
    time.sleep(5)

    for _ in range(5):
      num_segs = random.randint(1, 10)
      length = random.randint(2, 5)
      os.environ["LOGGERD_SEGMENT_LENGTH"] = str(length)

      managed_processes["loggerd"].start()
      time.sleep((num_segs + 1) * length)
      managed_processes["loggerd"].stop()

      route_path = str(self._get_latest_log_dir()).rsplit("--", 1)[0]
      for n in range(num_segs):
        p = Path(f"{route_path}--{n}")
        logged = set([f.name for f in p.iterdir() if f.is_file()])
        diff = logged ^ expected_files
        self.assertEqual(len(diff), 0, f"{_=} {route_path=} {n=}, {logged=} {expected_files=}")

  def test_bootlog(self):
    # generate bootlog with fake launch log
    launch_log = ''.join([str(random.choice(string.printable)) for _ in range(100)])
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
        expected_val = open(path, "rb").read()
        bootlog_val = [e.value for e in boot.pstore.entries if e.key == fn][0]
        self.assertEqual(expected_val, bootlog_val)

if __name__ == "__main__":
  unittest.main()
