#!/usr/bin/env python3
import os
import sys
import time
from tqdm import tqdm

import cereal.messaging as messaging
from common.android import ANDROID
from tools.lib.framereader import FrameReader
from tools.lib.logreader import LogReader
from selfdrive.test.openpilotci import BASE_URL, get_url
from selfdrive.test.process_replay.compare_logs import compare_logs, save_log
from selfdrive.test.process_replay.test_processes import format_diff
from selfdrive.version import get_git_commit

if ANDROID:
  os.environ['QCOM_REPLAY'] = "1"
import selfdrive.manager as manager

TEST_ROUTE = "5b7c365c50084530|2020-04-15--16-13-24"

def camera_replay(lr, fr):

  pm = messaging.PubMaster(['frame', 'liveCalibration'])
  sm = messaging.SubMaster(['model'])

  # TODO: add dmonitoringmodeld
  print("preparing procs")
  manager.prepare_managed_process("camerad")
  manager.prepare_managed_process("modeld")
  try:
    print("starting procs")
    manager.start_managed_process("camerad")
    manager.start_managed_process("modeld")
    time.sleep(5)
    print("procs started")

    cal = [msg for msg in lr if msg.which() == "liveCalibration"]
    for msg in cal[:5]:
      pm.send(msg.which(), msg.as_builder())

    log_msgs = []
    frame_idx = 0
    for msg in tqdm(lr):
      if msg.which() == "liveCalibrationd":
        pm.send(msg.which(), msg.as_builder())
      elif msg.which() == "frame":
        f = msg.as_builder()
        img = fr.get(frame_idx, pix_fmt="rgb24")[0][:, ::, -1]
        f.frame.image = img.flatten().tobytes()
        frame_idx += 1

        pm.send(msg.which(), f)
        log_msgs.append(messaging.recv_one(sm.sock['model']))

        if frame_idx >= fr.frame_count:
          break
  except KeyboardInterrupt:
    pass

  print("replay done")
  manager.kill_managed_process('modeld')
  time.sleep(2)
  manager.kill_managed_process('camerad')
  return log_msgs


if __name__ == "__main__":

  update = "--update" in sys.argv

  lr = LogReader(get_url(TEST_ROUTE, 0))
  fr = FrameReader(get_url(TEST_ROUTE, 0, log_type="fcamera"))

  log_msgs = camera_replay(list(lr), fr)

  if update:
    ref_commit = get_git_commit()
    log_fn = "%s_%s_%s.bz2" % (TEST_ROUTE, "model", ref_commit)
    save_log(log_fn, log_msgs)
    with open("model_replay_ref_commit", "w") as f:
      f.write(ref_commit)
  else:
    ref_commit = open("model_replay_ref_commit").read().strip()
    log_fn = "%s_%s_%s.bz2" % (TEST_ROUTE, "model", ref_commit)
    cmp_log = LogReader(BASE_URL + log_fn)
    result = compare_logs(cmp_log, log_msgs, ignore_fields=['logMonoTime', 'valid'])
    diff1, diff2, failed = format_diff(result)

    print(diff1)
    sys.exit(int(failed))

