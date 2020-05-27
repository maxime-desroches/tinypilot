#!/usr/bin/env python3
import argparse
import bisect
import select
import sys
import termios
import time
import tty
from collections import defaultdict

import cereal.messaging as messaging

from tools.lib.framereader import FrameReader
from tools.lib.logreader import LogReader

IGNORE = ['initData', 'sentinel']


def input_ready():
  return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])


def replay(route, loop):
  route = route.replace('|', '/')

  lr = LogReader(f"cd:/{route}/rlog.bz2")
  fr = FrameReader(f"cd:/{route}/fcamera.hevc", readahead=True)

  # Build mapping from frameId to segmentId from encodeIdx, type == fullHEVC
  msgs = [m for m in lr if m.which() not in IGNORE]
  msgs = sorted(msgs, key=lambda m: m.logMonoTime)
  times = [m.logMonoTime for m in msgs]
  frame_idx = {m.encodeIdx.frameId: m.encodeIdx.segmentId for m in msgs if m.which() == 'encodeIdx' and m.encodeIdx.type == 'fullHEVC'}

  socks = {}
  lag = 0
  i = 0
  max_i = len(msgs) - 2

  while True:
    msg = msgs[i].as_builder()
    next_msg = msgs[i + 1]

    start_time = time.time()
    w = msg.which()

    if w == 'frame':
      try:
        img = fr.get(frame_idx[msg.frame.frameId], pix_fmt="rgb24")
        img = img[0][:, :, ::-1]  # Convert RGB to BGR, which is what the camera outputs
        msg.frame.image = img.flatten().tobytes()
      except (KeyError, ValueError):
        pass

    if w not in socks:
      socks[w] = messaging.pub_sock(w)

    try:
      if socks[w]:
        socks[w].send(msg.to_bytes())
    except messaging.messaging_pyx.MultiplePublishersError:
      socks[w] = None

    lag += (next_msg.logMonoTime - msg.logMonoTime) / 1e9
    lag -= time.time() - start_time

    dt = max(lag, 0)
    lag -= dt
    time.sleep(dt)

    if lag < -1.0 and i % 1000 == 0:
      print(f"{-lag:.2f} s behind")

    if input_ready():
      key = sys.stdin.read(1)

      # Handle pause
      if key == " ":
        while True:
          if input_ready() and sys.stdin.read(1) == " ":
            break
          time.sleep(0.01)

      # Handle seek
      dt = defaultdict(int, s=10, S=-10)[key]
      new_time = msgs[i].logMonoTime + dt * 1e9
      i = bisect.bisect_left(times, new_time)

    i = (i + 1) % max_i if loop else min(i + 1, max_i)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--loop", action='store_true')
  parser.add_argument("segment")
  args = parser.parse_args()

  orig_settings = termios.tcgetattr(sys.stdin)
  tty.setcbreak(sys.stdin)

  try:
    replay(args.segment, args.loop)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, orig_settings)
  except:
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, orig_settings)
    raise
