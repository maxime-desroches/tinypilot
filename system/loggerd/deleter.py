#!/usr/bin/env python3
import os
import shutil
import threading
from openpilot.system.hardware.hw import Paths
from openpilot.common.swaglog import cloudlog
from openpilot.system.loggerd.config import get_available_bytes, get_available_percent
from openpilot.system.loggerd.uploader import listdir_by_creation
from openpilot.system.loggerd.xattr_cache import getxattr

MIN_BYTES = 5 * 1024 * 1024 * 1024
MIN_PERCENT = 10

DELETE_LAST = ['boot', 'crash']

PRESERVE_ATTR_NAME = 'user.preserve'
PRESERVE_ATTR_VALUE = b'1'
PRESERVE_COUNT = 5


def has_preserve_xattr(d: str) -> bool:
  return getxattr(os.path.join(Paths.log_root(), d), PRESERVE_ATTR_NAME) == PRESERVE_ATTR_VALUE


def get_preserved_segments(dirs_by_creation: list[str]) -> list[str]:
  preserved = []
  for n, d in enumerate(filter(has_preserve_xattr, reversed(dirs_by_creation))):
    if n == PRESERVE_COUNT:
      break
    date_str, _, seg_str = d.rpartition("--")

    # ignore non-segment directories
    if not date_str:
      continue
    try:
      seg_num = int(seg_str)
    except ValueError:
      continue

    # preserve segment and its prior
    preserved.append(d)
    preserved.append(f"{date_str}--{seg_num - 1}")

  return preserved

def is_locked(file_path):
  return os.path.exists(f'{file_path}.lock')

def delete(path: str) -> bool:
  if not os.path.exists(path):
    return False

  if os.path.isfile(path) and not is_locked(path):
    cloudlog.info(f"deleting file: {path}")
    os.remove(path)
    return True
  elif os.path.isdir(path):
    for f in filter(lambda n: not n.endswith('.lock'), os.listdir(path)):
      delete(os.path.join(path, f))
      if len(os.listdir(path)) == 0:
        cloudlog.info(f"deleting directory: {path}")
        shutil.rmtree(path)
        return True
  return False

def deleter_thread(exit_event):
  while not exit_event.is_set():
    out_of_bytes = get_available_bytes(default=MIN_BYTES + 1) < MIN_BYTES
    out_of_percent = get_available_percent(default=MIN_PERCENT + 1) < MIN_PERCENT

    if out_of_percent or out_of_bytes:
      log_root = Paths.log_root()

      dirs = listdir_by_creation(log_root)
      files = [f for f in os.listdir(log_root) if os.path.isfile(os.path.join(log_root, f))]

      # skip deleting most recent N preserved segments (and their prior segment)
      preserved_dirs = get_preserved_segments(dirs)

      # remove the earliest directory we can, lastly check for files in log_root()
      for f in sorted(files + dirs, key=lambda d: (d in DELETE_LAST, d in preserved_dirs)):
        to_delete = os.path.join(log_root, f)
        try:
          if delete(to_delete):
            break
        except OSError:
          cloudlog.exception(f"issue deleting {to_delete}")
      exit_event.wait(.1)
    else:
      exit_event.wait(30)


def main():
  deleter_thread(threading.Event())


if __name__ == "__main__":
  main()
