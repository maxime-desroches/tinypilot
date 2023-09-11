
import platform
import sys
import unittest


def skip_if_process_replay_unsupported(func):
  if platform.system() == "Darwin":
    return unittest.skip("process_replay is not supported on macOS")(func)
  
  return func

def exit_if_process_replay_unsupported():
  if platform.system() == "Darwin":
    print(f"{sys.argv[0]} is not supported on macOS")
    sys.exit(1)
