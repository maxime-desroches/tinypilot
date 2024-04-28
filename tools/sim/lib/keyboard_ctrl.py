import sys
import termios
import time

from multiprocessing import Queue
from termios import (BRKINT, CS8, CSIZE, ECHO, ICANON, ICRNL, IEXTEN, INPCK,
                     ISTRIP, IXON, PARENB, VMIN, VTIME)
from typing import NoReturn

from openpilot.tools.sim.bridge.common import QueueMessage

# Indexes for termios list.
IFLAG = 0
OFLAG = 1
CFLAG = 2
LFLAG = 3
ISPEED = 4
OSPEED = 5
CC = 6


KEYBOARD_HELP = """
  | key  |   functionality       |
  |------|-----------------------|
  |  1   | Cruise Resume / Accel |
  |  2   | Cruise Set    / Decel |
  |  3   | Cruise Cancel         |
  |  r   | Reset Simulation      |
  |  i   | Toggle Ignition       |
  |  q   | Exit all              |
  | wasd | Control manually      |
"""


def getch() -> str:
  STDIN_FD = sys.stdin.fileno()
  old_settings = termios.tcgetattr(STDIN_FD)
  try:
    # set
    mode = old_settings.copy()
    mode[IFLAG] &= ~(BRKINT | ICRNL | INPCK | ISTRIP | IXON)
    #mode[OFLAG] &= ~(OPOST)
    mode[CFLAG] &= ~(CSIZE | PARENB)
    mode[CFLAG] |= CS8
    mode[LFLAG] &= ~(ECHO | ICANON | IEXTEN)
    mode[CC][VMIN] = 1
    mode[CC][VTIME] = 0
    termios.tcsetattr(STDIN_FD, termios.TCSAFLUSH, mode)

    ch = sys.stdin.read(1)
  finally:
    termios.tcsetattr(STDIN_FD, termios.TCSADRAIN, old_settings)
  return ch

def print_keyboard_help():
  print(f"Keyboard Commands:\n{KEYBOARD_HELP}")

def keyboard_poll_thread(q: Queue[QueueMessage]):
  print_keyboard_help()

  while True:
    c = getch()
    if c == '1':
      message = QueueMessage("control_command", "cruise_up")
    elif c == '2':
      message = QueueMessage("control_command", "cruise_down")
    elif c == '3':
      message = QueueMessage("control_command", "cruise_cancel")
    elif c == 'w':
      message = QueueMessage("control_command", f"throttle_{1.0}")
    elif c == 'a':
      message = QueueMessage("control_command", f"steer_{-0.15}")
    elif c == 's':
      message = QueueMessage("control_command", f"brake_{1.0}")
    elif c == 'd':
      message = QueueMessage("control_command", f"steer_{0.15}")
    elif c == 'z':
      message = QueueMessage("control_command", "blinker_left")
    elif c == 'x':
      message = QueueMessage("control_command", "blinker_right")
    elif c == 'i':
      message = QueueMessage("control_command", "ignition")
    elif c == 'r':
      message = QueueMessage("control_command", "reset")
    elif c == 'q':
      message = QueueMessage("control_command", "quit")
      q.put(message)
      break
    else:
      print_keyboard_help()
    q.put(message)

def test(q: 'Queue[str]') -> NoReturn:
  while True:
    print([q.get_nowait() for _ in range(q.qsize())] or None)
    time.sleep(0.25)

if __name__ == '__main__':
  from multiprocessing import Process, Queue
  q: Queue[QueueMessage] = Queue()
  p = Process(target=test, args=(q,))
  p.daemon = True
  p.start()

  keyboard_poll_thread(q)
