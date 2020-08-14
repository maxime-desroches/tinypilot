import sys
import termios
import time
from termios import (BRKINT, CS8, CSIZE, ECHO, ICANON, ICRNL, IEXTEN, INPCK,
                     ISIG, ISTRIP, IXON, PARENB, VMIN, VTIME)
from typing import Any

# Indexes for termios list.
IFLAG = 0
OFLAG = 1
CFLAG = 2
LFLAG = 3
ISPEED = 4
OSPEED = 5
CC = 6

def getch():
  fd = sys.stdin.fileno()
  old_settings = termios.tcgetattr(fd)
  try:
    # set
    mode = termios.tcgetattr(fd)
    mode[IFLAG] = mode[IFLAG] & ~(BRKINT | ICRNL | INPCK | ISTRIP | IXON)
    #mode[OFLAG] = mode[OFLAG] & ~(OPOST)
    mode[CFLAG] = mode[CFLAG] & ~(CSIZE | PARENB)
    mode[CFLAG] = mode[CFLAG] | CS8
    mode[LFLAG] = mode[LFLAG] & ~(ECHO | ICANON | IEXTEN | ISIG)
    mode[CC][VMIN] = 1
    mode[CC][VTIME] = 0
    termios.tcsetattr(fd, termios.TCSAFLUSH, mode)

    ch = sys.stdin.read(1)
  finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
  return ch

def keyboard_poll_thread(q):
  while True:
    c = getch()
    # print("got %s" % c)
    if c == '1':
      q.put(str("cruise_up"))
    if c == '2':
      q.put(str("cruise_down"))
    if c == '3':
      q.put(str("cruise_cancel"))
    if c == 'q':
      exit(0)
    if c=='\033':
      # print('arrow')
      c=getch()
      c=getch()
      if c=='A':
        # print('up')
        q.put(str("throttle_1"))
      if c=='B':
        # print('dowm')
        q.put(str("brake_1"))
      if c=='C':
        # print('right')
        q.put(str("steer_-1"))
      if c=='D':
        # print('left')
        q.put(str("steer_1"))


def test(q):
  while 1:
    print("hello")
    time.sleep(1.0)

if __name__ == '__main__':
  from multiprocessing import Process, Queue
  q : Any = Queue()
  p = Process(target=test, args=(q,))
  p.daemon = True
  p.start()

  keyboard_poll_thread(q)
