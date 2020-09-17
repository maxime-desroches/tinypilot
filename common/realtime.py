"""Utilities for reading real time clocks and keeping soft real time constraints."""
import gc
import os
import time
import multiprocessing

from common.hardware import PC
from common.common_pyx import sec_since_boot  # pylint: disable=no-name-in-module, import-error


# time step for each process
DT_CTRL = 0.01  # controlsd
DT_MDL = 0.05  # model
DT_DMON = 0.1  # driver monitoring
DT_TRML = 0.5  # thermald and manager


class Priority:
  MIN_REALTIME = 52 # highest android process priority is 51
  CTRL_LOW = MIN_REALTIME
  CTRL_HIGH = MIN_REALTIME + 1


def set_realtime_priority(level):
  if not PC:
    os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(level))


def set_core_affinity(core):
  if not PC:
    os.sched_setaffinity(0, [core,])


def config_rt_process(core, priority):
  gc.disable()
  set_realtime_priority(priority)
  set_core_affinity(core)


class Ratekeeper():
  def __init__(self, rate, print_delay_threshold=0.):
    """Rate in Hz for ratekeeping. print_delay_threshold must be nonnegative."""
    self._interval = 1. / rate
    self._next_frame_time = sec_since_boot() + self._interval
    self._print_delay_threshold = print_delay_threshold
    self._frame = 0
    self._remaining = 0
    self._process_name = multiprocessing.current_process().name

  @property
  def frame(self):
    return self._frame

  @property
  def remaining(self):
    return self._remaining

  # Maintain loop rate by calling this at the end of each loop
  def keep_time(self):
    lagged = self.monitor_time()
    if self._remaining > 0:
      time.sleep(self._remaining)
    return lagged

  # this only monitor the cumulative lag, but does not enforce a rate
  def monitor_time(self):
    lagged = False
    remaining = self._next_frame_time - sec_since_boot()
    self._next_frame_time += self._interval
    if self._print_delay_threshold is not None and remaining < -self._print_delay_threshold:
      print("%s lagging by %.2f ms" % (self._process_name, -remaining * 1000))
      lagged = True
    self._frame += 1
    self._remaining = remaining
    return lagged
