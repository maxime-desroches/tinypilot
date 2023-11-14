import os
import pytest
import threading
import time
import unittest
from unittest import mock

from cereal import log, messaging
from cereal.services import SERVICE_LIST
from openpilot.common.realtime import Ratekeeper
from openpilot.selfdrive.test.helpers import with_processes


NetworkType = log.DeviceState.NetworkType


def get_modem_data_usage():
  return int(time.time()), int(time.time())

def get_network_type():
  return NetworkType.cell4G


@pytest.mark.tici
class TestModem(unittest.TestCase):
  def setUp(self):
    self.pm = messaging.PubMaster(['pandaStates'])
    self.end_event = threading.Event()
    self.send_panda_state_thread = threading.Thread(target=self.send_panda_state)
    self.send_panda_state_thread.start()

    self.sm = messaging.SubMaster(['deviceState'])

  def tearDown(self):
    self.end_event.set()
    self.send_panda_state_thread.join()

  def _run_test(self, test_time=30, setup_time=2):
    for _ in range(int(setup_time * SERVICE_LIST['deviceState'].frequency)):
      self.sm.update()

    for _ in range(int(test_time * SERVICE_LIST['deviceState'].frequency)):
      self.sm.update()
      assert self.sm.all_checks()

      # ensure networkState is being updated at least every 10 seconds
      self.assertLess(time.time() - self.sm["deviceState"].networkStats.wwanTx, 12)

  def send_panda_state(self):
    rk = Ratekeeper(10)
    while not self.end_event.is_set():
      msg = messaging.new_message('pandaStates', 1)
      msg.pandaStates[0].pandaType = log.PandaState.PandaType.uno
      self.pm.send('pandaStates', msg)
      rk.keep_time()

  def restart_mm(self):
    os.system("sudo systemctl restart ModemManager.service")

  @mock.patch('openpilot.system.hardware.HARDWARE.get_network_type', get_network_type)
  @mock.patch('openpilot.system.hardware.HARDWARE.get_modem_data_usage', get_modem_data_usage)
  @with_processes(['thermald'])
  def test_mm_restart(self):
    self._run_test()

    self.restart_mm()

    self._run_test()


if __name__ == "__main__":
  unittest.main()