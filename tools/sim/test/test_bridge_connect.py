#!/usr/bin/env python3
from multiprocessing import Queue, Process
import unittest
from tools.sim.bridge import connect_carla_client, CarlaBridge


class TestBridgeConnect(unittest.TestCase):
  """
  These tests need Carla simulator to run in parallel
  """
  BRIDGE_DURATION = 50

  def test_connect_with_carla(self):
    # Test connecting to Carla within 10 seconds and return no RuntimeError
    client = connect_carla_client(connect_timeout=10)
    assert client is not None
    # Will raise an error if not connected
    client.load_world('Town04_Opt')

  def test_run_bridge(self):
    # Test bridge connect with carla and runs without any errors for BRIDGE_DURATION seconds
    carla_bridge = CarlaBridge()

    p = Process(target=carla_bridge.bridge_keep_alive, args=(Queue(), self.BRIDGE_DURATION), daemon=True)
    p.start()

    p.join(self.BRIDGE_DURATION + 1)  # to ensure script terminates


if __name__ == "__main__":
  unittest.main()
