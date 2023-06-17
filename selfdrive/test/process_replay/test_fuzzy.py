#!/usr/bin/env python3
from hypothesis import given, HealthCheck, Phase, settings
import hypothesis.strategies as st
from parameterized import parameterized
import unittest

from cereal import log
from selfdrive.car.toyota.values import CAR as TOYOTA
from selfdrive.test.fuzzy_generation import get_random_event_msg
import selfdrive.test.process_replay.process_replay as pr

NOT_TESTED = ['controlsd', 'plannerd', 'calibrationd', 'dmonitoringd', 'paramsd', 'laikad']

class TestFuzzProcesses(unittest.TestCase):

  @parameterized.expand([(cfg.proc_name, cfg) for cfg in pr.CONFIGS if cfg.proc_name not in NOT_TESTED])
  @given(st.data())
  @settings(phases=[Phase.generate, Phase.target], deadline=1000, suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
  def test_fuzz_process(self, proc_name, cfg, data):
    msgs = data.draw(get_random_event_msg(required=cfg.pubs, real_floats=True))
    lr = [log.Event.new_message(**m).as_reader() for m in msgs]
    cfg.timeout = 5
    pr.replay_process(cfg, lr, TOYOTA.COROLLA_TSS2, disable_progress=True)

if __name__ == "__main__":
  unittest.main()
