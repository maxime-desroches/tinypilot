#!/usr/bin/env python3

import unittest

from openpilot.selfdrive.car.values import PLATFORMS


class TestPlatformConfigs:
  def test_configs(subtests):

    for name, platform in PLATFORMS.items():
      with subtests.test(platform=str(platform)):
        assert platform.config._frozen

        if platform != "MOCK":
          assert "pt" in platform.config.dbc_dict
        assert len(platform.config.platform_str) > 0

        assert name == platform.config.platform_str

        assert platform.config.specs is not None


if __name__ == "__main__":
  unittest.main()
