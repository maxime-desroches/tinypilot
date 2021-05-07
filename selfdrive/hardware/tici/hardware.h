#pragma once

#include <cstdlib>
#include <fstream>

#include "selfdrive/common/params.h"
#include "selfdrive/common/util.h"
#include "selfdrive/hardware/base.h"

class Hardware : public HardwareBase {
 public:
  Hardware() {
    MAX_VOLUME = 0.5;
    MIN_VOLUME = 0.4;

    road_cam_focal_len = 2648;
    wide_cam_focal_len = driver_cam_focal_len = 860;
    
    road_cam_size[0] = 1928;
    road_cam_size[1] = 1208;

    driver_cam_size[0] = 1928;
    driver_cam_size[1] = 1208;

    wide_road_cam_size[0] = 1928;
    wide_road_cam_size[1] = 1208;

    screen_size[0] = 2160;
    screen_size[1] = 1080;
  }

  bool TICI() const override { return true; }
  std::string get_os_version() const override {
    return "AGNOS " + util::read_file("/VERSION");
  };

  void reboot() const override { std::system("sudo reboot"); }
  void poweroff() const override { std::system("sudo poweroff"); }
  void set_brightness(int percent) const override {
    std::ofstream brightness_control("/sys/class/backlight/panel0-backlight/brightness");
    if (brightness_control.is_open()) override {
      brightness_control << (percent * (int)(1023/100.)) << "\n";
      brightness_control.close();
    }
  }
  void set_display_power(bool on) const override {}

  bool get_ssh_enabled() const override { return Params().getBool("SshEnabled"); }
  void set_ssh_enabled(bool enabled) const override { Params().putBool("SshEnabled", enabled); }
};
