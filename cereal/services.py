#!/usr/bin/env python3
import os
from typing import Optional

EON = os.path.isfile('/EON')
RESERVED_PORT = 8022  # sshd
STARTING_PORT = 8001


def new_port(port: int):
  port += STARTING_PORT
  return port + 1 if port >= RESERVED_PORT else port


class Service:
  def __init__(self, port: int, should_log: bool, frequency: float, decimation: Optional[int] = None):
    self.port = port
    self.should_log = should_log
    self.frequency = frequency
    self.decimation = decimation


services = {
  "roadCameraState": (True, 20., 1),  # should_log, frequency, decimation (optional)
  "sensorEvents": (True, 100., 100),
  "gpsNMEA": (True, 9.),
  "deviceState": (True, 2., 1),
  "can": (True, 100.),
  "controlsState": (True, 100., 100),
  "features": (True, 0.),
  "pandaState": (True, 2., 1),
  "radarState": (True, 20., 5),
  "roadEncodeIdx": (True, 20., 1),
  "liveTracks": (True, 20.),
  "sendcan": (True, 100.),
  "logMessage": (True, 0.),
  "liveCalibration": (True, 4., 4),
  "androidLog": (True, 0., 1),
  "carState": (True, 100., 10),
  "carControl": (True, 100., 10),
  "longitudinalPlan": (True, 20., 2),
  "liveLocation": (True, 0., 1),
  "procLog": (True, 0.5),
  "gpsLocationExternal": (True, 10., 1),
  "ubloxGnss": (True, 10.),
  "clocks": (True, 1., 1),
  "liveMpc": (False, 20.),
  "liveLongitudinalMpc": (False, 20.),
  "ubloxRaw": (True, 20.),
  "liveLocationKalman": (True, 20., 2),
  "uiLayoutState": (True, 0.),
  "liveParameters": (True, 20., 2),
  "cameraOdometry": (True, 20., 5),
  "lateralPlan": (True, 20., 2),
  "thumbnail": (True, 0.2, 1),
  "carEvents": (True, 1., 1),
  "carParams": (True, 0.02, 1),
  "driverCameraState": (True, 10. if EON else 20., 1),
  "driverEncodeIdx": (True, 10. if EON else 20., 1),
  "driverState": (True, 10. if EON else 20., 1),
  "driverMonitoringState": (True, 10. if EON else 20., 1),
  "offroadLayout": (False, 0.),
  "wideRoadEncodeIdx": (True, 20., 1),
  "wideRoadCameraState": (True, 20., 1),
  "modelV2": (True, 20., 20),
  "managerState": (True, 2., 1),

  "testModel": (False, 0.),
  "testLiveLocation": (False, 0.),
  "testJoystick": (False, 0.),
}
service_list = {name: Service(new_port(idx), *vals) for  # type: ignore
                idx, (name, vals) in enumerate(services.items())}


def build_header():
  h = ""
  h += "/* THIS IS AN AUTOGENERATED FILE, PLEASE EDIT services.py */\n"
  h += "#ifndef __SERVICES_H\n"
  h += "#define __SERVICES_H\n"
  h += "struct service { char name[0x100]; int port; bool should_log; int frequency; int decimation; };\n"
  h += "static struct service services[] = {\n"
  for k, v in service_list.items():
    should_log = "true" if v.should_log else "false"
    decimation = -1 if v.decimation is None else v.decimation
    h += '  { .name = "%s", .port = %d, .should_log = %s, .frequency = %d, .decimation = %d },\n' % \
         (k, v.port, should_log, v.frequency, decimation)
  h += "};\n"
  h += "#endif\n"
  return h


if __name__ == "__main__":
  print(build_header())
