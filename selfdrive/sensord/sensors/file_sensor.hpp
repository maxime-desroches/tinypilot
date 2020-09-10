#pragma once

#include <fstream>
#include <string>

#include "cereal/gen/cpp/log.capnp.h"
#include "sensors/sensor.hpp"


class FileSensor : public Sensor {
private:
  std::ifstream file;

public:
  FileSensor(std::string filename);
  ~FileSensor();
  int init();
  void get_event(cereal::SensorEventData::Builder &event);
};
