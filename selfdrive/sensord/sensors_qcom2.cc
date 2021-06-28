#include <sys/resource.h>

#include <chrono>
#include <thread>
#include <vector>

#include "cereal/messaging/messaging.h"
#include "selfdrive/common/i2c.h"
#include "selfdrive/common/swaglog.h"
#include "selfdrive/common/timing.h"
#include "selfdrive/common/util.h"
#include "selfdrive/sensord/sensors/bmx055_accel.h"
#include "selfdrive/sensord/sensors/bmx055_gyro.h"
#include "selfdrive/sensord/sensors/bmx055_magn.h"
#include "selfdrive/sensord/sensors/bmx055_temp.h"
#include "selfdrive/sensord/sensors/constants.h"
#include "selfdrive/sensord/sensors/light_sensor.h"
#include "selfdrive/sensord/sensors/lsm6ds3_accel.h"
#include "selfdrive/sensord/sensors/lsm6ds3_gyro.h"
#include "selfdrive/sensord/sensors/lsm6ds3_temp.h"
#include "selfdrive/sensord/sensors/sensor.h"

#define I2C_BUS_IMU 1

ExitHandler do_exit;

int sensor_loop() {
  I2CBus *i2c_bus_imu = nullptr;

  try {
    i2c_bus_imu = new I2CBus(I2C_BUS_IMU);
  } catch (std::exception &e) {
    LOGE("I2CBus init failed");
    return -1;
  }

  BMX055_Accel bmx055_accel(i2c_bus_imu);
  BMX055_Gyro bmx055_gyro(i2c_bus_imu);
  BMX055_Magn bmx055_magn(i2c_bus_imu);
  BMX055_Temp bmx055_temp(i2c_bus_imu);

  LSM6DS3_Accel lsm6ds3_accel(i2c_bus_imu);
  LSM6DS3_Gyro lsm6ds3_gyro(i2c_bus_imu);
  LSM6DS3_Temp lsm6ds3_temp(i2c_bus_imu);

  LightSensor light("/sys/class/i2c-adapter/i2c-2/2-0038/iio:device1/in_intensity_both_raw");

  // Sensor init
    std::vector<Sensor *> sensors;
  sensors.push_back(&bmx055_accel);
  sensors.push_back(&bmx055_gyro);
  sensors.push_back(&bmx055_magn);
  sensors.push_back(&bmx055_temp);

  sensors.push_back(&lsm6ds3_accel);
  sensors.push_back(&lsm6ds3_gyro);
  sensors.push_back(&lsm6ds3_temp);

  sensors.push_back(&light);

  for (Sensor * sensor : sensors) {
    int err = sensor->init();
    if (err < 0) {
      LOGE("Error initializing sensors");
      delete i2c_bus_imu;
      return -1;
    }
  }

  PubMaster pm({"sensorEvents"});

  while (!do_exit) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    MessageBuilder msg;
    auto orphanage = msg.getOrphanage();
    std::vector<capnp::Orphan<cereal::SensorEventData>> sensor_events;
    sensor_events.reserve(std::size(sensors));
    for (auto &sensor : sensors) {
      auto orphan = orphanage.newOrphan<cereal::SensorEventData>();
      auto event = orphan.get();
      if (sensor->get_event(event)) {
        sensor_events.push_back(kj::mv(orphan));
      }
    }
    auto events = msg.initEvent().initSensorEvents(sensor_events.size());
    for (int i = 0; i < sensor_events.size(); ++i) {
      events.adoptWithCaveats(i, kj::mv(sensor_events[i]));
    }

    pm.send("sensorEvents", msg);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(10) - (end - begin));
  }

  delete i2c_bus_imu;
  return 0;
}

int main(int argc, char *argv[]) {
  setpriority(PRIO_PROCESS, 0, -18);
  return sensor_loop();
}
