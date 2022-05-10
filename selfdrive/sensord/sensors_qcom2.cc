#include <sys/resource.h>

#include <chrono>
#include <thread>
#include <vector>
#include <poll.h>

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
#include "selfdrive/sensord/sensors/mmc5603nj_magn.h"
#include "selfdrive/sensord/sensors/sensor.h"

#define I2C_BUS_IMU 1

ExitHandler do_exit;
std::mutex pm_mutex;

void interrupt_loop(std::vector<Sensor *>& gpio_sensors, PubMaster& pm) {

  int cnt_sensors = gpio_sensors.size();
  std::unique_ptr<struct pollfd[]> fd_list(new struct pollfd[cnt_sensors]);

  for (int i = 0; i < cnt_sensors; ++i) {
    fd_list[i].fd = gpio_sensors[i]->gpio_fd;
    fd_list[i].events = POLLPRI;
  }

  while (!do_exit) {
    // events are received at 2kHz frequency, the bandwidth devider returns in
    // case of 125Hz 7 times the same data, this needs to be filtered, or hanled
    // smart in other way

    // TODO: check here also for epoll? might have a nicer interface
    int err = poll(fd_list.get(), cnt_sensors, -1); // no timeout
    if (err == -1) {
      return;
    }

    MessageBuilder msg;
    auto orphanage = msg.getOrphanage();
    std::vector<capnp::Orphan<cereal::SensorEventData>> collected_events;
    collected_events.reserve(cnt_sensors);

    for (int i = 0; i < cnt_sensors; ++i) {
      if ((fd_list[i].revents & POLLPRI) == 0) {
        continue;
      }

      auto orphan = orphanage.newOrphan<cereal::SensorEventData>();
      auto event = orphan.get();
      if (gpio_sensors[i]->get_event(event)) {
        // only send collected events
        collected_events.push_back(kj::mv(orphan));
      }
    }

    auto events = msg.initEvent().initSensorEvents(collected_events.size());
    for (int i = 0; i < collected_events.size(); ++i) {
      events.adoptWithCaveats(i, kj::mv(collected_events[i]));
    }

    pm_mutex.lock();
    pm.send("sensorEvents", msg);
    pm_mutex.unlock();
  }
}

int sensor_loop() {
  I2CBus *i2c_bus_imu;

  try {
    i2c_bus_imu = new I2CBus(I2C_BUS_IMU);
  } catch (std::exception &e) {
    LOGE("I2CBus init failed");
    return -1;
  }

  BMX055_Accel bmx055_accel(i2c_bus_imu, 21);
  BMX055_Gyro bmx055_gyro(i2c_bus_imu, 23);
  BMX055_Magn bmx055_magn(i2c_bus_imu);
  BMX055_Temp bmx055_temp(i2c_bus_imu);

  LSM6DS3_Accel lsm6ds3_accel(i2c_bus_imu, 84);
  LSM6DS3_Gyro lsm6ds3_gyro(i2c_bus_imu, 84);
  LSM6DS3_Temp lsm6ds3_temp(i2c_bus_imu);

  MMC5603NJ_Magn mmc5603nj_magn(i2c_bus_imu);

  LightSensor light("/sys/class/i2c-adapter/i2c-2/2-0038/iio:device1/in_intensity_both_raw");

  // Sensor init
  std::vector<std::pair<Sensor *, bool>> sensors_init; // Sensor, required
  sensors_init.push_back({&bmx055_accel, false});
  sensors_init.push_back({&bmx055_gyro, false});
  sensors_init.push_back({&bmx055_magn, false});
  sensors_init.push_back({&bmx055_temp, false}); // TODO: read with interrupt gyro

  sensors_init.push_back({&lsm6ds3_accel, true});
  sensors_init.push_back({&lsm6ds3_gyro, true});
  sensors_init.push_back({&lsm6ds3_temp, true}); // TODO: read with interrupt gyro

  sensors_init.push_back({&mmc5603nj_magn, false});

  sensors_init.push_back({&light, true});

  bool has_magnetometer = false;

  // Initialize sensors
  std::vector<Sensor *> sensors;
  std::vector<Sensor *> interrupt_sensors;
  for (auto &sensor : sensors_init) {
    int err = sensor.first->init();
    if (err < 0) {
      // Fail on required sensors
      if (sensor.second) {
        LOGE("Error initializing sensors");
        delete i2c_bus_imu;
        return -1;
      }
    } else {
      if (sensor.first == &bmx055_magn || sensor.first == &mmc5603nj_magn) {
        has_magnetometer = true;
      }

      // split between interrupt read sensors and non
      if (sensor.first->gpio_fd != -1) {
        interrupt_sensors.push_back(sensor.first);
      }
      else {
        sensors.push_back(sensor.first);
      }
    }
  }

  if (!has_magnetometer) {
    LOGE("No magnetometer present");
    delete i2c_bus_imu;
    return -1;
  }

  PubMaster pm({"sensorEvents"});

  // thread for reading events via interrupts
  std::thread interrupt_thread(&interrupt_loop, std::ref(interrupt_sensors), std::ref(pm));

  // polling loop for non interrupt handled sensors
  while (!do_exit) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    const int num_events = sensors.size();
    MessageBuilder msg;
    auto sensor_events = msg.initEvent().initSensorEvents(num_events);

    for (int i = 0; i < num_events; i++) {
      auto event = sensor_events[i];
      sensors[i]->get_event(event);
    }

    pm_mutex.lock();
    pm.send("sensorEvents", msg);
    pm_mutex.unlock();

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(10) - (end - begin));
  }

  interrupt_thread.join();
  delete i2c_bus_imu;
  return 0;
}

int main(int argc, char *argv[]) {
  setpriority(PRIO_PROCESS, 0, -18);
  return sensor_loop();
}
