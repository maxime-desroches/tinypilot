#pragma once

#include <eigen3/Eigen/Dense>
#include <fstream>
#include <memory>
#include <map>
#include <string>

#include "cereal/messaging/messaging.h"
#include "common/transformations/coordinates.hpp"
#include "common/transformations/orientation.hpp"
#include "common/params.h"
#include "common/swaglog.h"
#include "common/timing.h"
#include "common/util.h"

#include "selfdrive/sensord/sensors/constants.h"
#define VISION_DECIMATION 2
#define SENSOR_DECIMATION 10
#include "selfdrive/locationd/models/live_kf.h"

#define POSENET_STD_HIST_HALF 20

const int LLD_FREQ = 20; // Hz

class Localizer {
public:
  Localizer();
  Localizer(bool has_ublox);

  int locationd_thread();

  typedef enum MsgHandlerStatus {
    MSG_IGNORE,
    MSG_TIME_INVALID,
    MSG_VALUE_INVALID,
    MSG_VALID
  } MsgHandlerStatus;

  void reset_kalman(double current_time = NAN);
  void reset_kalman(double current_time, Eigen::VectorXd init_orient, Eigen::VectorXd init_pos, Eigen::VectorXd init_vel, MatrixXdr init_pos_R, MatrixXdr init_vel_R);
  void reset_kalman(double current_time, Eigen::VectorXd init_x, MatrixXdr init_P);
  void finite_check(double current_time = NAN);
  void time_check(double current_time = NAN);
  void update_reset_tracker();
  bool is_gps_ok();
  bool is_timestamp_valid(double current_time);
  void determine_gps_mode(double current_time);
  bool are_inputs_ok();
  void observation_timings_invalid_reset();
  void update_critical_services();
  void initialize_msg_validity_filters();

  kj::ArrayPtr<capnp::byte> get_message_bytes(MessageBuilder& msg_builder,
    bool inputsOK, bool sensorsOK, bool gpsOK, bool msgValid);
  void build_live_location(cereal::LiveLocationKalman::Builder& fix);

  Eigen::VectorXd get_position_geodetic();
  Eigen::VectorXd get_state();
  Eigen::VectorXd get_stdev();

  void handle_msg_bytes(const char* service, const char *data, const size_t size);
  void handle_msg(const char* service, const cereal::Event::Reader& log);
  MsgHandlerStatus handle_sensor(double current_time, const cereal::SensorEventData::Reader& log);
  MsgHandlerStatus handle_gps(double current_time, const cereal::GpsLocationData::Reader& log, const double sensor_time_offset);
  MsgHandlerStatus handle_gnss(double current_time, const cereal::GnssMeasurements::Reader& log);
  MsgHandlerStatus handle_car_state(double current_time, const cereal::CarState::Reader& log);
  MsgHandlerStatus handle_cam_odo(double current_time, const cereal::CameraOdometry::Reader& log);
  MsgHandlerStatus handle_live_calib(double current_time, const cereal::LiveCalibrationData::Reader& log);

  void input_fake_gps_observations(double current_time);

private:
  std::unique_ptr<LiveKalman> kf;

  Eigen::VectorXd calib;
  MatrixXdr device_from_calib;
  MatrixXdr calib_from_device;
  bool calibrated = false;

  double car_speed = 0.0;
  double last_reset_time = NAN;
  std::deque<double> posenet_stds;

  std::unique_ptr<LocalCoord> converter;

  int64_t unix_timestamp_millis = 0;
  double reset_tracker = 0.0;
  bool device_fell = false;
  bool gps_mode = false;
  double last_gps_msg = 0;
  bool ublox_available = true;
  std::map<std::string, bool> observation_timings_invalid;
  std::map<std::string, FirstOrderFilter*> observation_values_invalid;
  bool standstill = true;
  int32_t orientation_reset_count = 0;
  std::vector<std::string> critical_input_services;
};
