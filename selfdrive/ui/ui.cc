#include "selfdrive/ui/ui.h"

#include <cassert>
#include <cmath>

#include <QtConcurrent>

#include "common/transformations/orientation.hpp"
#include "common/params.h"
#include "common/swaglog.h"
#include "common/util.h"
#include "common/watchdog.h"
#include "system/hardware/hw.h"

#define BACKLIGHT_DT 0.05
#define BACKLIGHT_TS 10.00
#define BACKLIGHT_OFFROAD 50

// Projects a point in car to space to the corresponding point in full frame
// image space.
static bool calib_frame_to_full_frame(const UIState *s, float in_x, float in_y, float in_z, QPointF *out) {
  const float margin = 500.0f;
  const QRectF clip_region{-margin, -margin, s->fb_w + 2 * margin, s->fb_h + 2 * margin};

  const vec3 pt = (vec3){{in_x, in_y, in_z}};
  const vec3 Ep = matvecmul3(s->scene.wide_cam ? s->scene.view_from_wide_calib : s->scene.view_from_calib, pt);
  const vec3 KEp = matvecmul3(s->scene.wide_cam ? ecam_intrinsic_matrix : fcam_intrinsic_matrix, Ep);

  // Project.
  QPointF point = s->car_space_transform.map(QPointF{KEp.v[0] / KEp.v[2], KEp.v[1] / KEp.v[2]});
  if (clip_region.contains(point)) {
    *out = point;
    return true;
  }
  return false;
}

int get_path_length_idx(const cereal::ModelDataV2::XYZTData::Reader &line, const float path_height) {
  const auto line_x = line.getX();
  int max_idx = 0;
  for (int i = 1; i < TRAJECTORY_SIZE && line_x[i] <= path_height; ++i) {
    max_idx = i;
  }
  return max_idx;
}

void update_leads(UIState *s, const cereal::RadarState::Reader &radar_state, const cereal::ModelDataV2::XYZTData::Reader &line) {
  for (int i = 0; i < 2; ++i) {
    auto lead_data = (i == 0) ? radar_state.getLeadOne() : radar_state.getLeadTwo();
    if (lead_data.getStatus()) {
      float z = line.getZ()[get_path_length_idx(line, lead_data.getDRel())];
      calib_frame_to_full_frame(s, lead_data.getDRel(), -lead_data.getYRel(), z + 1.22, &s->scene.lead_vertices[i]);
    }
  }
}

void update_line_data(const UIState *s, const cereal::ModelDataV2::XYZTData::Reader &line,
                      float y_off, float z_off, QPolygonF *pvd, int max_idx, bool allow_invert=true) {
  const auto line_x = line.getX(), line_y = line.getY(), line_z = line.getZ();
  QPolygonF left_points, right_points;
  left_points.reserve(max_idx + 1);
  right_points.reserve(max_idx + 1);

  for (int i = 0; i <= max_idx; i++) {
    // highly negative x positions  are drawn above the frame and cause flickering, clip to zy plane of camera
    if (line_x[i] < 0) continue;
    QPointF left, right;
    bool l = calib_frame_to_full_frame(s, line_x[i], line_y[i] - y_off, line_z[i] + z_off, &left);
    bool r = calib_frame_to_full_frame(s, line_x[i], line_y[i] + y_off, line_z[i] + z_off, &right);
    if (l && r) {
      // For wider lines the drawn polygon will "invert" when going over a hill and cause artifacts
      if (!allow_invert && left_points.size() && left.y() > left_points.back().y()) {
        continue;
      }
      left_points.push_back(left);
      right_points.push_front(right);
    }
  }
  *pvd = left_points + right_points;
}

void update_model(UIState *s, const cereal::ModelDataV2::Reader &model) {
  UIScene &scene = s->scene;
  auto model_position = model.getPosition();
  float max_distance = std::clamp(model_position.getX()[TRAJECTORY_SIZE - 1],
                                  MIN_DRAW_DISTANCE, MAX_DRAW_DISTANCE);

  // update lane lines
  const auto lane_lines = model.getLaneLines();
  const auto lane_line_probs = model.getLaneLineProbs();
  int max_idx = get_path_length_idx(lane_lines[0], max_distance);
  for (int i = 0; i < std::size(scene.lane_line_vertices); i++) {
    scene.lane_line_probs[i] = lane_line_probs[i];
    update_line_data(s, lane_lines[i], 0.025 * scene.lane_line_probs[i], 0, &scene.lane_line_vertices[i], max_idx);
  }

  // update road edges
  const auto road_edges = model.getRoadEdges();
  const auto road_edge_stds = model.getRoadEdgeStds();
  for (int i = 0; i < std::size(scene.road_edge_vertices); i++) {
    scene.road_edge_stds[i] = road_edge_stds[i];
    update_line_data(s, road_edges[i], 0.025, 0, &scene.road_edge_vertices[i], max_idx);
  }

  // update path
  auto lead_one = (*s->sm)["radarState"].getRadarState().getLeadOne();
  if (lead_one.getStatus()) {
    const float lead_d = lead_one.getDRel() * 2.;
    max_distance = std::clamp((float)(lead_d - fmin(lead_d * 0.35, 10.)), 0.0f, max_distance);
  }
  max_idx = get_path_length_idx(model_position, max_distance);
  update_line_data(s, model_position, 0.9, 1.22, &scene.track_vertices, max_idx, false);

  // update driver
  const auto driver_orient = (*s->sm)["driverStateV2"].getDriverStateV2().getLeftDriverData().getFaceOrientation();
  float p_this = driver_orient[0];
  float y_this = driver_orient[1];
  float r_this = driver_orient[2];
  p_this = 0.33 * p_this + 0.66 * scene.driver_pose_pitch;
  y_this = 0.33 * y_this + 0.66 * scene.driver_pose_yaw;
  r_this = 0.33 * r_this + 0.66 * scene.driver_pose_roll;
  scene.driver_pose_pitch = p_this;
  scene.driver_pose_yaw = y_this;
  scene.driver_pose_roll = r_this;

  const mat3 rx = (mat3){{
    1, 0, 0,
    0, cosf(scene.driver_pose_pitch), -sinf(scene.driver_pose_pitch),
    0, sinf(scene.driver_pose_pitch), cosf(scene.driver_pose_pitch),
  }};

  const mat3 ry = (mat3){{
    cosf(-scene.driver_pose_yaw), 0, sinf(-scene.driver_pose_yaw),
    0, 1, 0,
    -sinf(-scene.driver_pose_yaw), 0, cosf(-scene.driver_pose_yaw),
  }};

  const mat3 rz = (mat3){{
    cosf(-scene.driver_pose_roll), -sinf(-scene.driver_pose_roll), 0,
    sinf(-scene.driver_pose_roll), cosf(-scene.driver_pose_roll), 0,
    0, 0, 1,
  }};
  // populate vertices
  int face_vertices_idx = 0;
  for (int kpi = 0; kpi < std::size(default_face_kpts_3d); kpi++) {
    bool in_end_idxs = false;
    for (int ei = face_vertices_idx; ei < std::size(face_end_idxs); ei++) {
      if (kpi == face_end_idxs[ei]) {
        in_end_idxs = true;
        break;
      }
    }
    if (in_end_idxs) {
      face_vertices_idx +=1;
      continue;
    }
    vec3 kpt_this = default_face_kpts_3d[kpi];
    vec3 kpt_next = default_face_kpts_3d[kpi+1];
    kpt_this = matvecmul3(rz, matvecmul3(ry, matvecmul3(rx, kpt_this)));
    kpt_next = matvecmul3(rz, matvecmul3(ry, matvecmul3(rx, kpt_next)));
    scene.face_kpt_segments[kpi-face_vertices_idx] = QLineF(kpt_this.v[0]+115,
                                                                                                kpt_this.v[1]+900,
                                                                                                kpt_next.v[0]+115,
                                                                                                kpt_next.v[1]+900);
  }
}

static void update_sockets(UIState *s) {
  s->sm->update(0);
}

static void update_state(UIState *s) {
  SubMaster &sm = *(s->sm);
  UIScene &scene = s->scene;

  if (sm.updated("liveCalibration")) {
    auto rpy_list = sm["liveCalibration"].getLiveCalibration().getRpyCalib();
    auto wfde_list = sm["liveCalibration"].getLiveCalibration().getWideFromDeviceEuler();
    Eigen::Vector3d rpy;
    Eigen::Vector3d wfde;
    if (rpy_list.size() == 3) rpy << rpy_list[0], rpy_list[1], rpy_list[2];
    if (wfde_list.size() == 3) wfde << wfde_list[0], wfde_list[1], wfde_list[2];
    Eigen::Matrix3d device_from_calib = euler2rot(rpy);
    Eigen::Matrix3d wide_from_device = euler2rot(wfde);
    Eigen::Matrix3d view_from_device;
    view_from_device << 0,1,0,
                        0,0,1,
                        1,0,0;
    Eigen::Matrix3d view_from_calib = view_from_device * device_from_calib;
    Eigen::Matrix3d view_from_wide_calib = view_from_device * wide_from_device * device_from_calib ;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        scene.view_from_calib.v[i*3 + j] = view_from_calib(i,j);
        scene.view_from_wide_calib.v[i*3 + j] = view_from_wide_calib(i,j);
      }
    }
    scene.calibration_valid = sm["liveCalibration"].getLiveCalibration().getCalStatus() == 1;
    scene.calibration_wide_valid = wfde_list.size() == 3;
  }
  if (sm.updated("pandaStates")) {
    auto pandaStates = sm["pandaStates"].getPandaStates();
    if (pandaStates.size() > 0) {
      scene.pandaType = pandaStates[0].getPandaType();

      if (scene.pandaType != cereal::PandaState::PandaType::UNKNOWN) {
        scene.ignition = false;
        for (const auto& pandaState : pandaStates) {
          scene.ignition |= pandaState.getIgnitionLine() || pandaState.getIgnitionCan();
        }
      }
    }
  } else if ((s->sm->frame - s->sm->rcv_frame("pandaStates")) > 5*UI_FREQ) {
    scene.pandaType = cereal::PandaState::PandaType::UNKNOWN;
  }
  if (sm.updated("carParams")) {
    scene.longitudinal_control = sm["carParams"].getCarParams().getOpenpilotLongitudinalControl();
  }
  if (sm.updated("wideRoadCameraState")) {
    float scale = (sm["wideRoadCameraState"].getWideRoadCameraState().getSensor() == cereal::FrameData::ImageSensor::AR0231) ? 6.0f : 1.0f;
    scene.light_sensor = std::max(100.0f - scale * sm["wideRoadCameraState"].getWideRoadCameraState().getExposureValPercent(), 0.0f);
  }
  scene.started = sm["deviceState"].getDeviceState().getStarted() && scene.ignition;
}

void ui_update_params(UIState *s) {
  auto params = Params();
  s->scene.is_metric = params.getBool("IsMetric");
  s->scene.map_on_left = params.getBool("NavSettingLeftSide");
}

void UIState::updateStatus() {
  if (scene.started && sm->updated("controlsState")) {
    auto controls_state = (*sm)["controlsState"].getControlsState();
    auto alert_status = controls_state.getAlertStatus();
    auto state = controls_state.getState();
    if (alert_status == cereal::ControlsState::AlertStatus::USER_PROMPT) {
      status = STATUS_WARNING;
    } else if (alert_status == cereal::ControlsState::AlertStatus::CRITICAL) {
      status = STATUS_ALERT;
    } else if (state == cereal::ControlsState::OpenpilotState::PRE_ENABLED || state == cereal::ControlsState::OpenpilotState::OVERRIDING) {
      status = STATUS_OVERRIDE;
    } else {
      status = controls_state.getEnabled() ? STATUS_ENGAGED : STATUS_DISENGAGED;
    }
  }

  // Handle onroad/offroad transition
  if (scene.started != started_prev || sm->frame == 1) {
    if (scene.started) {
      status = STATUS_DISENGAGED;
      scene.started_frame = sm->frame;
      wide_cam_only = Params().getBool("WideCameraOnly");
    }
    started_prev = scene.started;
    emit offroadTransition(!scene.started);
  }

  // Handle prime type change
  if (prime_type != prime_type_prev) {
    prime_type_prev = prime_type;
    emit primeTypeChanged(prime_type);
    Params().put("PrimeType", std::to_string(prime_type));
  }
}

UIState::UIState(QObject *parent) : QObject(parent) {
  sm = std::make_unique<SubMaster, const std::initializer_list<const char *>>({
    "modelV2", "controlsState", "liveCalibration", "radarState", "deviceState", "roadCameraState",
    "pandaStates", "carParams", "driverMonitoringState", "carState", "liveLocationKalman", "driverStateV2",
    "wideRoadCameraState", "managerState", "navInstruction", "navRoute", "gnssMeasurements",
  });

  Params params;
  wide_cam_only = params.getBool("WideCameraOnly");
  prime_type = std::atoi(params.get("PrimeType").c_str());
  language = QString::fromStdString(params.get("LanguageSetting"));

  // update timer
  timer = new QTimer(this);
  QObject::connect(timer, &QTimer::timeout, this, &UIState::update);
  timer->start(1000 / UI_FREQ);
}

void UIState::update() {
  update_sockets(this);
  update_state(this);
  updateStatus();

  if (sm->frame % UI_FREQ == 0) {
    watchdog_kick(nanos_since_boot());
  }
  emit uiUpdate(*this);
}

Device::Device(QObject *parent) : brightness_filter(BACKLIGHT_OFFROAD, BACKLIGHT_TS, BACKLIGHT_DT), QObject(parent) {
  setAwake(true);
  resetInteractiveTimout();

  QObject::connect(uiState(), &UIState::uiUpdate, this, &Device::update);
}

void Device::update(const UIState &s) {
  updateBrightness(s);
  updateWakefulness(s);

  // TODO: remove from UIState and use signals
  uiState()->awake = awake;
}

void Device::setAwake(bool on) {
  if (on != awake) {
    awake = on;
    Hardware::set_display_power(awake);
    LOGD("setting display power %d", awake);
    emit displayPowerChanged(awake);
  }
}

void Device::resetInteractiveTimout() {
  interactive_timeout = (ignition_on ? 10 : 30) * UI_FREQ;
}

void Device::updateBrightness(const UIState &s) {
  float clipped_brightness = BACKLIGHT_OFFROAD;
  if (s.scene.started) {
    clipped_brightness = s.scene.light_sensor;

    // CIE 1931 - https://www.photonstophotos.net/GeneralTopics/Exposure/Psychometric_Lightness_and_Gamma.htm
    if (clipped_brightness <= 8) {
      clipped_brightness = (clipped_brightness / 903.3);
    } else {
      clipped_brightness = std::pow((clipped_brightness + 16.0) / 116.0, 3.0);
    }

    // Scale back to 10% to 100%
    clipped_brightness = std::clamp(100.0f * clipped_brightness, 10.0f, 100.0f);
  }

  int brightness = brightness_filter.update(clipped_brightness);
  if (!awake) {
    brightness = 0;
  }

  if (brightness != last_brightness) {
    if (!brightness_future.isRunning()) {
      brightness_future = QtConcurrent::run(Hardware::set_brightness, brightness);
      last_brightness = brightness;
    }
  }
}

void Device::updateWakefulness(const UIState &s) {
  bool ignition_just_turned_off = !s.scene.ignition && ignition_on;
  ignition_on = s.scene.ignition;

  if (ignition_just_turned_off) {
    resetInteractiveTimout();
  } else if (interactive_timeout > 0 && --interactive_timeout == 0) {
    emit interactiveTimout();
  }

  setAwake(s.scene.ignition || interactive_timeout > 0);
}

UIState *uiState() {
  static UIState ui_state;
  return &ui_state;
}
