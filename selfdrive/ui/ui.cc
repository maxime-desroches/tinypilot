#include <stdio.h>
#include <cmath>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <assert.h>

#include "common/util.h"
#include "common/swaglog.h"
#include "common/visionimg.h"
#include "ui.hpp"
#include "paint.hpp"


int write_param_float(float param, const char* param_name, bool persistent_param) {
  char s[16];
  int size = snprintf(s, sizeof(s), "%f", param);
  return Params(persistent_param).write_db_value(param_name, s, size < sizeof(s) ? size : sizeof(s));
}


static void ui_init_vision(UIState *s) {
  // Invisible until we receive a calibration message.
  s->scene.world_objects_visible = false;

  for (int i = 0; i < s->vipc_client->num_buffers; i++) {
    s->texture[i].reset(new EGLImageTexture(&s->vipc_client->buffers[i]));

    glBindTexture(GL_TEXTURE_2D, s->texture[i]->frame_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    // BGR
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_R, GL_BLUE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_GREEN);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_RED);
  }
  assert(glGetError() == GL_NO_ERROR);
}


void ui_init(UIState *s) {
  s->sm = new SubMaster({"modelV2", "controlsState", "uiLayoutState", "liveCalibration", "radarState", "thermal", "frame",
                         "health", "carParams", "ubloxGnss", "driverState", "dMonitoringState", "sensorEvents"});

  s->started = false;
  s->status = STATUS_OFFROAD;
  s->scene.satelliteCount = -1;

  s->fb = framebuffer_init("ui", 0, true, &s->fb_w, &s->fb_h);
  assert(s->fb);

  ui_nvg_init(s);

  s->last_frame = nullptr;
  s->vipc_client_rear = new VisionIpcClient("camerad", VISION_STREAM_RGB_BACK, true);
  s->vipc_client_front = new VisionIpcClient("camerad", VISION_STREAM_RGB_FRONT, true);
  s->vipc_client = s->vipc_client_rear;
}

template <class T>
static void update_line_data(const UIState *s, const cereal::ModelDataV2::XYZTData::Reader &line,
                             float y_off, float z_off, T *pvd, float max_distance) {
  const auto line_x = line.getX(), line_y = line.getY(), line_z = line.getZ();
  int max_idx = -1;
  vertex_data *v = &pvd->v[0];
  const float margin = 500.0f;
  for (int i = 0; ((i < TRAJECTORY_SIZE) and (line_x[i] < fmax(MIN_DRAW_DISTANCE, max_distance))); i++) {
    v += car_space_to_full_frame(s, line_x[i], -line_y[i] - y_off, -line_z[i] + z_off, v, margin);
    max_idx = i;
  }
  for (int i = max_idx; i >= 0; i--) {
    v += car_space_to_full_frame(s, line_x[i], -line_y[i] + y_off, -line_z[i] + z_off, v, margin);
  }
  pvd->cnt = v - pvd->v;
  assert(pvd->cnt < std::size(pvd->v));
}

static void update_model(UIState *s, const cereal::ModelDataV2::Reader &model) {
  UIScene &scene = s->scene;
  const float max_distance = fmin(model.getPosition().getX()[TRAJECTORY_SIZE - 1], MAX_DRAW_DISTANCE);
  // update lane lines
  const auto lane_lines = model.getLaneLines();
  const auto lane_line_probs = model.getLaneLineProbs();
  for (int i = 0; i < std::size(scene.lane_line_vertices); i++) {
    scene.lane_line_probs[i] = lane_line_probs[i];
    update_line_data(s, lane_lines[i], 0.025 * scene.lane_line_probs[i], 1.22, &scene.lane_line_vertices[i], max_distance);
  }

  // update road edges
  const auto road_edges = model.getRoadEdges();
  const auto road_edge_stds = model.getRoadEdgeStds();
  for (int i = 0; i < std::size(scene.road_edge_vertices); i++) {
    scene.road_edge_stds[i] = road_edge_stds[i];
    update_line_data(s, road_edges[i], 0.025, 1.22, &scene.road_edge_vertices[i], max_distance);
  }

  // update path
  const float lead_d = scene.lead_data[0].getStatus() ? scene.lead_data[0].getDRel() * 2. : MAX_DRAW_DISTANCE;
  float path_length = (lead_d > 0.) ? lead_d - fmin(lead_d * 0.35, 10.) : MAX_DRAW_DISTANCE;
  path_length = fmin(path_length, max_distance);
  update_line_data(s, model.getPosition(), 0.5, 0, &scene.track_vertices, path_length);
}

void update_sockets(UIState *s) {

  UIScene &scene = s->scene;
  SubMaster &sm = *(s->sm);

  if (sm.update(0) == 0){
    return;
  }

  if (s->started && sm.updated("controlsState")) {
    auto event = sm["controlsState"];
    scene.controls_state = event.getControlsState();
  }
  if (sm.updated("radarState")) {
    auto data = sm["radarState"].getRadarState();
    scene.lead_data[0] = data.getLeadOne();
    scene.lead_data[1] = data.getLeadTwo();
  }
  if (sm.updated("liveCalibration")) {
    scene.world_objects_visible = true;
    auto extrinsicl = sm["liveCalibration"].getLiveCalibration().getExtrinsicMatrix();
    for (int i = 0; i < 3 * 4; i++) {
      scene.extrinsic_matrix.v[i] = extrinsicl[i];
    }
  }
  if (sm.updated("modelV2")) {
    update_model(s, sm["modelV2"].getModelV2());
  }
  if (sm.updated("uiLayoutState")) {
    auto data = sm["uiLayoutState"].getUiLayoutState();
    s->active_app = data.getActiveApp();
    scene.sidebar_collapsed = data.getSidebarCollapsed();
  }
  if (sm.updated("thermal")) {
    scene.thermal = sm["thermal"].getThermal();
  }
  if (sm.updated("ubloxGnss")) {
    auto data = sm["ubloxGnss"].getUbloxGnss();
    if (data.which() == cereal::UbloxGnss::MEASUREMENT_REPORT) {
      scene.satelliteCount = data.getMeasurementReport().getNumMeas();
    }
  }
  if (sm.updated("health")) {
    auto health = sm["health"].getHealth();
    scene.hwType = health.getHwType();
    s->ignition = health.getIgnitionLine() || health.getIgnitionCan();
  } else if ((s->sm->frame - s->sm->rcv_frame("health")) > 5*UI_FREQ) {
    scene.hwType = cereal::HealthData::HwType::UNKNOWN;
  }
  if (sm.updated("carParams")) {
    s->longitudinal_control = sm["carParams"].getCarParams().getOpenpilotLongitudinalControl();
  }
  if (sm.updated("driverState")) {
    scene.driver_state = sm["driverState"].getDriverState();
  }
  if (sm.updated("dMonitoringState")) {
    scene.dmonitoring_state = sm["dMonitoringState"].getDMonitoringState();
    scene.is_rhd = scene.dmonitoring_state.getIsRHD();
    scene.frontview = scene.dmonitoring_state.getIsPreview();
  } else if (scene.frontview && (sm.frame - sm.rcv_frame("dMonitoringState")) > UI_FREQ/2) {
    scene.frontview = false;
  }
  if (sm.updated("sensorEvents")) {
    for (auto sensor : sm["sensorEvents"].getSensorEvents()) {
      if (sensor.which() == cereal::SensorEventData::LIGHT) {
        s->light_sensor = sensor.getLight();
      } else if (!s->started && sensor.which() == cereal::SensorEventData::ACCELERATION) {
        s->accel_sensor = sensor.getAcceleration().getV()[2];
      } else if (!s->started && sensor.which() == cereal::SensorEventData::GYRO_UNCALIBRATED) {
        s->gyro_sensor = sensor.getGyroUncalibrated().getV()[1];
      }
    }
  }

  s->started = scene.thermal.getStarted() || scene.frontview;
}

static void ui_handle_alert(UIState *s) {
  if (!s->started) return;

  UIScene &scene = s->scene;
  const uint64_t frame = s->sm->frame;
  const uint64_t cs_frame = s->sm->rcv_frame("controlsState");
  // Handle controls/fcamera timeout
  if (const uint64_t since_started = frame - s->started_frame;
      !scene.frontview && since_started > 10 * UI_FREQ) {
    if (cs_frame < s->started_frame) {
      // car is started, but controlsState hasn't been seen at all
      scene.alert_text1 = "openpilot Unavailable";
      scene.alert_text2 = "Waiting for controls to start";
      scene.alert_size = cereal::ControlsState::AlertSize::MID;
      return;
    } else if ((frame - cs_frame) > 5 * UI_FREQ) {
      // car is started, but controls is lagging or died
      if (scene.alert_text2 != "Controls Unresponsive" &&
          scene.alert_text1 != "Camera Malfunction") {
        s->sound->play(AudibleAlert::CHIME_WARNING_REPEAT);
        LOGE("Controls unresponsive");
      }

      scene.alert_text1 = "TAKE CONTROL IMMEDIATELY";
      scene.alert_text2 = "Controls Unresponsive";
      scene.alert_size = cereal::ControlsState::AlertSize::FULL;
      s->status = STATUS_ALERT;
      return;
    }

    const uint64_t frame_pkt = (s->sm)->rcv_frame("frame");
    if ((frame_pkt > s->started_frame || since_started > 15 * UI_FREQ) &&
        (frame - frame_pkt) > 5 * UI_FREQ) {
      // controls is fine, but rear camera is lagging or died
      scene.alert_text1 = "Camera Malfunction";
      scene.alert_text2 = "Contact Support";
      scene.alert_size = cereal::ControlsState::AlertSize::FULL;
      s->status = STATUS_DISENGAGED;
      s->sound->stop();
      return;
    }
  }

  if (cs_frame != frame) { return; }

  const auto &cs = scene.controls_state;
  if (scene.alert_type.compare(cs.getAlertType()) != 0) {
    auto alert_sound = cs.getAlertSound();
    if (alert_sound == AudibleAlert::NONE) {
      s->sound->stop();
    } else {
      s->sound->play(alert_sound);
    }
  }
  scene.alert_text1 = cs.getAlertText1();
  scene.alert_text2 = cs.getAlertText2();
  scene.alert_size = cs.getAlertSize();
  scene.alert_type = cs.getAlertType();
  auto alert_status = cs.getAlertStatus();
  if (alert_status == cereal::ControlsState::AlertStatus::USER_PROMPT) {
    s->status = STATUS_WARNING;
  } else if (alert_status == cereal::ControlsState::AlertStatus::CRITICAL) {
    s->status = STATUS_ALERT;
  } else {
    s->status = cs.getEnabled() ? STATUS_ENGAGED : STATUS_DISENGAGED;
  }

  float alert_blinkingrate = cs.getAlertBlinkingRate();
  if (alert_blinkingrate > 0.) {
    if (s->alert_blinked) {
      if (s->alert_blinking_alpha > 0.0 && s->alert_blinking_alpha < 1.0) {
        s->alert_blinking_alpha += (0.05 * alert_blinkingrate);
      } else {
        s->alert_blinked = false;
      }
    } else {
      if (s->alert_blinking_alpha > 0.25) {
        s->alert_blinking_alpha -= (0.05 * alert_blinkingrate);
      } else {
        s->alert_blinking_alpha += 0.25;
        s->alert_blinked = true;
      }
    }
  }
}

void ui_update_vision(UIState *s) {
  if (!s->vipc_client->connected && s->started) {
    s->vipc_client = s->scene.frontview ? s->vipc_client_front : s->vipc_client_rear;

    if (s->vipc_client->connect(false)){
      ui_init_vision(s);
    }
  }

  if (s->vipc_client->connected){
    VisionBuf * buf = s->vipc_client->recv();
    if (buf != nullptr){
      s->last_frame = buf;
    }
  }
}

void ui_update(UIState *s) {

void ui_update(UIState *s) {
  ui_read_params(s);
  update_sockets(s);
  ui_update_vision(s);

  // Handle onroad/offroad transition
  if (!s->started && s->status != STATUS_OFFROAD) {
    s->status = STATUS_OFFROAD;
    s->active_app = cereal::UiLayoutState::App::HOME;
    s->scene.sidebar_collapsed = false;
    s->sound->stop();
    s->vipc_client->connected = false;
  } else if (s->started && s->status == STATUS_OFFROAD) {
    s->status = STATUS_DISENGAGED;
    s->started_frame = s->sm->frame;

    s->active_app = cereal::UiLayoutState::App::NONE;
    s->scene.sidebar_collapsed = true;
    s->scene.alert_size = cereal::ControlsState::AlertSize::NONE;
  }

  // Handle controls timeout
  if (s->started && !s->scene.frontview && ((s->sm)->frame - s->started_frame) > 10*UI_FREQ) {
    if ((s->sm)->rcv_frame("controlsState") < s->started_frame) {
      // car is started, but controlsState hasn't been seen at all
      s->scene.alert_text1 = "openpilot Unavailable";
      s->scene.alert_text2 = "Waiting for controls to start";
      s->scene.alert_size = cereal::ControlsState::AlertSize::MID;
    } else if (((s->sm)->frame - (s->sm)->rcv_frame("controlsState")) > 5*UI_FREQ) {
      // car is started, but controls is lagging or died
      if (s->scene.alert_text2 != "Controls Unresponsive") {
        s->sound->play(AudibleAlert::CHIME_WARNING_REPEAT);
        LOGE("Controls unresponsive");
      }

      s->scene.alert_text1 = "TAKE CONTROL IMMEDIATELY";
      s->scene.alert_text2 = "Controls Unresponsive";
      s->scene.alert_size = cereal::ControlsState::AlertSize::FULL;
      s->status = STATUS_ALERT;
    }
  }
}
