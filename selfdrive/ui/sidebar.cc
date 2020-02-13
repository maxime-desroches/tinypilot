#include <stdio.h>
#include <string.h>
#include <math.h>
#include "ui.hpp"

static void ui_draw_sidebar_background(UIState *s) {
  const UIScene *scene = &s->scene;
  bool hasSidebar = !s->scene.uilayout_sidebarcollapsed;
  int sbr_x = hasSidebar ? 0 : -(sbr_w) + bdr_s * 2;

  nvgBeginPath(s->vg);
  nvgRect(s->vg, sbr_x, 0, sbr_w, vwp_h);
  nvgFillColor(s->vg, nvgRGBAf(0,0,0,0.53));
  nvgFill(s->vg);
}

static void ui_draw_sidebar_settings_button(UIState *s) {
  const UIScene *scene = &s->scene;
  bool hasSidebar = !s->scene.uilayout_sidebarcollapsed;
  const int settings_btn_h = 117;
  const int settings_btn_w = 200;
  const int settings_btn_x = hasSidebar ? 50 : -(sbr_w);
  const int settings_btn_y = 35;

  nvgBeginPath(s->vg);
  NVGpaint imgPaint = nvgImagePattern(s->vg, settings_btn_x, settings_btn_y,
    settings_btn_w, settings_btn_h, 0, s->img_button_settings, 1.0f);
  nvgRect(s->vg, settings_btn_x, settings_btn_y, settings_btn_w, settings_btn_h);
  nvgFillPaint(s->vg, imgPaint);
  nvgFill(s->vg);
}

static void ui_draw_sidebar_home_button(UIState *s) {
  const UIScene *scene = &s->scene;
  bool hasSidebar = !s->scene.uilayout_sidebarcollapsed;
  const int home_btn_h = 180;
  const int home_btn_w = 180;
  const int home_btn_x = hasSidebar ? 60 : -(sbr_w);
  const int home_btn_y = vwp_h - home_btn_h - 40;

  nvgBeginPath(s->vg);
  NVGpaint imgPaint = nvgImagePattern(s->vg, home_btn_x, home_btn_y,
    home_btn_w, home_btn_h, 0, s->img_button_home, 1.0f);
  nvgRect(s->vg, home_btn_x, home_btn_y, home_btn_w, home_btn_h);
  nvgFillPaint(s->vg, imgPaint);
  nvgFill(s->vg);
}

static void ui_draw_sidebar_battery_icon(UIState *s) {
  const UIScene *scene = &s->scene;
  bool hasSidebar = !s->scene.uilayout_sidebarcollapsed;
  const int battery_img_h = 36;
  const int battery_img_w = 76;
  const int battery_img_x = hasSidebar ? 160 : -(sbr_w);
  const int battery_img_y = 255;

  int battery_img = strcmp(s->scene.batteryStatus, "Charging") == 0 ?
    s->img_battery_0_charging : s->img_battery_0;

  nvgBeginPath(s->vg);
  nvgRect(s->vg, battery_img_x + 6, battery_img_y + 5,
    ((battery_img_w - 19) * (s->scene.batteryPercent * 0.01)), battery_img_h - 11);
  nvgFillColor(s->vg, nvgRGBAf(255,255,255,1.0));
  nvgFill(s->vg);

  nvgBeginPath(s->vg);
  NVGpaint imgPaint = nvgImagePattern(s->vg, battery_img_x, battery_img_y,
    battery_img_w, battery_img_h, 0, battery_img, 1.0f);
  nvgRect(s->vg, battery_img_x, battery_img_y, battery_img_w, battery_img_h);
  nvgFillPaint(s->vg, imgPaint);
  nvgFill(s->vg);
}

static void ui_draw_sidebar_network_type(UIState *s) {
  const UIScene *scene = &s->scene;
  bool hasSidebar = !s->scene.uilayout_sidebarcollapsed;
  const int network_x = hasSidebar ? 50 : -(sbr_w);
  const int network_y = 273;
  const int network_w = 100;
  const int network_h = 100;
  char network_type_str[32];

  if (s->scene.networkType == NETWORKTYPE_NONE) {
    snprintf(network_type_str, sizeof(network_type_str), "%s", "--");
  } else if (s->scene.networkType == NETWORKTYPE_WIFI) {
    snprintf(network_type_str, sizeof(network_type_str), "%s", "WiFi");
  } else if (s->scene.networkType == NETWORKTYPE_CELL2G) {
    snprintf(network_type_str, sizeof(network_type_str), "%s", "2G");
  } else if (s->scene.networkType == NETWORKTYPE_CELL3G) {
    snprintf(network_type_str, sizeof(network_type_str), "%s", "3G");
  } else if (s->scene.networkType == NETWORKTYPE_CELL4G) {
    snprintf(network_type_str, sizeof(network_type_str), "%s", "LTE");
  } else if (s->scene.networkType == NETWORKTYPE_CELL5G) {
    snprintf(network_type_str, sizeof(network_type_str), "%s", "5G");
  }

  nvgFillColor(s->vg, nvgRGBA(255, 255, 255, 255));
  nvgFontSize(s->vg, 48);
  nvgFontFace(s->vg, "sans-regular");
  nvgTextAlign(s->vg, NVG_ALIGN_CENTER | NVG_ALIGN_MIDDLE);
  nvgTextBox(s->vg, network_x, network_y, network_w, network_type_str, NULL);
}

static void ui_draw_sidebar_metric(UIState *s, const char* label_str, const char* value_str, const int severity, const int y_offset) {
  const UIScene *scene = &s->scene;
  bool hasSidebar = !s->scene.uilayout_sidebarcollapsed;
  const int metric_x = hasSidebar ? 30 : -(sbr_w);
  const int metric_y = 338 + y_offset;
  const int metric_w = 240;
  const int metric_h = 148;
  NVGcolor status_color;

  if (severity == 0) {
    status_color = nvgRGBA(255, 255, 255, 255);
  } else if (severity == 1) {
    status_color = nvgRGBA(218, 202, 37, 255);
  } else if (severity == 2) {
    status_color = nvgRGBA(201, 34, 49, 255);
  } else if (severity == 3) {
    status_color = nvgRGBA(201, 34, 49, 255);
  }

  nvgBeginPath(s->vg);
  nvgRoundedRect(s->vg, metric_x, metric_y, metric_w, metric_h, 20);
  nvgStrokeColor(s->vg, nvgRGBA(255, 255, 255, severity > 0 ? 255 : 125));
  nvgStrokeWidth(s->vg, 2);
  nvgStroke(s->vg);

  nvgBeginPath(s->vg);
  nvgRoundedRectVarying(s->vg, metric_x + 6, metric_y + 6, 18, metric_h - 12, 25, 0, 0, 25);
  nvgFillColor(s->vg, status_color);
  nvgFill(s->vg);

  nvgFillColor(s->vg, nvgRGBA(255, 255, 255, 255));
  nvgFontSize(s->vg, 78);
  nvgFontFace(s->vg, "sans-bold");
  nvgTextAlign(s->vg, NVG_ALIGN_LEFT | NVG_ALIGN_MIDDLE);
  nvgTextBox(s->vg, metric_x + 50, metric_y + 50, metric_w - 60, value_str, NULL);

  nvgFillColor(s->vg, nvgRGBA(255, 255, 255, 255));
  nvgFontSize(s->vg, 48);
  nvgFontFace(s->vg, "sans-regular");
  nvgTextAlign(s->vg, NVG_ALIGN_LEFT | NVG_ALIGN_MIDDLE);
  nvgTextBox(s->vg, metric_x + 50, metric_y + 50 + 66, metric_w - 60, label_str, NULL);
}

static void ui_draw_sidebar_metric_storage(UIState *s) {
  int storage_severity;
  char storage_label_str[32];
  char storage_value_str[32];
  char storage_value_unit[32];
  const int storage_y_offset = 0;
  const float storage_pct = ceilf((1.0 - s->scene.freeSpace) * 100);

  if (storage_pct < 75.0) {
    storage_severity = 0;
  } else if (storage_pct >= 75.0 && storage_pct < 87.0) {
    storage_severity = 1;
  } else if (storage_pct >= 87.0) {
    storage_severity = 2;
  }

  snprintf(storage_value_str, sizeof(storage_value_str), "%d", (int)storage_pct);
  snprintf(storage_value_unit, sizeof(storage_value_unit), "%s", "%");
  snprintf(storage_label_str, sizeof(storage_label_str), "%s", "STORAGE");
  strcat(storage_value_str, storage_value_unit);

  ui_draw_sidebar_metric(s, storage_label_str, storage_value_str, storage_severity, storage_y_offset);
}

static void ui_draw_sidebar_metric_temp(UIState *s) {
  int temp_severity;
  char temp_label_str[32];
  char temp_value_str[32];
  char temp_value_unit[32];
  const int temp_y_offset = 148 + 32;

  if (s->scene.thermalStatus == THERMALSTATUS_GREEN) {
    temp_severity = 0;
  } else if (s->scene.thermalStatus == THERMALSTATUS_YELLOW) {
    temp_severity = 1;
  } else if (s->scene.thermalStatus == THERMALSTATUS_RED) {
    temp_severity = 2;
  } else if (s->scene.thermalStatus == THERMALSTATUS_DANGER) {
    temp_severity = 3;
  }

  snprintf(temp_value_str, sizeof(temp_value_str), "%d", s->scene.paTemp);
  snprintf(temp_value_unit, sizeof(temp_value_unit), "%s", "°C");
  snprintf(temp_label_str, sizeof(temp_label_str), "%s", "TEMP");
  strcat(temp_value_str, temp_value_unit);

  ui_draw_sidebar_metric(s, temp_label_str, temp_value_str, temp_severity, temp_y_offset);
}

void ui_draw_sidebar(UIState *s) {
  ui_draw_sidebar_background(s);
  ui_draw_sidebar_settings_button(s);
  ui_draw_sidebar_home_button(s);
  ui_draw_sidebar_battery_icon(s);
  ui_draw_sidebar_network_type(s);
  ui_draw_sidebar_metric_storage(s);
  ui_draw_sidebar_metric_temp(s);
}
