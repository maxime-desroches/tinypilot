#pragma once

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "selfdrive/camerad/cameras/camera_common.h"

#define FRAME_BUF_COUNT 16

typedef struct CameraState {
  CameraInfo ci;
  int camera_num;
  int fps;
  float digital_gain;
  CameraBuf buf;
} CameraState;

class CameraServer : public CameraServerBase {
public:
  CameraServer();
  ~CameraServer();
  void run() override;

  CameraState road_cam = {};
  CameraState driver_cam = {};
};
