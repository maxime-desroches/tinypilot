#pragma once

#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <memory>
#include <thread>
#include "common/mat.h"
#include "common/swaglog.h"
#include "common/queue.h"
#include "visionbuf.h"
#include "common/visionimg.h"
#include "messaging.hpp"
#include "transforms/rgb_to_yuv.h"

#include "visionipc.h"
#include "visionipc_server.h"

#define CAMERA_ID_IMX298 0
#define CAMERA_ID_IMX179 1
#define CAMERA_ID_S5K3P8SP 2
#define CAMERA_ID_OV8865 3
#define CAMERA_ID_IMX298_FLIPPED 4
#define CAMERA_ID_OV10640 5
#define CAMERA_ID_LGC920 6
#define CAMERA_ID_LGC615 7
#define CAMERA_ID_AR0231 8
#define CAMERA_ID_MAX 9

#define UI_BUF_COUNT 4
#define YUV_COUNT 40
#define LOG_CAMERA_ID_FCAMERA 0
#define LOG_CAMERA_ID_DCAMERA 1
#define LOG_CAMERA_ID_ECAMERA 2
#define LOG_CAMERA_ID_QCAMERA 3
#define LOG_CAMERA_ID_MAX 4

#define HLC_THRESH 222
#define HLC_A 80
#define HISTO_CEIL_K 5

const bool env_send_driver = getenv("SEND_DRIVER") != NULL;
const bool env_send_road = getenv("SEND_ROAD") != NULL;
const bool env_send_wide_road = getenv("SEND_WIDE_ROAD") != NULL;

typedef void (*release_cb)(void *cookie, int buf_idx);

typedef struct CameraInfo {
  int frame_width, frame_height;
  int frame_stride;
  bool bayer;
  int bayer_flip;
  bool hdr;
} CameraInfo;

typedef struct LogCameraInfo {
  const char* filename;
  const char* frame_packet_name;
  const char* encode_idx_name;
  VisionStreamType stream_type;
  int frame_width, frame_height;
  int fps;
  int bitrate;
  bool is_h265;
  bool downscale;
  bool has_qcamera;
} LogCameraInfo;

typedef struct FrameMetadata {
  uint32_t frame_id;
  uint64_t timestamp_sof; // only set on tici
  uint64_t timestamp_eof;
  unsigned int frame_length;
  unsigned int integ_lines;
  unsigned int global_gain;
  unsigned int lens_pos;
  float lens_sag;
  float lens_err;
  float lens_true_pos;
  float gain_frac;
} FrameMetadata;

typedef struct CameraExpInfo {
  int op_id;
  float grey_frac;
} CameraExpInfo;

class CameraServer;
struct CameraState;

class CameraServerBase {
public:
  CameraServerBase();
  virtual ~CameraServerBase();
  void start();

  cl_device_id device_id;
  cl_context context;
  VisionIpcServer *vipc_server;
  PubMaster *pm;

protected:
  virtual void run() = 0;
  virtual void process_camera(CameraState *cs, cereal::FrameData::Builder& framed, uint32_t cnt) {}
  void start_process_thread(CameraState *cs, bool is_frame_stream = false);
  std::vector<std::thread> camera_threads;

private:
  void process_camera_thread(CameraState *cs, bool is_frame_stream);
};

class CameraBuf {
private:
  VisionIpcServer *vipc_server;
  CameraState *camera_state;
  cl_kernel krnl_debayer;

  RGBToYUVState rgb_to_yuv_state;

  VisionStreamType rgb_type, yuv_type;

  int cur_buf_idx;

  SafeQueue<int> safe_queue;

  int frame_buf_count;
  release_cb release_callback;

public:
  cl_command_queue q;
  FrameMetadata cur_frame_data;
  VisionBuf *cur_rgb_buf;
  VisionBuf *cur_yuv_buf;
  std::unique_ptr<VisionBuf[]> camera_bufs;
  std::unique_ptr<FrameMetadata[]> camera_bufs_metadata;
  int rgb_width, rgb_height, rgb_stride;

  mat3 yuv_transform;

  CameraBuf() = default;
  ~CameraBuf();
  void init(CameraServer *server, CameraState *s, int frame_cnt, release_cb release_callback = nullptr);
  bool acquire();
  void release();
  void queue(size_t buf_idx);
};

float set_exposure_target(const CameraBuf *b, int x_start, int x_end, int x_skip, int y_start, int y_end, int y_skip, int analog_gain, bool hist_ceil, bool hl_weighted);
void camera_autoexposure(CameraState *s, float grey_frac);
