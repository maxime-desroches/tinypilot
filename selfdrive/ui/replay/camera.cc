#include "selfdrive/ui/replay/camera.h"

#include <cassert>
#include <iostream>

const int YUV_BUF_COUNT = 50;

CameraServer::CameraServer(std::pair<int, int> cameras[MAX_CAMERAS]) {
  if (cameras) {
    for (auto type : ALL_CAMERAS) {
      std::tie(cameras_[type].width, cameras_[type].height) = cameras[type];
    }
    startVipcServer();
  }
}

CameraServer::~CameraServer() {
  for (auto &cam : cameras_) {
    if (cam.thread.joinable()) {
      cam.queue.push({});
      cam.thread.join();
    }
  }
  vipc_server_.reset(nullptr);
}

void CameraServer::startVipcServer() {
  vipc_server_.reset(new VisionIpcServer("camerad"));
  for (auto &type : ALL_CAMERAS) {
    auto &cam = cameras_[type];
    if (cam.width > 0 && cam.height > 0) {
      vipc_server_->create_buffers(cam.rgb_type, UI_BUF_COUNT, true, cam.width, cam.height);
      vipc_server_->create_buffers(cam.yuv_type, YUV_BUF_COUNT, false, cam.width, cam.height);
      if (!cam.thread.joinable()) {
        cam.thread = std::thread(&CameraServer::cameraThread, this, type, std::ref(cam));
      }
    }
  }
  vipc_server_->start_listener();
}

void CameraServer::cameraThread(CameraType type, Camera &cam) {
  while (true) {
    const auto [fr, eidx] = cam.queue.pop();
    if (!fr) break;

    // send frame
    VisionBuf *rgb_buf = vipc_server_->get_buffer(cam.rgb_type);
    VisionBuf *yuv_buf = vipc_server_->get_buffer(cam.yuv_type);
    if (fr->get(eidx.getSegmentId(), (uint8_t *)rgb_buf->addr, (uint8_t *)yuv_buf->addr)) {
      VisionIpcBufExtra extra = {
          .frame_id = eidx.getFrameId(),
          .timestamp_sof = eidx.getTimestampSof(),
          .timestamp_eof = eidx.getTimestampEof(),
      };
      vipc_server_->send(rgb_buf, &extra, false);
      vipc_server_->send(yuv_buf, &extra, false);
    } else {
      std::cout << "camera[" << type << "] failed to get frame:" << eidx.getSegmentId() << std::endl;
    }

    --publishing_;
  }
}

void CameraServer::pushFrame(CameraType type, FrameReader *fr, const cereal::EncodeIndex::Reader &eidx) {
  auto &cam = cameras_[type];
  if (cam.width != fr->width || cam.height != fr->height) {
    cam.width = fr->width;
    cam.height = fr->height;
    std::cout << "camera[" << type << "] frame size " << cam.width << "x" << cam.height << std::endl;
    waitFinish();
    startVipcServer();
  }

  ++publishing_;
  cam.queue.push({fr, eidx});
}
