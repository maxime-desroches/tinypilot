#pragma once

#include "cereal/messaging/messaging.h"
#include "cereal/services.h"
#include "cereal/visionipc/visionipc_client.h"
#include "system/camerad/cameras/camera_common.h"
#include "system/hardware/hw.h"
#include "common/params.h"
#include "common/swaglog.h"
#include "common/util.h"

#include "system/loggerd/logger.h"

constexpr int MAIN_FPS = 20;
const int MAIN_BITRATE = 10000000;

#define NO_CAMERA_PATIENCE 500 // fall back to time-based rotation if all cameras are dead

const bool LOGGERD_TEST = getenv("LOGGERD_TEST");
const int SEGMENT_LENGTH = LOGGERD_TEST ? atoi(getenv("LOGGERD_SEGMENT_LENGTH")) : 60;

class EncoderInfo {
public:
  const char *publish_name;
  const char *filename;
  bool record = true;
  int frame_width = 1928;
  int frame_height = 1208;
  int fps = MAIN_FPS;
  int bitrate = MAIN_BITRATE;
  cereal::EncodeIndex::Type encode_type = Hardware::PC() ? cereal::EncodeIndex::Type::BIG_BOX_LOSSLESS
                                                         : cereal::EncodeIndex::Type::FULL_H_E_V_C;
  ::cereal::EncodeData::Reader (cereal::Event::Reader::*get_encode_data_func)() const;
  void (cereal::Event::Builder::*set_encode_idx_func)(::cereal::EncodeIndex::Reader);
  cereal::EncodeData::Builder (cereal::Event::Builder::*init_encode_data_func)();
};

class LogCameraInfo {
public:
  const char *thread_name;
  int fps = MAIN_FPS;
  CameraType type;
  VisionStreamType stream_type;
  std::vector<EncoderInfo> encoder_infos;
};

const EncoderInfo main_road_encoder_info = {
  .publish_name = "roadEncodeData",
  .filename = "fcamera.hevc",
  .get_encode_data_func = &cereal::Event::Reader::getRoadEncodeData,
  .set_encode_idx_func = &cereal::Event::Builder::setRoadEncodeIdx,
  .init_encode_data_func = &cereal::Event::Builder::initRoadEncodeData,
};
const EncoderInfo main_wide_road_encoder_info = {
  .publish_name = "wideRoadEncodeData",
  .filename = "ecamera.hevc",
  .get_encode_data_func = &cereal::Event::Reader::getWideRoadEncodeData,
  .set_encode_idx_func = &cereal::Event::Builder::setWideRoadEncodeIdx,
  .init_encode_data_func = &cereal::Event::Builder::initWideRoadEncodeData,
};
const EncoderInfo main_driver_encoder_info = {
   .publish_name = "driverEncodeData",
  .filename = "dcamera.hevc",
  .record = Params().getBool("RecordFront"),
  .get_encode_data_func = &cereal::Event::Reader::getDriverEncodeData,
  .set_encode_idx_func = &cereal::Event::Builder::setDriverEncodeIdx,
  .init_encode_data_func = &cereal::Event::Builder::initDriverEncodeData,
};

const EncoderInfo qcam_encoder_info = {
  .publish_name = "qRoadEncodeData",
  .filename = "qcamera.ts",
  .bitrate = 256000,
  .encode_type = cereal::EncodeIndex::Type::QCAMERA_H264,
  .frame_width = 526,
  .frame_height = 330,
  .get_encode_data_func = &cereal::Event::Reader::getQRoadEncodeData,
  .set_encode_idx_func = &cereal::Event::Builder::setQRoadEncodeIdx,
  .init_encode_data_func = &cereal::Event::Builder::initQRoadEncodeData,
};


const LogCameraInfo road_camera_info{
    .thread_name = "road_cam_encoder",
    .type = RoadCam,
    .stream_type = VISION_STREAM_ROAD,
    .encoder_infos = {main_road_encoder_info, qcam_encoder_info}
    };

const LogCameraInfo wide_road_camera_info{
    .thread_name = "wide_road_cam_encoder",
    .type = WideRoadCam,
    .stream_type = VISION_STREAM_WIDE_ROAD,
   .encoder_infos = {main_wide_road_encoder_info}
    };

const LogCameraInfo driver_camera_info{
    .thread_name = "driver_cam_encoder",
    .type = DriverCam,
    .stream_type = VISION_STREAM_DRIVER,
    .encoder_infos = {main_driver_encoder_info}
    };

const LogCameraInfo cameras_logged[] = {road_camera_info, wide_road_camera_info, driver_camera_info};
