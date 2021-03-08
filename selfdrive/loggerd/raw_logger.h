#pragma once

#include <cstdio>
#include <cstdlib>

#include <string>
#include <vector>
#include <condition_variable>

extern "C" {
#include <libavutil/imgutils.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

#include "encoder.h"

class RawLogger : public VideoEncoder {
public:
  RawLogger(const char* filename, int width, int height, int fps,
            int bitrate, bool h265, bool downscale);
  ~RawLogger();
  int encode_frame(const uint8_t *y_ptr, const uint8_t *u_ptr, const uint8_t *v_ptr,
                   int in_width, int in_height, uint64_t ts);
  void encoder_open(const char* path);
  void encoder_close();

private:
  const char* filename;
  int fps;
  int counter = 0;
  bool is_open = false;

  std::string vid_path, lock_path;

  AVCodec *codec = NULL;
  AVCodecContext *codec_ctx = NULL;

  AVStream *stream = NULL;
  AVFormatContext *format_ctx = NULL;

  AVFrame *frame = NULL;
};
