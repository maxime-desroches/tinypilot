#pragma once

#include <string>

extern "C" {
#include <libavformat/avformat.h>
}

class VideoWriter {
public:
  VideoWriter(const char *path, const char *filename, bool remuxing, int width, int height, int fps, bool h265);
  void write(uint8_t *data, int len, long long timestamp, bool codecconfig, bool keyframe);
  ~VideoWriter();
private:
  std::string vid_path, lock_path;

  FILE *of = nullptr;

  AVFormatContext *ofmt_ctx;
  AVCodecContext *codec_ctx;
  AVStream *out_stream;
  bool remuxing;

  bool wrote_codec_config;
};