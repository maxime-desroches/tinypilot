#pragma once

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>
#include <OMX_Component.h>
#include <libavformat/avformat.h>

#include "common/cqueue.h"
#include "common/visionipc.h"
#include "camerad/cameras/camera_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct EncoderState {
  pthread_mutex_t lock;
  int width, height;
  char lock_path[4096];
  bool open;
  bool dirty;
  int counter;

  LogCameraInfo camera_info;
  FILE *of;

  size_t codec_config_len;
  uint8_t *codec_config;
  bool wrote_codec_config;

  pthread_mutex_t state_lock;
  pthread_cond_t state_cv;
  OMX_STATETYPE state;

  OMX_HANDLETYPE handle;

  int num_in_bufs;
  OMX_BUFFERHEADERTYPE** in_buf_headers;

  int num_out_bufs;
  OMX_BUFFERHEADERTYPE** out_buf_headers;

  Queue free_in;
  Queue done_out;

  AVFormatContext *ofmt_ctx;
  AVCodecContext *codec_ctx;
  AVStream *out_stream;
  bool remuxing;
} EncoderState;

void encoder_init(EncoderState *s, LogCameraInfo *camera_info, int width, int height);
int encoder_encode_frame(EncoderState *s, const uint8_t *y_ptr, const uint8_t *u_ptr, const uint8_t *v_ptr, VIPCBufExtra *extra);
void encoder_open(EncoderState *s, const char* path);
void encoder_rotate(EncoderState *s, const char* new_path);
void encoder_close(EncoderState *s);
void encoder_destroy(EncoderState *s);

#ifdef __cplusplus
}
#endif
