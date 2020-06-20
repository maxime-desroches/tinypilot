#pragma once
// gate this here
#define TEMPORAL
#define DESIRE
#define TRAFFIC_CONVENTION


#include "common/mat.h"
#include "common/util.h"

#include "commonmodel.h"
#include "runners/run.h"

#include <czmq.h>
#include "messaging.hpp"

#define MODEL_WIDTH 512
#define MODEL_HEIGHT 256
#define MODEL_FRAME_SIZE MODEL_WIDTH * MODEL_HEIGHT * 3 / 2
#define MODEL_NAME "supercombo_dlc"

#define MODEL_PATH_DISTANCE 192
#define POLYFIT_DEGREE 4
#define SPEED_PERCENTILES 10
#define DESIRE_LEN 8
#define TRAFFIC_CONVENTION_LEN 2
#define DESIRE_PRED_SIZE 32
#define OTHER_META_SIZE 4
#define LEAD_MDN_N 5 // probs for 5 groups
#define MDN_VALS 4 // output xyva for each lead group
#define SELECTION 3 //output 3 group (lead now, in 2s and 6s)
#define MDN_GROUP_SIZE 11
#define TIME_DISTANCE 100
#define POSE_SIZE 12

struct ModelDataRaw {
    float *path;
    float *left_lane;
    float *right_lane;
    float *lead;
    float *long_x;
    float *long_v;
    float *long_a;
    float *desire_state;
    float *meta;
    float *pose;
  };


typedef struct ModelState {
  ModelFrame frame;
  float *output;
  float *input_frames;
  RunModel *m;
#ifdef DESIRE
  float *prev_desire;
  float *pulse_desire;
#endif
#ifdef TRAFFIC_CONVENTION
  float *traffic_convention;
#endif
} ModelState;

void model_init(ModelState* s, cl::Context &ctx, cl::Device &device, int temporal);
ModelDataRaw model_eval_frame(ModelState* s,  cl::Buffer &yuv_cl, int width, int height,
                           mat3 transform, void* sock, float *desire_in);
void model_free(ModelState* s);
void poly_fit(float *in_pts, float *in_stds, float *out);

void model_publish(PubMaster &pm, uint32_t frame_id,
                   const ModelDataRaw &data, uint64_t timestamp_eof);
void posenet_publish(PubMaster &pm, uint32_t frame_id,
                   const ModelDataRaw &data, uint64_t timestamp_eof);
