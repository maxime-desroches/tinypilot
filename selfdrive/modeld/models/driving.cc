
#include <string.h>
#include <assert.h>
#include <fcntl.h>
#include <unistd.h>
#include <eigen3/Eigen/Dense>

#include "common/timing.h"
#include "common/params.h"
#include "driving.h"

#define PATH_IDX 0
#define LL_IDX PATH_IDX + MODEL_PATH_DISTANCE*2 + 1
#define RL_IDX LL_IDX + MODEL_PATH_DISTANCE*2 + 2
#define LEAD_IDX RL_IDX + MODEL_PATH_DISTANCE*2 + 2
#define LONG_X_IDX LEAD_IDX + MDN_GROUP_SIZE*LEAD_MDN_N + SELECTION
#define LONG_V_IDX LONG_X_IDX + TIME_DISTANCE*2
#define LONG_A_IDX LONG_V_IDX + TIME_DISTANCE*2
#define DESIRE_STATE_IDX LONG_A_IDX + TIME_DISTANCE*2
#define META_IDX DESIRE_STATE_IDX + DESIRE_LEN
#define POSE_IDX META_IDX + OTHER_META_SIZE + DESIRE_PRED_SIZE
#define OUTPUT_SIZE  POSE_IDX + POSE_SIZE
#ifdef TEMPORAL
  #define TEMPORAL_SIZE 512
#else
  #define TEMPORAL_SIZE 0
#endif

// #define DUMP_YUV

Eigen::Matrix<float, MODEL_PATH_DISTANCE, POLYFIT_DEGREE - 1> vander;

void model_init(ModelState* s, cl::Device &device, cl::Context &ctx, int temporal) {
  s->frame.init(ctx, device);

  const int output_size = OUTPUT_SIZE + TEMPORAL_SIZE;
  s->output = std::make_unique<float[]>(output_size);

  s->m = std::make_unique<DefaultRunModel>("../../models/supercombo.dlc", s->output.get(), output_size, USE_GPU_RUNTIME);

#ifdef TEMPORAL
  assert(temporal);
  s->m->addRecurrent(&s->output[OUTPUT_SIZE], TEMPORAL_SIZE);
#endif

#ifdef DESIRE
  s->prev_desire = std::make_unique<float[]>(DESIRE_LEN);
  s->pulse_desire = std::make_unique<float[]>(DESIRE_LEN);
  s->m->addDesire(s->pulse_desire.get(), DESIRE_LEN);
#endif

#ifdef TRAFFIC_CONVENTION
  s->traffic_convention = std::make_unique<float[]>(TRAFFIC_CONVENTION_LEN);
  s->m->addTrafficConvention(s->traffic_convention.get(), TRAFFIC_CONVENTION_LEN);

  std::vector<char> result = read_db_bytes("IsRHD");
  if (result.size() > 0) {
    bool is_rhd = result[0] == '1';
    if (is_rhd) {
      s->traffic_convention[1] = 1.0;
    } else {
      s->traffic_convention[0] = 1.0;
    }
  }
#endif

  // Build Vandermonde matrix
  for(int i = 0; i < MODEL_PATH_DISTANCE; i++) {
    for(int j = 0; j < POLYFIT_DEGREE - 1; j++) {
      vander(i, j) = pow(i, POLYFIT_DEGREE-j-1);
    }
  }
}

ModelDataRaw model_eval_frame(ModelState *s, cl::Buffer &yuv_cl, int width, int height,
                              mat3 transform, void *sock, float *desire_in) {
#ifdef DESIRE
  if (desire_in != NULL) {
    for (int i = 0; i < DESIRE_LEN; i++) {
      // Model decides when action is completed
      // so desire input is just a pulse triggered on rising edge
      if (desire_in[i] - s->prev_desire[i] > .99) {
        s->pulse_desire[i] = desire_in[i];
      } else {
        s->pulse_desire[i] = 0.0;
      }
      s->prev_desire[i] = desire_in[i];
    }
  }
#endif


  //for (int i = 0; i < OUTPUT_SIZE + TEMPORAL_SIZE; i++) { printf("%f ", s->output[i]); } printf("\n");
  s->frame.prepare(yuv_cl, width, height, transform);
  s->m->execute(s->frame.getFrame(), s->frame.getFrameSize());

  #ifdef DUMP_YUV
    FILE *dump_yuv_file = fopen("/sdcard/dump.yuv", "wb");
    fwrite(new_frame_buf, MODEL_HEIGHT*MODEL_WIDTH*3/2, sizeof(float), dump_yuv_file);
    fclose(dump_yuv_file);
    assert(1==2);
  #endif

  // net outputs
  ModelDataRaw net_outputs;
  net_outputs.path = &s->output[PATH_IDX];
  net_outputs.left_lane = &s->output[LL_IDX];
  net_outputs.right_lane = &s->output[RL_IDX];
  net_outputs.lead = &s->output[LEAD_IDX];
  net_outputs.long_x = &s->output[LONG_X_IDX];
  net_outputs.long_v = &s->output[LONG_V_IDX];
  net_outputs.long_a = &s->output[LONG_A_IDX];
  net_outputs.meta = &s->output[DESIRE_STATE_IDX];
  net_outputs.pose = &s->output[POSE_IDX];
  return net_outputs;
}

void model_free(ModelState* s) {}

void poly_fit(float *in_pts, float *in_stds, float *out, int valid_len) {
  // References to inputs
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 1> > pts(in_pts, valid_len);
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 1> > std(in_stds, valid_len);
  Eigen::Map<Eigen::Matrix<float, POLYFIT_DEGREE - 1, 1> > p(out, POLYFIT_DEGREE - 1);

  float y0 = pts[0];
  pts = pts.array() - y0;

  // Build Least Squares equations
  Eigen::Matrix<float, Eigen::Dynamic, POLYFIT_DEGREE - 1> lhs = vander.topRows(valid_len).array().colwise() / std.array();
  Eigen::Matrix<float, Eigen::Dynamic, 1> rhs = pts.array() / std.array();

  // Improve numerical stability
  Eigen::Matrix<float, POLYFIT_DEGREE - 1, 1> scale = 1. / (lhs.array()*lhs.array()).sqrt().colwise().sum();
  lhs = lhs * scale.asDiagonal();

  // Solve inplace
  p = lhs.colPivHouseholderQr().solve(rhs);

  // Apply scale to output
  p = p.transpose() * scale.asDiagonal();
  out[3] = y0;
}

void fill_path(cereal::ModelData::PathData::Builder path, const float * data, bool has_prob, const float offset) {
  float points_arr[MODEL_PATH_DISTANCE];
  float stds_arr[MODEL_PATH_DISTANCE];
  float poly_arr[POLYFIT_DEGREE];
  float std;
  float prob;
  float valid_len;

  // clamp to 5 and 192
  valid_len = fmin(192, fmax(5, data[MODEL_PATH_DISTANCE*2]));
  for (int i=0; i<MODEL_PATH_DISTANCE; i++) {
    points_arr[i] = data[i] + offset;
    stds_arr[i] = softplus(data[MODEL_PATH_DISTANCE + i]) + 1e-6;
  }
  if (has_prob) {
    prob =  sigmoid(data[MODEL_PATH_DISTANCE*2 + 1]);
  } else {
    prob = 1.0;
  }
  std = softplus(data[MODEL_PATH_DISTANCE]) + 1e-6;
  poly_fit(points_arr, stds_arr, poly_arr, valid_len);

  if (std::getenv("DEBUG")){
    kj::ArrayPtr<const float> stds(&stds_arr[0], ARRAYSIZE(stds_arr));
    path.setStds(stds);

    kj::ArrayPtr<const float> points(&points_arr[0], ARRAYSIZE(points_arr));
    path.setPoints(points);
  }

  kj::ArrayPtr<const float> poly(&poly_arr[0], ARRAYSIZE(poly_arr));
  path.setPoly(poly);
  path.setProb(prob);
  path.setStd(std);
  path.setValidLen(valid_len);
}

void fill_lead(cereal::ModelData::LeadData::Builder lead, const float * data, int mdn_max_idx, int t_offset) {
  const double x_scale = 10.0;
  const double y_scale = 10.0;

  lead.setProb(sigmoid(data[LEAD_MDN_N*MDN_GROUP_SIZE + t_offset]));
  lead.setDist(x_scale * data[mdn_max_idx*MDN_GROUP_SIZE]);
  lead.setStd(x_scale * softplus(data[mdn_max_idx*MDN_GROUP_SIZE + MDN_VALS]));
  lead.setRelY(y_scale * data[mdn_max_idx*MDN_GROUP_SIZE + 1]);
  lead.setRelYStd(y_scale * softplus(data[mdn_max_idx*MDN_GROUP_SIZE + MDN_VALS + 1]));
  lead.setRelVel(data[mdn_max_idx*MDN_GROUP_SIZE + 2]);
  lead.setRelVelStd(softplus(data[mdn_max_idx*MDN_GROUP_SIZE + MDN_VALS + 2]));
  lead.setRelA(data[mdn_max_idx*MDN_GROUP_SIZE + 3]);
  lead.setRelAStd(softplus(data[mdn_max_idx*MDN_GROUP_SIZE + MDN_VALS + 3]));
}

void fill_meta(cereal::ModelData::MetaData::Builder meta, const float * meta_data) {
  kj::ArrayPtr<const float> desire_state(&meta_data[0], DESIRE_LEN);
  meta.setDesireState(desire_state);
  meta.setEngagedProb(meta_data[DESIRE_LEN]);
  meta.setGasDisengageProb(meta_data[DESIRE_LEN + 1]);
  meta.setBrakeDisengageProb(meta_data[DESIRE_LEN + 2]);
  meta.setSteerOverrideProb(meta_data[DESIRE_LEN + 3]);
  kj::ArrayPtr<const float> desire_pred(&meta_data[DESIRE_LEN + OTHER_META_SIZE], DESIRE_PRED_SIZE);
  meta.setDesirePrediction(desire_pred);
}

void fill_longi(cereal::ModelData::LongitudinalData::Builder longi, const float * long_x_data, const float * long_v_data, const float * long_a_data) {
  // just doing 10 vals, 1 every sec for now
  float dist_arr[TIME_DISTANCE/10];
  float speed_arr[TIME_DISTANCE/10];
  float accel_arr[TIME_DISTANCE/10];
  for (int i=0; i<TIME_DISTANCE/10; i++) {
    dist_arr[i] = long_x_data[i*10];
    speed_arr[i] = long_v_data[i*10];
    accel_arr[i] = long_a_data[i*10];
  }
  kj::ArrayPtr<const float> dist(&dist_arr[0], ARRAYSIZE(dist_arr));
  longi.setDistances(dist);
  kj::ArrayPtr<const float> speed(&speed_arr[0], ARRAYSIZE(speed_arr));
  longi.setSpeeds(speed);
  kj::ArrayPtr<const float> accel(&accel_arr[0], ARRAYSIZE(accel_arr));
  longi.setAccelerations(accel);
}

void model_publish(PubMaster &pm, uint32_t frame_id,
                   const ModelDataRaw &net_outputs, uint64_t timestamp_eof) {
  // make msg
  capnp::MallocMessageBuilder msg;
  cereal::Event::Builder event = msg.initRoot<cereal::Event>();
  event.setLogMonoTime(nanos_since_boot());

  auto framed = event.initModel();
  framed.setFrameId(frame_id);
  framed.setTimestampEof(timestamp_eof);

  auto lpath = framed.initPath();
  fill_path(lpath, net_outputs.path, false, 0);
  auto left_lane = framed.initLeftLane();
  fill_path(left_lane, net_outputs.left_lane, true, 1.8);
  auto right_lane = framed.initRightLane();
  fill_path(right_lane, net_outputs.right_lane, true, -1.8);
  auto longi = framed.initLongitudinal();
  fill_longi(longi, net_outputs.long_x, net_outputs.long_v, net_outputs.long_a);

  // Find the distribution that corresponds to the current lead
  int mdn_max_idx = 0;
  int t_offset = 0;
  for (int i=1; i<LEAD_MDN_N; i++) {
    if (net_outputs.lead[i*MDN_GROUP_SIZE + 8 + t_offset] > net_outputs.lead[mdn_max_idx*MDN_GROUP_SIZE + 8 + t_offset]) {
      mdn_max_idx = i;
    }
  }
  auto lead = framed.initLead();
  fill_lead(lead, net_outputs.lead, mdn_max_idx, t_offset);
  // Find the distribution that corresponds to the lead in 2s
  mdn_max_idx = 0;
  t_offset = 1;
  for (int i=1; i<LEAD_MDN_N; i++) {
    if (net_outputs.lead[i*MDN_GROUP_SIZE + 8 + t_offset] > net_outputs.lead[mdn_max_idx*MDN_GROUP_SIZE + 8 + t_offset]) {
      mdn_max_idx = i;
    }
  }
  auto lead_future = framed.initLeadFuture();
  fill_lead(lead_future, net_outputs.lead, mdn_max_idx, t_offset);


  auto meta = framed.initMeta();
  fill_meta(meta, net_outputs.meta);

  pm.send("model", msg);
}

void posenet_publish(PubMaster &pm, uint32_t frame_id,
                   const ModelDataRaw &net_outputs, uint64_t timestamp_eof) {
  capnp::MallocMessageBuilder msg;
  cereal::Event::Builder event = msg.initRoot<cereal::Event>();
  event.setLogMonoTime(nanos_since_boot());

  float trans_arr[3];
  float trans_std_arr[3];
  float rot_arr[3];
  float rot_std_arr[3];

  for (int i =0; i < 3; i++) {
    trans_arr[i] = net_outputs.pose[i];
    trans_std_arr[i] = softplus(net_outputs.pose[6 + i]) + 1e-6;

    rot_arr[i] = M_PI * net_outputs.pose[3 + i] / 180.0;
    rot_std_arr[i] = M_PI * (softplus(net_outputs.pose[9 + i]) + 1e-6) / 180.0;
  }

  auto posenetd = event.initCameraOdometry();
  kj::ArrayPtr<const float> trans_vs(&trans_arr[0], 3);
  posenetd.setTrans(trans_vs);
  kj::ArrayPtr<const float> rot_vs(&rot_arr[0], 3);
  posenetd.setRot(rot_vs);
  kj::ArrayPtr<const float> trans_std_vs(&trans_std_arr[0], 3);
  posenetd.setTransStd(trans_std_vs);
  kj::ArrayPtr<const float> rot_std_vs(&rot_std_arr[0], 3);
  posenetd.setRotStd(rot_std_vs);

  posenetd.setTimestampEof(timestamp_eof);
  posenetd.setFrameId(frame_id);

  pm.send("cameraOdometry", msg);
}
