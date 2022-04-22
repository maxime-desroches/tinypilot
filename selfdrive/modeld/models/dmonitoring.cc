#include <cstring>

#include "libyuv.h"

#include "selfdrive/common/mat.h"
#include "selfdrive/common/modeldata.h"
#include "selfdrive/common/params.h"
#include "selfdrive/common/timing.h"
#include "selfdrive/hardware/hw.h"

#include "selfdrive/modeld/models/dmonitoring.h"

constexpr int MODEL_WIDTH = 1440;
constexpr int MODEL_HEIGHT = 960;

template <class T>
static inline T *get_buffer(std::vector<T> &buf, const size_t size) {
  if (buf.size() < size) buf.resize(size);
  return buf.data();
}

void dmonitoring_init(DMonitoringModelState* s) {
  s->is_rhd = Params().getBool("IsRHD");

#ifdef USE_ONNX_MODEL
  s->m = new ONNXModel("../../models/dmonitoring_model.onnx", &s->output[0], OUTPUT_SIZE, USE_DSP_RUNTIME);
#else
  s->m = new SNPEModel("../../models/dmonitoring_model_q.dlc", &s->output[0], OUTPUT_SIZE, USE_DSP_RUNTIME);
#endif

  s->m->addCalib(s->calib, CALIB_LEN);
}

static inline auto get_yuv_buf(std::vector<uint8_t> &buf, const int width, int height) {
  uint8_t *y = get_buffer(buf, width * height * 3 / 2);
  uint8_t *u = y + width * height;
  uint8_t *v = u + (width /2) * (height / 2);
  return std::make_tuple(y, u, v);
}

DMonitoringResult dmonitoring_eval_frame(DMonitoringModelState* s, void* stream_buf, int width, int height, float *calib) {
  // int v_off = height - MODEL_HEIGHT;
  // int h_off = (width - MODEL_WIDTH) / 2;
  int yuv_buf_len = (MODEL_WIDTH/2) * (MODEL_HEIGHT/2) * 6; // Y|u|v, frame2tensor done in dsp

  // uint8_t *raw_buf = (uint8_t *) stream_buf;
  auto [cropped_y, cropped_u, cropped_v] = get_yuv_buf(s->cropped_buf, MODEL_WIDTH, MODEL_HEIGHT);
  float *net_input_buf = get_buffer(s->net_input_buf, yuv_buf_len);

  /*
  libyuv::ConvertToI420(raw_buf, (width/2)*(height/2)*6,
                        cropped_y, MODEL_WIDTH,
                        cropped_u, MODEL_WIDTH/2,
                        cropped_v, MODEL_WIDTH/2,
                        h_off, v_off,
                        width, height,
                        MODEL_WIDTH, MODEL_HEIGHT,
                        libyuv::kRotate0,
                        libyuv::FOURCC_I420);
  */

  // snpe UserBufferEncodingUnsigned8Bit doesn't work
  // fast float conversion instead, also scales to 0-1
  libyuv::ByteToFloat(cropped_y, net_input_buf, 0.003921569f, yuv_buf_len);

  // printf("preprocess completed. %d \n", yuv_buf_len);
  // FILE *dump_yuv_file = fopen("/tmp/rawdump.yuv", "wb");
  // fwrite(net_input_buf, yuv_buf_len, sizeof(float), dump_yuv_file);
  // fclose(dump_yuv_file);

  // # testing:
  // dat = np.fromfile('/tmp/rawdump.yuv', dtype=np.float32)
  // dat = dat.reshape(1,6,320,512) * 128. + 128.
  // frame = tensor_to_frames(dat)[0]
  // frame = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB_I420)

  double t1 = millis_since_boot();
  s->m->addImage(net_input_buf, yuv_buf_len);
  for (int i = 0; i < CALIB_LEN; i++) {
    s->calib[i] = calib[i];
  }
  s->m->execute();
  double t2 = millis_since_boot();

  DMonitoringResult ret = {0};
  for (int i = 0; i < 3; ++i) {
    ret.face_orientation[i] = s->output[i] * REG_SCALE;
    ret.face_orientation_meta[i] = exp(s->output[6 + i]);
  }
  for (int i = 0; i < 2; ++i) {
    ret.face_position[i] = s->output[3 + i] * REG_SCALE;
    ret.face_position_meta[i] = exp(s->output[9 + i]);
  }
  for (int i = 0; i < 4; ++i) {
    ret.ready_prob[i] = sigmoid(s->output[39 + i]);
  }
  for (int i = 0; i < 2; ++i) {
    ret.not_ready_prob[i] = sigmoid(s->output[43 + i]);
  }
  ret.face_prob = sigmoid(s->output[12]);
  ret.left_eye_prob = sigmoid(s->output[21]);
  ret.right_eye_prob = sigmoid(s->output[30]);
  ret.left_blink_prob = sigmoid(s->output[31]);
  ret.right_blink_prob = sigmoid(s->output[32]);
  ret.sg_prob = sigmoid(s->output[33]);
  ret.poor_vision = sigmoid(s->output[34]);
  ret.partial_face = sigmoid(s->output[35]);
  ret.distracted_pose = sigmoid(s->output[36]);
  ret.distracted_eyes = sigmoid(s->output[37]);
  ret.occluded_prob = sigmoid(s->output[38]);
  ret.dsp_execution_time = (t2 - t1) / 1000.;
  return ret;
}

void dmonitoring_publish(PubMaster &pm, uint32_t frame_id, const DMonitoringResult &res, float execution_time, kj::ArrayPtr<const float> raw_pred) {
  // make msg
  MessageBuilder msg;
  auto framed = msg.initEvent().initDriverState();
  framed.setFrameId(frame_id);
  framed.setModelExecutionTime(execution_time);
  framed.setDspExecutionTime(res.dsp_execution_time);

  framed.setFaceOrientation(res.face_orientation);
  framed.setFaceOrientationStd(res.face_orientation_meta);
  framed.setFacePosition(res.face_position);
  framed.setFacePositionStd(res.face_position_meta);
  framed.setFaceProb(res.face_prob);
  framed.setLeftEyeProb(res.left_eye_prob);
  framed.setRightEyeProb(res.right_eye_prob);
  framed.setLeftBlinkProb(res.left_blink_prob);
  framed.setRightBlinkProb(res.right_blink_prob);
  framed.setSunglassesProb(res.sg_prob);
  framed.setPoorVision(res.poor_vision);
  framed.setPartialFace(res.partial_face);
  framed.setDistractedPose(res.distracted_pose);
  framed.setDistractedEyes(res.distracted_eyes);
  framed.setOccludedProb(res.occluded_prob);
  framed.setReadyProb(res.ready_prob);
  framed.setNotReadyProb(res.not_ready_prob);
  if (send_raw_pred) {
    framed.setRawPredictions(raw_pred.asBytes());
  }

  pm.send("driverState", msg);
}

void dmonitoring_free(DMonitoringModelState* s) {
  delete s->m;
}
