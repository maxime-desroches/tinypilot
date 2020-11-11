#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <assert.h>
#include <unistd.h>
#include <fcntl.h>
#include <math.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <atomic>
#include <algorithm>

#include <linux/media.h>

#include <cutils/properties.h>

#include <pthread.h>
#include <capnp/serialize.h>
#include "msmb_isp.h"
#include "msmb_ispif.h"
#include "msmb_camera.h"
#include "msm_cam_sensor.h"

#include "common/util.h"
#include "common/timing.h"
#include "common/swaglog.h"
#include "common/params.h"
#include "clutil.h"

#include "cereal/gen/cpp/log.capnp.h"

#include "sensor_i2c.h"

#include "camera_qcom.h"


extern volatile sig_atomic_t do_exit;

// global var for AE/AF ops
std::atomic<CameraExpInfo> rear_exp{{0}};
std::atomic<CameraExpInfo> front_exp{{0}};

CameraInfo cameras_supported[CAMERA_ID_MAX] = {
  [CAMERA_ID_IMX298] = {
    .frame_width = 2328,
    .frame_height = 1748,
    .frame_stride = 2912,
    .bayer = true,
    .bayer_flip = 0,
    .hdr = true
  },
  [CAMERA_ID_IMX179] = {
    .frame_width = 3280,
    .frame_height = 2464,
    .frame_stride = 4104,
    .bayer = true,
    .bayer_flip = 0,
    .hdr = false
  },
  [CAMERA_ID_S5K3P8SP] = {
    .frame_width = 2304,
    .frame_height = 1728,
    .frame_stride = 2880,
    .bayer = true,
    .bayer_flip = 1,
    .hdr = false
  },
  [CAMERA_ID_OV8865] = {
    .frame_width = 1632,
    .frame_height = 1224,
    .frame_stride = 2040, // seems right
    .bayer = true,
    .bayer_flip = 3,
    .hdr = false
  },
  // this exists to get the kernel to build for the LeEco in release
  [CAMERA_ID_IMX298_FLIPPED] = {
    .frame_width = 2328,
    .frame_height = 1748,
    .frame_stride = 2912,
    .bayer = true,
    .bayer_flip = 3,
    .hdr = true
  },
  [CAMERA_ID_OV10640] = {
    .frame_width = 1280,
    .frame_height = 1080,
    .frame_stride = 2040,
    .bayer = true,
    .bayer_flip = 0,
    .hdr = true
  },
};

static void camera_release_buffer(void* cookie, int buf_idx) {
  CameraState *s = (CameraState *)cookie;
  // printf("camera_release_buffer %d\n", buf_idx);
  s->ss[0].qbuf_info[buf_idx].dirty_buf = 1;
  ioctl(s->isp_fd, VIDIOC_MSM_ISP_ENQUEUE_BUF, &s->ss[0].qbuf_info[buf_idx]);
}

static void camera_init(CameraState *s, int camera_id, int camera_num,
                        uint32_t pixel_clock, uint32_t line_length_pclk,
                        unsigned int max_gain, unsigned int fps, cl_device_id device_id, cl_context ctx) {
  s->camera_num = camera_num;
  s->camera_id = camera_id;

  assert(camera_id < ARRAYSIZE(cameras_supported));
  s->ci = cameras_supported[camera_id];
  assert(s->ci.frame_width != 0);

  s->pixel_clock = pixel_clock;
  s->line_length_pclk = line_length_pclk;
  s->max_gain = max_gain;
  s->fps = fps;

  s->self_recover = 0;

  s->buf.init(device_id, ctx, s, FRAME_BUF_COUNT, "frame", camera_release_buffer);

  pthread_mutex_init(&s->frame_info_lock, NULL);
}


int sensor_write_regs(CameraState *s, struct msm_camera_i2c_reg_array* arr, size_t size, msm_camera_i2c_data_type data_type) {
  struct msm_camera_i2c_reg_setting out_settings = {
    .reg_setting = arr,
    .size = (uint16_t)size,
    .addr_type = MSM_CAMERA_I2C_WORD_ADDR,
    .data_type = data_type,
    .delay = 0,
  };
  struct sensorb_cfg_data cfg_data = {0};
  cfg_data.cfgtype = CFG_WRITE_I2C_ARRAY;
  cfg_data.cfg.setting = &out_settings;
  return ioctl(s->sensor_fd, VIDIOC_MSM_SENSOR_CFG, &cfg_data);
}

static int imx298_apply_exposure(CameraState *s, int gain, int integ_lines, int frame_length) {
  int err;

  int analog_gain = std::min(gain, 448);

  if (gain > 448) {
    s->digital_gain = (512.0/(512-(gain))) / 8.0;
  } else {
    s->digital_gain = 1.0;
  }

  //printf("%5d/%5d %5d %f\n", s->cur_integ_lines, s->cur_frame_length, analog_gain, s->digital_gain);

  struct msm_camera_i2c_reg_array reg_array[] = {
    // REG_HOLD
    {0x104,0x1,0},
    {0x3002,0x0,0}, // long autoexposure off

    // FRM_LENGTH
    {0x340, (uint16_t)(frame_length >> 8), 0}, {0x341, (uint16_t)(frame_length & 0xff), 0},
    // INTEG_TIME aka coarse_int_time_addr aka shutter speed
    {0x202, (uint16_t)(integ_lines >> 8), 0}, {0x203, (uint16_t)(integ_lines & 0xff),0},
    // global_gain_addr
    // if you assume 1x gain is 32, 448 is 14x gain, aka 2^14=16384
    {0x204, (uint16_t)(analog_gain >> 8), 0}, {0x205, (uint16_t)(analog_gain & 0xff),0},

    // digital gain for colors: gain_greenR, gain_red, gain_blue, gain_greenB
    /*{0x20e, digital_gain_gr >> 8, 0}, {0x20f,digital_gain_gr & 0xFF,0},
    {0x210, digital_gain_r >> 8, 0}, {0x211,digital_gain_r & 0xFF,0},
    {0x212, digital_gain_b >> 8, 0}, {0x213,digital_gain_b & 0xFF,0},
    {0x214, digital_gain_gb >> 8, 0}, {0x215,digital_gain_gb & 0xFF,0},*/

    // REG_HOLD
    {0x104,0x0,0},
  };

  err = sensor_write_regs(s, reg_array, ARRAYSIZE(reg_array), MSM_CAMERA_I2C_BYTE_DATA);
  if (err != 0) {
    LOGE("apply_exposure err %d", err);
  }
  return err;
}

static int ov8865_apply_exposure(CameraState *s, int gain, int integ_lines, int frame_length) {
  //printf("front camera: %d %d %d\n", gain, integ_lines, frame_length);
  int err, coarse_gain_bitmap, fine_gain_bitmap;

  // get bitmaps from iso
  static const int gains[] = {0, 100, 200, 400, 800};
  int i;
  for (i = 1; i < ARRAYSIZE(gains); i++) {
    if (gain >= gains[i - 1] && gain < gains[i])
      break;
  }
  int coarse_gain = i - 1;
  float fine_gain = (gain - gains[coarse_gain])/(float)(gains[coarse_gain+1]-gains[coarse_gain]);
  coarse_gain_bitmap = (1 << coarse_gain) - 1;
  fine_gain_bitmap = ((int)(16*fine_gain) << 3) + 128; // 7th is always 1, 0-2nd are always 0

  integ_lines *= 16; // The exposure value in reg is in 16ths of a line

  struct msm_camera_i2c_reg_array reg_array[] = {
    //{0x104,0x1,0},

    // FRM_LENGTH
    {0x380e, (uint16_t)(frame_length >> 8), 0}, {0x380f, (uint16_t)(frame_length & 0xff), 0},
    // AEC EXPO
    {0x3500, (uint16_t)(integ_lines >> 16), 0}, {0x3501, (uint16_t)(integ_lines >> 8), 0}, {0x3502, (uint16_t)(integ_lines & 0xff),0},
    // AEC MANUAL
    {0x3503, 0x4, 0},
    // AEC GAIN
    {0x3508, (uint16_t)(coarse_gain_bitmap), 0}, {0x3509, (uint16_t)(fine_gain_bitmap), 0},

    //{0x104,0x0,0},
  };
  err = sensor_write_regs(s, reg_array, ARRAYSIZE(reg_array), MSM_CAMERA_I2C_BYTE_DATA);
  if (err != 0) {
    LOGE("apply_exposure err %d", err);
  }
  return err;
}

static int imx179_s5k3p8sp_apply_exposure(CameraState *s, int gain, int integ_lines, int frame_length) {
  //printf("front camera: %d %d %d\n", gain, integ_lines, frame_length);
  int err;

  struct msm_camera_i2c_reg_array reg_array[] = {
    {0x104,0x1,0},

    // FRM_LENGTH
    {0x340, (uint16_t)(frame_length >> 8), 0}, {0x341, (uint16_t)(frame_length & 0xff), 0},
    // coarse_int_time
    {0x202, (uint16_t)(integ_lines >> 8), 0}, {0x203, (uint16_t)(integ_lines & 0xff),0},
    // global_gain
    {0x204, (uint16_t)(gain >> 8), 0}, {0x205, (uint16_t)(gain & 0xff),0},

    {0x104,0x0,0},
  };
  err = sensor_write_regs(s, reg_array, ARRAYSIZE(reg_array), MSM_CAMERA_I2C_BYTE_DATA);
  if (err != 0) {
    LOGE("apply_exposure err %d", err);
  }
  return err;
}

cl_program build_conv_program(cl_device_id device_id, cl_context context, int image_w, int image_h, int filter_size) {
  char args[4096];
  snprintf(args, sizeof(args),
          "-cl-fast-relaxed-math -cl-denorms-are-zero "
          "-DIMAGE_W=%d -DIMAGE_H=%d -DFLIP_RB=%d "
          "-DFILTER_SIZE=%d -DHALF_FILTER_SIZE=%d -DTWICE_HALF_FILTER_SIZE=%d -DHALF_FILTER_SIZE_IMAGE_W=%d",
          image_w, image_h, 1,
          filter_size, filter_size/2, (filter_size/2)*2, (filter_size/2)*image_w);
  return CLU_LOAD_FROM_FILE(context, device_id, "imgproc/conv.cl", args);
}

void cameras_init(MultiCameraState *s, cl_device_id device_id, cl_context ctx) {
  char project_name[1024] = {0};
  property_get("ro.boot.project_name", project_name, "");

  char product_name[1024] = {0};
  property_get("ro.product.name", product_name, "");

  if (strlen(project_name) == 0) {
    LOGD("LePro 3 op system detected");
    s->device = DEVICE_LP3;

    // sensor is flipped in LP3
    // IMAGE_ORIENT = 3
    init_array_imx298[0].reg_data = 3;
    cameras_supported[CAMERA_ID_IMX298].bayer_flip = 3;
  } else if (strcmp(product_name, "OnePlus3") == 0 && strcmp(project_name, "15811") != 0) {
    // no more OP3 support
    s->device = DEVICE_OP3;
    assert(false);
  } else if (strcmp(product_name, "OnePlus3") == 0 && strcmp(project_name, "15811") == 0) {
    // only OP3T support
    s->device = DEVICE_OP3T;
  } else {
    assert(false);
  }

  // 0   = ISO 100
  // 256 = ISO 200
  // 384 = ISO 400
  // 448 = ISO 800
  // 480 = ISO 1600
  // 496 = ISO 3200
  // 504 = ISO 6400, 8x digital gain
  // 508 = ISO 12800, 16x digital gain
  // 510 = ISO 25600, 32x digital gain

  camera_init(&s->rear, CAMERA_ID_IMX298, 0,
              /*pixel_clock=*/600000000, /*line_length_pclk=*/5536,
              /*max_gain=*/510,  //0 (ISO 100)- 448 (ISO 800, max analog gain) - 511 (super noisy)
#ifdef HIGH_FPS
              /*fps*/ 60,
#else
              /*fps*/ 20,
#endif
              device_id, ctx);
  s->rear.apply_exposure = imx298_apply_exposure;

  if (s->device == DEVICE_OP3T) {
    camera_init(&s->front, CAMERA_ID_S5K3P8SP, 1,
                /*pixel_clock=*/560000000, /*line_length_pclk=*/5120,
                /*max_gain=*/510, 10, device_id, ctx);
    s->front.apply_exposure = imx179_s5k3p8sp_apply_exposure;
  } else if (s->device == DEVICE_LP3) {
    camera_init(&s->front, CAMERA_ID_OV8865, 1,
                /*pixel_clock=*/72000000, /*line_length_pclk=*/1602,
                /*max_gain=*/510, 10, device_id, ctx);
    s->front.apply_exposure = ov8865_apply_exposure;
  } else {
    camera_init(&s->front, CAMERA_ID_IMX179, 1,
                /*pixel_clock=*/251200000, /*line_length_pclk=*/3440,
                /*max_gain=*/224, 20, device_id, ctx);
    s->front.apply_exposure = imx179_s5k3p8sp_apply_exposure;
  }

  // assume the device is upside-down (not anymore)
  s->rear.transform = (mat3){{
     1.0,  0.0, 0.0,
     0.0,  1.0, 0.0,
     0.0,  0.0, 1.0,
  }};

  // probably wrong
  s->front.transform = (mat3){{
     1.0,  0.0, 0.0,
     0.0,  1.0, 0.0,
     0.0,  0.0, 1.0,
  }};

  s->rear.device = s->device;
  s->front.device = s->device;

  s->sm_driver = new SubMaster({"driverState"});
  s->sm_sensor = new SubMaster({"sensorEvents"});
  s->pm = new PubMaster({"frame", "frontFrame", "thumbnail"});

  int err;
  const int rgb_width = s->rear.buf.rgb_width;
  const int rgb_height = s->rear.buf.rgb_height;
  for (int i = 0; i < FRAME_BUF_COUNT; i++) {
    // TODO: make lengths correct
    s->focus_bufs[i] = visionbuf_allocate(0xb80);
    s->stats_bufs[i] = visionbuf_allocate(0xb80);
  }
  s->prg_rgb_laplacian = build_conv_program(device_id, ctx, rgb_width/NUM_SEGMENTS_X, rgb_height/NUM_SEGMENTS_Y, 3);
  s->krnl_rgb_laplacian = clCreateKernel(s->prg_rgb_laplacian, "rgb2gray_conv2d", &err);
  assert(err == 0);
  // TODO: Removed CL_MEM_SVM_FINE_GRAIN_BUFFER, confirm it doesn't matter
  s->rgb_conv_roi_cl = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
      rgb_width/NUM_SEGMENTS_X * rgb_height/NUM_SEGMENTS_Y * 3 * sizeof(uint8_t), NULL, NULL);
  s->rgb_conv_result_cl = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
      rgb_width/NUM_SEGMENTS_X * rgb_height/NUM_SEGMENTS_Y * sizeof(int16_t), NULL, NULL);
  s->rgb_conv_filter_cl = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      9 * sizeof(int16_t), (void*)&lapl_conv_krnl, NULL);
  s->conv_cl_localMemSize = ( CONV_LOCAL_WORKSIZE + 2 * (3 / 2) ) * ( CONV_LOCAL_WORKSIZE + 2 * (3 / 2) );
  s->conv_cl_localMemSize *= 3 * sizeof(uint8_t);
  s->conv_cl_globalWorkSize[0] = rgb_width/NUM_SEGMENTS_X;
  s->conv_cl_globalWorkSize[1] = rgb_height/NUM_SEGMENTS_Y;
  s->conv_cl_localWorkSize[0] = CONV_LOCAL_WORKSIZE;
  s->conv_cl_localWorkSize[1] = CONV_LOCAL_WORKSIZE;

  for (int i=0; i<ARRAYSIZE(s->lapres); i++) {s->lapres[i] = 16160;}

  const size_t size = (rgb_width/NUM_SEGMENTS_X)*(rgb_height/NUM_SEGMENTS_Y);
  s->rgb_roi_buf = std::make_unique<uint8_t[]>(size*3);
  s->conv_result = std::make_unique<int16_t[]>(size);
}

static void set_exposure(CameraState *s, float exposure_frac, float gain_frac) {
  int err = 0;

  unsigned int frame_length = s->pixel_clock / s->line_length_pclk / s->fps;

  unsigned int gain = s->cur_gain;
  unsigned int integ_lines = s->cur_integ_lines;

  if (exposure_frac >= 0) {
    exposure_frac = std::clamp(exposure_frac, 2.0f / frame_length, 1.0f);
    integ_lines = frame_length * exposure_frac;

    // See page 79 of the datasheet, this is the max allowed (-1 for phase adjust)
    integ_lines = std::min(integ_lines, frame_length-11);
  }

  if (gain_frac >= 0) {
    // ISO200 is minimum gain
    gain_frac = std::clamp(gain_frac, 1.0f/64, 1.0f);

    // linearize gain response
    // TODO: will be wrong for front camera
    // 0.125 -> 448
    // 0.25  -> 480
    // 0.5   -> 496
    // 1.0   -> 504
    // 512 - 512/(128*gain_frac)
    gain = (s->max_gain/510) * (512 - 512/(256*gain_frac));
  }

  if (gain != s->cur_gain
    || integ_lines != s->cur_integ_lines
    || frame_length != s->cur_frame_length) {

    if (s->apply_exposure == ov8865_apply_exposure) {
      gain = 800 * gain_frac; // ISO
      err = s->apply_exposure(s, gain, integ_lines, frame_length);
    } else if (s->apply_exposure) {
      err = s->apply_exposure(s, gain, integ_lines, frame_length);
    }

    if (err == 0) {
      pthread_mutex_lock(&s->frame_info_lock);
      s->cur_gain = gain;
      s->cur_integ_lines = integ_lines;
      s->cur_frame_length = frame_length;
      pthread_mutex_unlock(&s->frame_info_lock);
    }
  }

  if (err == 0) {
    s->cur_exposure_frac = exposure_frac;
    pthread_mutex_lock(&s->frame_info_lock);
    s->cur_gain_frac = gain_frac;
    pthread_mutex_unlock(&s->frame_info_lock);
  }

  //LOGD("set exposure: %f %f - %d", exposure_frac, gain_frac, err);
}

static void do_autoexposure(CameraState *s, float grey_frac) {
  const float target_grey = 0.3;
  if (s->apply_exposure == ov8865_apply_exposure) {
    // gain limits downstream
    const float gain_frac_min = 0.015625;
    const float gain_frac_max = 1.0;
    // exposure time limits
    unsigned int frame_length = s->pixel_clock / s->line_length_pclk / s->fps;
    const unsigned int exposure_time_min = 16;
    const unsigned int exposure_time_max = frame_length - 11; // copied from set_exposure()

    float cur_gain_frac = s->cur_gain_frac;
    float exposure_factor = pow(1.05, (target_grey - grey_frac) / 0.05);
    if (cur_gain_frac > 0.125 && exposure_factor < 1) {
      cur_gain_frac *= exposure_factor;
    } else if (s->cur_integ_lines * exposure_factor <= exposure_time_max && s->cur_integ_lines * exposure_factor >= exposure_time_min) { // adjust exposure time first
      s->cur_exposure_frac *= exposure_factor;
    } else if (cur_gain_frac * exposure_factor <= gain_frac_max && cur_gain_frac * exposure_factor >= gain_frac_min) {
      cur_gain_frac *= exposure_factor;
    }
    pthread_mutex_lock(&s->frame_info_lock);
    s->cur_gain_frac = cur_gain_frac;
    pthread_mutex_unlock(&s->frame_info_lock);

    set_exposure(s, s->cur_exposure_frac, cur_gain_frac);

  } else { // keep the old for others
    float new_exposure = s->cur_exposure_frac;
    new_exposure *= pow(1.05, (target_grey - grey_frac) / 0.05 );
    //LOGD("diff %f: %f to %f", target_grey - grey_frac, s->cur_exposure_frac, new_exposure);

    float new_gain = s->cur_gain_frac;
    if (new_exposure < 0.10) {
      new_gain *= 0.95;
    } else if (new_exposure > 0.40) {
      new_gain *= 1.05;
    }

    set_exposure(s, new_exposure, new_gain);
  }
}

static uint8_t* get_eeprom(int eeprom_fd, size_t *out_len) {
  int err;

  struct msm_eeprom_cfg_data cfg = {};
  cfg.cfgtype = CFG_EEPROM_GET_CAL_DATA;
  err = ioctl(eeprom_fd, VIDIOC_MSM_EEPROM_CFG, &cfg);
  assert(err >= 0);

  uint32_t num_bytes = cfg.cfg.get_data.num_bytes;
  assert(num_bytes > 100);

  uint8_t* buffer = (uint8_t*)malloc(num_bytes);
  assert(buffer);
  memset(buffer, 0, num_bytes);

  cfg.cfgtype = CFG_EEPROM_READ_CAL_DATA;
  cfg.cfg.read_data.num_bytes = num_bytes;
  cfg.cfg.read_data.dbuffer = buffer;
  err = ioctl(eeprom_fd, VIDIOC_MSM_EEPROM_CFG, &cfg);
  assert(err >= 0);

  *out_len = num_bytes;
  return buffer;
}

static void imx298_ois_calibration(int ois_fd, uint8_t* eeprom) {
  int err;

  const int ois_registers[][2] = {
    // == SET_FADJ_PARAM() == (factory adjustment)

    // Set Hall Current DAC
    {0x8230, *(uint16_t*)(eeprom+0x102)}, //_P_30_ADC_CH0 (CURDAT)

    // Set Hall     PreAmp Offset
    {0x8231, *(uint16_t*)(eeprom+0x104)}, //_P_31_ADC_CH1 (HALOFS_X)
    {0x8232, *(uint16_t*)(eeprom+0x106)}, //_P_32_ADC_CH2 (HALOFS_Y)

    // Set Hall-X/Y PostAmp Offset
    {0x841e, *(uint16_t*)(eeprom+0x108)}, //_M_X_H_ofs
    {0x849e, *(uint16_t*)(eeprom+0x10a)}, //_M_Y_H_ofs

    // Set Residual Offset
    {0x8239, *(uint16_t*)(eeprom+0x10c)}, //_P_39_Ch3_VAL_1 (PSTXOF)
    {0x823b, *(uint16_t*)(eeprom+0x10e)}, //_P_3B_Ch3_VAL_3 (PSTYOF)

    // DIGITAL GYRO OFFSET
    {0x8406, *(uint16_t*)(eeprom+0x110)}, //_M_Kgx00
    {0x8486, *(uint16_t*)(eeprom+0x112)}, //_M_Kgy00
    {0x846a, *(uint16_t*)(eeprom+0x120)}, //_M_TMP_X_
    {0x846b, *(uint16_t*)(eeprom+0x122)}, //_M_TMP_Y_

    // HALLSENSE
    // Set Hall Gain
    {0x8446, *(uint16_t*)(eeprom+0x114)}, //_M_KgxHG
    {0x84c6, *(uint16_t*)(eeprom+0x116)}, //_M_KgyHG
    // Set Cross Talk Canceller
    {0x8470, *(uint16_t*)(eeprom+0x124)}, //_M_KgxH0
    {0x8472, *(uint16_t*)(eeprom+0x126)}, //_M_KgyH0

    // LOOPGAIN
    {0x840f, *(uint16_t*)(eeprom+0x118)}, //_M_KgxG
    {0x848f, *(uint16_t*)(eeprom+0x11a)}, //_M_KgyG

    // Position Servo ON ( OIS OFF )
    {0x847f, 0x0c0c}, //_M_EQCTL
  };


  struct msm_ois_cfg_data cfg = {0};
  struct msm_camera_i2c_seq_reg_array ois_reg_settings[ARRAYSIZE(ois_registers)] = {{0}};
  for (int i=0; i<ARRAYSIZE(ois_registers); i++) {
    ois_reg_settings[i].reg_addr = ois_registers[i][0];
    ois_reg_settings[i].reg_data[0] = ois_registers[i][1] & 0xff;
    ois_reg_settings[i].reg_data[1] = (ois_registers[i][1] >> 8) & 0xff;
    ois_reg_settings[i].reg_data_size = 2;
  }
  struct msm_camera_i2c_seq_reg_setting ois_reg_setting = {
    .reg_setting = &ois_reg_settings[0],
    .size = ARRAYSIZE(ois_reg_settings),
    .addr_type = MSM_CAMERA_I2C_WORD_ADDR,
    .delay = 0,
  };
  cfg.cfgtype = CFG_OIS_I2C_WRITE_SEQ_TABLE;
  cfg.cfg.settings = &ois_reg_setting;
  err = ioctl(ois_fd, VIDIOC_MSM_OIS_CFG, &cfg);
  LOG("ois reg calibration: %d", err);
}




static void sensors_init(MultiCameraState *s) {
  int err;

  unique_fd sensorinit_fd;
  if (s->device == DEVICE_LP3) {
    sensorinit_fd = open("/dev/v4l-subdev11", O_RDWR | O_NONBLOCK);
  } else {
    sensorinit_fd = open("/dev/v4l-subdev12", O_RDWR | O_NONBLOCK);
  }
  assert(sensorinit_fd >= 0);

  struct sensor_init_cfg_data sensor_init_cfg = {};

  // init rear sensor

  struct msm_camera_sensor_slave_info slave_info = {0};
  if (s->device == DEVICE_LP3) {
    slave_info = (struct msm_camera_sensor_slave_info){
      .sensor_name = "imx298",
      .eeprom_name = "sony_imx298",
      .actuator_name = "dw9800w",
      .ois_name = "",
      .flash_name = "pmic",
      .camera_id = 	CAMERA_0,
      .slave_addr = 32,
      .i2c_freq_mode = I2C_FAST_MODE,
      .addr_type = MSM_CAMERA_I2C_WORD_ADDR,
      .sensor_id_info = {
        .sensor_id_reg_addr = 22,
        .sensor_id = 664,
        .sensor_id_mask = 0,
        .module_id = 9,
        .vcm_id = 6,
      },
      .power_setting_array = {
        .power_setting_a = {
          {
            .seq_type = SENSOR_GPIO,
            .seq_val = 0,
            .config_val = 0,
            .delay = 1,
          },{
            .seq_type = SENSOR_VREG,
            .seq_val = 2,
            .config_val = 0,
            .delay = 0,
          },{
            .seq_type = SENSOR_GPIO,
            .seq_val = 5,
            .config_val = 2,
            .delay = 0,
          },{
            .seq_type = SENSOR_VREG,
            .seq_val = 1,
            .config_val = 0,
            .delay = 0,
          },{
            .seq_type = SENSOR_VREG,
            .seq_val = 3,
            .config_val = 0,
            .delay = 1,
          },{
            .seq_type = SENSOR_CLK,
            .seq_val = 0,
            .config_val = 24000000,
            .delay = 1,
          },{
            .seq_type = SENSOR_GPIO,
            .seq_val = 0,
            .config_val = 2,
            .delay = 10,
          },
        },
        .size = 7,
        .power_down_setting_a = {
          {
            .seq_type = SENSOR_CLK,
            .seq_val = 0,
            .config_val = 0,
            .delay = 1,
          },{
            .seq_type = SENSOR_GPIO,
            .seq_val = 0,
            .config_val = 0,
            .delay = 1,
          },{
            .seq_type = SENSOR_VREG,
            .seq_val = 1,
            .config_val = 0,
            .delay = 0,
          },{
            .seq_type = SENSOR_GPIO,
            .seq_val = 5,
            .config_val = 0,
            .delay = 0,
          },{
            .seq_type = SENSOR_VREG,
            .seq_val = 2,
            .config_val = 0,
            .delay = 0,
          },{
            .seq_type = SENSOR_VREG,
            .seq_val = 3,
            .config_val = 0,
            .delay = 1,
          },
        },
        .size_down = 6,
      },
      .is_init_params_valid = 0,
      .sensor_init_params = {
        .modes_supported = 1,
        .position = BACK_CAMERA_B,
        .sensor_mount_angle = 90,
      },
      .output_format = MSM_SENSOR_BAYER,
    };
  } else {
    slave_info = (struct msm_camera_sensor_slave_info){
      .sensor_name = "imx298",
      .eeprom_name = "sony_imx298",
      .actuator_name = "rohm_bu63165gwl",
      .ois_name = "rohm_bu63165gwl",
      .camera_id = CAMERA_0,
      .slave_addr = 52,
      .i2c_freq_mode = I2C_CUSTOM_MODE,
      .addr_type = MSM_CAMERA_I2C_WORD_ADDR,
      .sensor_id_info = {
        .sensor_id_reg_addr = 22,
        .sensor_id = 664,
        .sensor_id_mask = 0,
      },
      .power_setting_array = {
        .power_setting_a = {
          {
            .seq_type = SENSOR_GPIO,
            .seq_val = 0,
            .config_val = 0,
            .delay = 2,
          },{
            .seq_type = SENSOR_VREG,
            .seq_val = 2,
            .config_val = 0,
            .delay = 2,
          },{
            .seq_type = SENSOR_VREG,
            .seq_val = 0,
            .config_val = 0,
            .delay = 2,
          },{
            .seq_type = SENSOR_VREG,
            .seq_val = 1,
            .config_val = 0,
            .delay = 2,
          },{
            .seq_type = SENSOR_GPIO,
            .seq_val = 6,
            .config_val = 2,
            .delay = 0,
          },{
            .seq_type = SENSOR_VREG,
            .seq_val = 3,
            .config_val = 0,
            .delay = 5,
          },{
            .seq_type = SENSOR_VREG,
            .seq_val = 4,
            .config_val = 0,
            .delay = 5,
          },{
            .seq_type = SENSOR_CLK,
            .seq_val = 0,
            .config_val = 24000000,
            .delay = 2,
          },{
            .seq_type = SENSOR_GPIO,
            .seq_val = 0,
            .config_val = 2,
            .delay = 2,
          },
        },
        .size = 9,
        .power_down_setting_a = {
          {
            .seq_type = SENSOR_GPIO,
            .seq_val = 0,
            .config_val = 0,
            .delay = 10,
          },{
            .seq_type = SENSOR_CLK,
            .seq_val = 0,
            .config_val = 0,
            .delay = 1,
          },{
            .seq_type = SENSOR_VREG,
            .seq_val = 4,
            .config_val = 0,
            .delay = 0,
          },{
            .seq_type = SENSOR_VREG,
            .seq_val = 3,
            .config_val = 0,
            .delay = 1,
          },{
            .seq_type = SENSOR_GPIO,
            .seq_val = 6,
            .config_val = 0,
            .delay = 0,
          },{
            .seq_type = SENSOR_VREG,
            .seq_val = 1,
            .config_val = 0,
            .delay = 0,
          },{
            .seq_type = SENSOR_VREG,
            .seq_val = 0,
            .config_val = 0,
            .delay = 0,
          },{
            .seq_type = SENSOR_VREG,
            .seq_val = 2,
            .config_val = 0,
            .delay = 0,
          },
        },
        .size_down = 8,
      },
      .is_init_params_valid = 0,
      .sensor_init_params = {
        .modes_supported = 1,
        .position = BACK_CAMERA_B,
        .sensor_mount_angle = 360,
      },
      .output_format = MSM_SENSOR_BAYER,
    };
  }
  slave_info.power_setting_array.power_setting =
    (struct msm_sensor_power_setting *)&slave_info.power_setting_array.power_setting_a[0];
  slave_info.power_setting_array.power_down_setting =
    (struct msm_sensor_power_setting *)&slave_info.power_setting_array.power_down_setting_a[0];
  sensor_init_cfg.cfgtype = CFG_SINIT_PROBE;
  sensor_init_cfg.cfg.setting = &slave_info;
  err = ioctl(sensorinit_fd, VIDIOC_MSM_SENSOR_INIT_CFG, &sensor_init_cfg);
  LOG("sensor init cfg (rear): %d", err);
  assert(err >= 0);


  struct msm_camera_sensor_slave_info slave_info2 = {0};
  if (s->device == DEVICE_LP3) {
    slave_info2 = (struct msm_camera_sensor_slave_info){
      .sensor_name = "ov8865_sunny",
      .eeprom_name = "ov8865_plus",
      .actuator_name = "",
      .ois_name = "",
      .flash_name = "",
      .camera_id = CAMERA_2,
      .slave_addr = 108,
      .i2c_freq_mode = I2C_FAST_MODE,
      .addr_type = MSM_CAMERA_I2C_WORD_ADDR,
      .sensor_id_info = {
        .sensor_id_reg_addr = 12299,
        .sensor_id = 34917,
        .sensor_id_mask = 0,
        .module_id = 2,
        .vcm_id = 0,
      },
      .power_setting_array = {
        .power_setting_a = {
          {
            .seq_type = SENSOR_GPIO,
            .seq_val = 0,
            .config_val = 0,
            .delay = 5,
          },{
            .seq_type = SENSOR_VREG,
            .seq_val = 1,
            .config_val = 0,
            .delay = 0,
          },{
            .seq_type = SENSOR_VREG,
            .seq_val = 2,
            .config_val = 0,
            .delay = 0,
          },{
            .seq_type = SENSOR_VREG,
            .seq_val = 0,
            .config_val = 0,
            .delay = 0,
          },{
            .seq_type = SENSOR_CLK,
            .seq_val = 0,
            .config_val = 24000000,
            .delay = 1,
          },{
            .seq_type = SENSOR_GPIO,
            .seq_val = 0,
            .config_val = 2,
            .delay = 1,
          },
        },
        .size = 6,
        .power_down_setting_a = {
          {
            .seq_type = SENSOR_GPIO,
            .seq_val = 0,
            .config_val = 0,
            .delay = 5,
          },{
            .seq_type = SENSOR_CLK,
            .seq_val = 0,
            .config_val = 0,
            .delay = 1,
          },{
            .seq_type = SENSOR_VREG,
            .seq_val = 0,
            .config_val = 0,
            .delay = 0,
          },{
            .seq_type = SENSOR_VREG,
            .seq_val = 1,
            .config_val = 0,
            .delay = 0,
          },{
            .seq_type = SENSOR_VREG,
            .seq_val = 2,
            .config_val = 0,
            .delay = 1,
          },
        },
        .size_down = 5,
      },
      .is_init_params_valid = 0,
      .sensor_init_params = {
        .modes_supported = 1,
        .position = FRONT_CAMERA_B,
        .sensor_mount_angle = 270,
      },
      .output_format = MSM_SENSOR_BAYER,
    };
  } else if (s->front.camera_id == CAMERA_ID_S5K3P8SP) {
    // init front camera
    slave_info2 = (struct msm_camera_sensor_slave_info){
      .sensor_name = "s5k3p8sp",
      .eeprom_name = "s5k3p8sp_m24c64s",
      .actuator_name = "",
      .ois_name = "",
      .camera_id = CAMERA_1,
      .slave_addr = 32,
      .i2c_freq_mode = I2C_FAST_MODE,
      .addr_type = MSM_CAMERA_I2C_WORD_ADDR,
      .sensor_id_info = {
        .sensor_id_reg_addr = 0,
        .sensor_id = 12552,
        .sensor_id_mask = 0,
      },
      .power_setting_array = {
        .power_setting_a = {
          {
            .seq_type = SENSOR_GPIO,
            .seq_val = 0,
            .config_val = 0,
            .delay = 1,
          },{
            .seq_type = SENSOR_VREG,
            .seq_val = 2,
            .config_val = 0,
            .delay = 1,
          },{
            .seq_type = SENSOR_VREG,
            .seq_val = 1,
            .config_val = 0,
            .delay = 1,
          },{
            .seq_type = SENSOR_VREG,
            .seq_val = 0,
            .config_val = 0,
            .delay = 1,
          },{
            .seq_type = SENSOR_CLK,
            .seq_val = 0,
            .config_val = 24000000,
            .delay = 1,
          },{
            .seq_type = SENSOR_GPIO,
            .seq_val = 0,
            .config_val = 2,
            .delay = 1,
          },
        },
        .size = 6,
        .power_down_setting_a = {
          {
            .seq_type = SENSOR_CLK,
            .seq_val = 0,
            .config_val = 0,
            .delay = 1,
          },{
            .seq_type = SENSOR_GPIO,
            .seq_val = 0,
            .config_val = 0,
            .delay = 1,
          },{
            .seq_type = SENSOR_VREG,
            .seq_val = 0,
            .config_val = 0,
            .delay = 1,
          },{
            .seq_type = SENSOR_VREG,
            .seq_val = 1,
            .config_val = 0,
            .delay = 1,
          },{
            .seq_type = SENSOR_VREG,
            .seq_val = 2,
            .config_val = 0,
            .delay = 1,
          },
        },
        .size_down = 5,
      },
      .is_init_params_valid = 0,
      .sensor_init_params = {
        .modes_supported = 1,
        .position = FRONT_CAMERA_B,
        .sensor_mount_angle = 270,
      },
      .output_format = MSM_SENSOR_BAYER,
    };
  } else {
    // init front camera
    slave_info2 = (struct msm_camera_sensor_slave_info){
      .sensor_name = "imx179",
      .eeprom_name = "sony_imx179",
      .actuator_name = "",
      .ois_name = "",
      .camera_id = CAMERA_1,
      .slave_addr = 32,
      .i2c_freq_mode = I2C_FAST_MODE,
      .addr_type = MSM_CAMERA_I2C_WORD_ADDR,
      .sensor_id_info = {
        .sensor_id_reg_addr = 2,
        .sensor_id = 377,
        .sensor_id_mask = 4095,
      },
      .power_setting_array = {
        .power_setting_a = {
          {
            .seq_type = SENSOR_VREG,
            .seq_val = 2,
            .config_val = 0,
            .delay = 0,
          },{
            .seq_type = SENSOR_VREG,
            .seq_val = 1,
            .config_val = 0,
            .delay = 0,
          },{
            .seq_type = SENSOR_VREG,
            .seq_val = 0,
            .config_val = 0,
            .delay = 0,
          },{
            .seq_type = SENSOR_GPIO,
            .seq_val = 0,
            .config_val = 2,
            .delay = 0,
          },{
            .seq_type = SENSOR_CLK,
            .seq_val = 0,
            .config_val = 24000000,
            .delay = 0,
          },
        },
        .size = 5,
        .power_down_setting_a = {
          {
            .seq_type = SENSOR_CLK,
            .seq_val = 0,
            .config_val = 0,
            .delay = 0,
          },{
            .seq_type = SENSOR_GPIO,
            .seq_val = 0,
            .config_val = 0,
            .delay = 1,
          },{
            .seq_type = SENSOR_VREG,
            .seq_val = 0,
            .config_val = 0,
            .delay = 2,
          },{
            .seq_type = SENSOR_VREG,
            .seq_val = 1,
            .config_val = 0,
            .delay = 0,
          },{
            .seq_type = SENSOR_VREG,
            .seq_val = 2,
            .config_val = 0,
            .delay = 0,
          },
        },
        .size_down = 5,
      },
      .is_init_params_valid = 0,
      .sensor_init_params = {
        .modes_supported = 1,
        .position = FRONT_CAMERA_B,
        .sensor_mount_angle = 270,
      },
      .output_format = MSM_SENSOR_BAYER,
    };
  }
  slave_info2.power_setting_array.power_setting =
    (struct msm_sensor_power_setting *)&slave_info2.power_setting_array.power_setting_a[0];
  slave_info2.power_setting_array.power_down_setting =
    (struct msm_sensor_power_setting *)&slave_info2.power_setting_array.power_down_setting_a[0];
  sensor_init_cfg.cfgtype = CFG_SINIT_PROBE;
  sensor_init_cfg.cfg.setting = &slave_info2;
  err = ioctl(sensorinit_fd, VIDIOC_MSM_SENSOR_INIT_CFG, &sensor_init_cfg);
  LOG("sensor init cfg (front): %d", err);
  assert(err >= 0);
}

static void camera_open(CameraState *s, bool rear) {
  int err;

  struct sensorb_cfg_data sensorb_cfg_data = {};
  struct csid_cfg_data csid_cfg_data = {};
  struct csiphy_cfg_data csiphy_cfg_data = {};
  struct msm_camera_csiphy_params csiphy_params = {};
  struct msm_camera_csid_params csid_params = {};
  struct msm_vfe_input_cfg input_cfg = {};
  struct msm_vfe_axi_stream_update_cmd update_cmd = {};
  struct v4l2_event_subscription sub = {};

  struct msm_actuator_cfg_data actuator_cfg_data = {};
  struct msm_ois_cfg_data ois_cfg_data = {};

  // open devices
  const char *sensor_dev;
  if (rear) {
    s->csid_fd = open("/dev/v4l-subdev3", O_RDWR | O_NONBLOCK);
    assert(s->csid_fd >= 0);
    s->csiphy_fd = open("/dev/v4l-subdev0", O_RDWR | O_NONBLOCK);
    assert(s->csiphy_fd >= 0);
    if (s->device == DEVICE_LP3) {
      sensor_dev = "/dev/v4l-subdev17";
    } else {
      sensor_dev = "/dev/v4l-subdev18";
    }
    if (s->device == DEVICE_LP3) {
      s->isp_fd = open("/dev/v4l-subdev13", O_RDWR | O_NONBLOCK);
    } else {
      s->isp_fd = open("/dev/v4l-subdev14", O_RDWR | O_NONBLOCK);
    }
    assert(s->isp_fd >= 0);
    s->eeprom_fd = open("/dev/v4l-subdev8", O_RDWR | O_NONBLOCK);
    assert(s->eeprom_fd >= 0);

    s->actuator_fd = open("/dev/v4l-subdev7", O_RDWR | O_NONBLOCK);
    assert(s->actuator_fd >= 0);

    if (s->device != DEVICE_LP3) {
      s->ois_fd = open("/dev/v4l-subdev10", O_RDWR | O_NONBLOCK);
      assert(s->ois_fd >= 0);
    }
  } else {
    s->csid_fd = open("/dev/v4l-subdev5", O_RDWR | O_NONBLOCK);
    assert(s->csid_fd >= 0);
    s->csiphy_fd = open("/dev/v4l-subdev2", O_RDWR | O_NONBLOCK);
    assert(s->csiphy_fd >= 0);
    if (s->device == DEVICE_LP3) {
      sensor_dev = "/dev/v4l-subdev18";
    } else {
      sensor_dev = "/dev/v4l-subdev19";
    }
    if (s->device == DEVICE_LP3) {
      s->isp_fd = open("/dev/v4l-subdev14", O_RDWR | O_NONBLOCK);
    } else {
      s->isp_fd = open("/dev/v4l-subdev15", O_RDWR | O_NONBLOCK);
    }
    assert(s->isp_fd >= 0);
    s->eeprom_fd = open("/dev/v4l-subdev9", O_RDWR | O_NONBLOCK);
    assert(s->eeprom_fd >= 0);
  }

  // wait for sensor device
  // on first startup, these devices aren't present yet
  for (int i = 0; i < 10; i++) {
    s->sensor_fd = open(sensor_dev, O_RDWR | O_NONBLOCK);
    if (s->sensor_fd >= 0) break;
    LOGW("waiting for sensors...");
    sleep(1);
  }
  assert(s->sensor_fd >= 0);

  // *** SHUTDOWN ALL ***

  // CSIPHY: release csiphy
  struct msm_camera_csi_lane_params csi_lane_params = {0};
  csi_lane_params.csi_lane_mask = 0x1f;
  csiphy_cfg_data.cfg.csi_lane_params = &csi_lane_params;
  csiphy_cfg_data.cfgtype = CSIPHY_RELEASE;
  err = ioctl(s->csiphy_fd, VIDIOC_MSM_CSIPHY_IO_CFG, &csiphy_cfg_data);
  LOG("release csiphy: %d", err);

  // CSID: release csid
  csid_cfg_data.cfgtype = CSID_RELEASE;
  err = ioctl(s->csid_fd, VIDIOC_MSM_CSID_IO_CFG, &csid_cfg_data);
  LOG("release csid: %d", err);

  // SENSOR: send power down
  memset(&sensorb_cfg_data, 0, sizeof(sensorb_cfg_data));
  sensorb_cfg_data.cfgtype = CFG_POWER_DOWN;
  err = ioctl(s->sensor_fd, VIDIOC_MSM_SENSOR_CFG, &sensorb_cfg_data);
  LOG("sensor power down: %d", err);

  if (rear && s->device != DEVICE_LP3) {
    // ois powerdown
    ois_cfg_data.cfgtype = CFG_OIS_POWERDOWN;
    err = ioctl(s->ois_fd, VIDIOC_MSM_OIS_CFG, &ois_cfg_data);
    LOG("ois powerdown: %d", err);
  }

  // actuator powerdown
  actuator_cfg_data.cfgtype = CFG_ACTUATOR_POWERDOWN;
  err = ioctl(s->actuator_fd, VIDIOC_MSM_ACTUATOR_CFG, &actuator_cfg_data);
  LOG("actuator powerdown: %d", err);

  // reset isp
  // struct msm_vfe_axi_halt_cmd halt_cmd = {
  //   .stop_camif = 1,
  //   .overflow_detected = 1,
  //   .blocking_halt = 1,
  // };
  // err = ioctl(s->isp_fd, VIDIOC_MSM_ISP_AXI_HALT, &halt_cmd);
  // printf("axi halt: %d\n", err);

  // struct msm_vfe_axi_reset_cmd reset_cmd = {
  //   .blocking = 1,
  //   .frame_id = 1,
  // };
  // err = ioctl(s->isp_fd, VIDIOC_MSM_ISP_AXI_RESET, &reset_cmd);
  // printf("axi reset: %d\n", err);

  // struct msm_vfe_axi_restart_cmd restart_cmd = {
  //   .enable_camif = 1,
  // };
  // err = ioctl(s->isp_fd, VIDIOC_MSM_ISP_AXI_RESTART, &restart_cmd);
  // printf("axi restart: %d\n", err);

  // **** GO GO GO ****
  LOG("******************** GO GO GO ************************");

  s->eeprom = get_eeprom(s->eeprom_fd, &s->eeprom_size);

  // printf("eeprom:\n");
  // for (int i=0; i<s->eeprom_size; i++) {
  //   printf("%02x", s->eeprom[i]);
  // }
  // printf("\n");

  // CSID: init csid
  csid_cfg_data.cfgtype = CSID_INIT;
  err = ioctl(s->csid_fd, VIDIOC_MSM_CSID_IO_CFG, &csid_cfg_data);
  LOG("init csid: %d", err);

  // CSIPHY: init csiphy
  memset(&csiphy_cfg_data, 0, sizeof(csiphy_cfg_data));
  csiphy_cfg_data.cfgtype = CSIPHY_INIT;
  err = ioctl(s->csiphy_fd, VIDIOC_MSM_CSIPHY_IO_CFG, &csiphy_cfg_data);
  LOG("init csiphy: %d", err);

  // SENSOR: stop stream
  struct msm_camera_i2c_reg_setting stop_settings = {
    .reg_setting = stop_reg_array,
    .size = ARRAYSIZE(stop_reg_array),
    .addr_type = MSM_CAMERA_I2C_WORD_ADDR,
    .data_type = MSM_CAMERA_I2C_BYTE_DATA,
    .delay = 0
  };
  sensorb_cfg_data.cfgtype = CFG_SET_STOP_STREAM_SETTING;
  sensorb_cfg_data.cfg.setting = &stop_settings;
  err = ioctl(s->sensor_fd, VIDIOC_MSM_SENSOR_CFG, &sensorb_cfg_data);
  LOG("stop stream: %d", err);

  // SENSOR: send power up
  memset(&sensorb_cfg_data, 0, sizeof(sensorb_cfg_data));
  sensorb_cfg_data.cfgtype = CFG_POWER_UP;
  err = ioctl(s->sensor_fd, VIDIOC_MSM_SENSOR_CFG, &sensorb_cfg_data);
  LOG("sensor power up: %d", err);

  // **** configure the sensor ****

  // SENSOR: send i2c configuration
  if (s->camera_id == CAMERA_ID_IMX298) {
    err = sensor_write_regs(s, init_array_imx298, ARRAYSIZE(init_array_imx298), MSM_CAMERA_I2C_BYTE_DATA);
  } else if  (s->camera_id == CAMERA_ID_S5K3P8SP) {
    err = sensor_write_regs(s, init_array_s5k3p8sp, ARRAYSIZE(init_array_s5k3p8sp), MSM_CAMERA_I2C_WORD_DATA);
  } else if (s->camera_id == CAMERA_ID_IMX179) {
    err = sensor_write_regs(s, init_array_imx179, ARRAYSIZE(init_array_imx179), MSM_CAMERA_I2C_BYTE_DATA);
  } else if (s->camera_id == CAMERA_ID_OV8865) {
    err = sensor_write_regs(s, init_array_ov8865, ARRAYSIZE(init_array_ov8865), MSM_CAMERA_I2C_BYTE_DATA);
  } else {
    assert(false);
  }
  LOG("sensor init i2c: %d", err);

  if (rear) {
    // init the actuator
    actuator_cfg_data.cfgtype = CFG_ACTUATOR_POWERUP;
    err = ioctl(s->actuator_fd, VIDIOC_MSM_ACTUATOR_CFG, &actuator_cfg_data);
    LOG("actuator powerup: %d", err);

    actuator_cfg_data.cfgtype = CFG_ACTUATOR_INIT;
    err = ioctl(s->actuator_fd, VIDIOC_MSM_ACTUATOR_CFG, &actuator_cfg_data);
    LOG("actuator init: %d", err);


    // no OIS in LP3
    if (s->device != DEVICE_LP3) {
      // see sony_imx298_eeprom_format_afdata in libmmcamera_sony_imx298_eeprom.so
      const float far_margin = -0.28;
      uint16_t macro_dac = *(uint16_t*)(s->eeprom + 0x24);
      s->infinity_dac = *(uint16_t*)(s->eeprom + 0x26);
      LOG("macro_dac: %d infinity_dac: %d", macro_dac, s->infinity_dac);

      int dac_range = macro_dac - s->infinity_dac;
      s->infinity_dac += far_margin * dac_range;

      LOG(" -> macro_dac: %d infinity_dac: %d", macro_dac, s->infinity_dac);

      struct msm_actuator_reg_params_t actuator_reg_params[] = {
        {
          .reg_write_type = MSM_ACTUATOR_WRITE_DAC,
          .hw_mask = 0,
          .reg_addr = 240,
          .hw_shift = 0,
          .data_type = 10,
          .addr_type = 4,
          .reg_data = 0,
          .delay = 0,
        }, {
          .reg_write_type = MSM_ACTUATOR_WRITE_DAC,
          .hw_mask = 0,
          .reg_addr = 241,
          .hw_shift = 0,
          .data_type = 10,
          .addr_type = 4,
          .reg_data = 0,
          .delay = 0,
        }, {
          .reg_write_type = MSM_ACTUATOR_WRITE_DAC,
          .hw_mask = 0,
          .reg_addr = 242,
          .hw_shift = 0,
          .data_type = 10,
          .addr_type = 4,
          .reg_data = 0,
          .delay = 0,
        }, {
          .reg_write_type = MSM_ACTUATOR_WRITE_DAC,
          .hw_mask = 0,
          .reg_addr = 243,
          .hw_shift = 0,
          .data_type = 10,
          .addr_type = 4,
          .reg_data = 0,
          .delay = 0,
        },
      };

      //...
      struct reg_settings_t actuator_init_settings[1] = {0};

      struct region_params_t region_params[] = {
        {
          .step_bound = {512, 0,},
          .code_per_step = 118,
          .qvalue = 128,
        },
      };

      actuator_cfg_data.cfgtype = CFG_SET_ACTUATOR_INFO;
      actuator_cfg_data.cfg.set_info = (struct msm_actuator_set_info_t){
        .actuator_params = {
          .act_type = ACTUATOR_VCM,
          .reg_tbl_size = 4,
          .data_size = 10,
          .init_setting_size = 0,
          .i2c_freq_mode = I2C_CUSTOM_MODE,
          .i2c_addr = 28,
          .i2c_addr_type = MSM_ACTUATOR_BYTE_ADDR,
          .i2c_data_type = MSM_ACTUATOR_BYTE_DATA,
          .reg_tbl_params = &actuator_reg_params[0],
          .init_settings = &actuator_init_settings[0],
          .park_lens = {
            .damping_step = 1023,
            .damping_delay = 15000,
            .hw_params = 58404,
            .max_step = 20,
          }
        },
        .af_tuning_params =   {
          .initial_code = (int16_t)s->infinity_dac,
          .pwd_step = 0,
          .region_size = 1,
          .total_steps = 512,
          .region_params = &region_params[0],
        },
      };
      err = ioctl(s->actuator_fd, VIDIOC_MSM_ACTUATOR_CFG, &actuator_cfg_data);
      LOG("actuator set info: %d", err);

      // power up ois
      ois_cfg_data.cfgtype = CFG_OIS_POWERUP;
      err = ioctl(s->ois_fd, VIDIOC_MSM_OIS_CFG, &ois_cfg_data);
      LOG("ois powerup: %d", err);

      ois_cfg_data.cfgtype = CFG_OIS_INIT;
      err = ioctl(s->ois_fd, VIDIOC_MSM_OIS_CFG, &ois_cfg_data);
      LOG("ois init: %d", err);

      ois_cfg_data.cfgtype = CFG_OIS_CONTROL;
      ois_cfg_data.cfg.set_info.ois_params = (struct msm_ois_params_t){
        // .data_size = 26312,
        .setting_size = 120,
        .i2c_addr = 28,
        .i2c_freq_mode = I2C_CUSTOM_MODE,
        // .i2c_addr_type = wtf
        // .i2c_data_type = wtf
        .settings = &ois_init_settings[0],
      };
      err = ioctl(s->ois_fd, VIDIOC_MSM_OIS_CFG, &ois_cfg_data);
      LOG("ois init settings: %d", err);
    } else {
      // leeco actuator (DW9800W H-Bridge Driver IC)
      // from sniff
      s->infinity_dac = 364;

      struct msm_actuator_reg_params_t actuator_reg_params[] = {
        {
          .reg_write_type = MSM_ACTUATOR_WRITE_DAC,
          .hw_mask = 0,
          // MSB here at address 3
          .reg_addr = 3,
          .hw_shift = 0,
          .data_type = 9,
          .addr_type = 4,
          .reg_data = 0,
          .delay = 0,
        },
      };

      struct reg_settings_t actuator_init_settings[] = {
        { .reg_addr=2, .addr_type=MSM_ACTUATOR_BYTE_ADDR, .reg_data=1, .data_type = MSM_ACTUATOR_BYTE_DATA, .i2c_operation = MSM_ACT_WRITE, .delay = 0 },   // PD = power down
        { .reg_addr=2, .addr_type=MSM_ACTUATOR_BYTE_ADDR, .reg_data=0, .data_type = MSM_ACTUATOR_BYTE_DATA, .i2c_operation = MSM_ACT_WRITE, .delay = 2 },   // 0 = power up
        { .reg_addr=2, .addr_type=MSM_ACTUATOR_BYTE_ADDR, .reg_data=2, .data_type = MSM_ACTUATOR_BYTE_DATA, .i2c_operation = MSM_ACT_WRITE, .delay = 2 },   // RING = SAC mode
        { .reg_addr=6, .addr_type=MSM_ACTUATOR_BYTE_ADDR, .reg_data=64, .data_type = MSM_ACTUATOR_BYTE_DATA, .i2c_operation = MSM_ACT_WRITE, .delay = 0 },  // 0x40 = SAC3 mode
        { .reg_addr=7, .addr_type=MSM_ACTUATOR_BYTE_ADDR, .reg_data=113, .data_type = MSM_ACTUATOR_BYTE_DATA, .i2c_operation = MSM_ACT_WRITE, .delay = 0 },
        // 0x71 = DIV1 | DIV0 | SACT0 -- Tvib x 1/4 (quarter)
        // SAC Tvib = 6.3 ms + 0.1 ms = 6.4 ms / 4 = 1.6 ms
        // LSC 1-step = 252 + 1*4 = 256 ms / 4 = 64 ms
      };

      struct region_params_t region_params[] = {
        {
          .step_bound = {238, 0,},
          .code_per_step = 235,
          .qvalue = 128,
        },
      };

      actuator_cfg_data.cfgtype = CFG_SET_ACTUATOR_INFO;
      actuator_cfg_data.cfg.set_info = (struct msm_actuator_set_info_t){
        .actuator_params = {
          .act_type = ACTUATOR_BIVCM,
          .reg_tbl_size = 1,
          .data_size = 10,
          .init_setting_size = 5,
          .i2c_freq_mode = I2C_STANDARD_MODE,
          .i2c_addr = 24,
          .i2c_addr_type = MSM_ACTUATOR_BYTE_ADDR,
          .i2c_data_type = MSM_ACTUATOR_WORD_DATA,
          .reg_tbl_params = &actuator_reg_params[0],
          .init_settings = &actuator_init_settings[0],
          .park_lens = {
            .damping_step = 1023,
            .damping_delay = 14000,
            .hw_params = 11,
            .max_step = 20,
          }
        },
        .af_tuning_params =   {
          .initial_code = (int16_t)s->infinity_dac,
          .pwd_step = 0,
          .region_size = 1,
          .total_steps = 238,
          .region_params = &region_params[0],
        },
      };

      err = ioctl(s->actuator_fd, VIDIOC_MSM_ACTUATOR_CFG, &actuator_cfg_data);
      LOG("actuator set info: %d", err);
    }
  }

  if (s->camera_id == CAMERA_ID_IMX298) {
    err = sensor_write_regs(s, mode_setting_array_imx298, ARRAYSIZE(mode_setting_array_imx298), MSM_CAMERA_I2C_BYTE_DATA);
    LOG("sensor setup: %d", err);
  }

  // CSIPHY: configure csiphy
  if (s->camera_id == CAMERA_ID_IMX298) {
    csiphy_params.lane_cnt = 4;
    csiphy_params.settle_cnt = 14;
    csiphy_params.lane_mask = 0x1f;
    csiphy_params.csid_core = 0;
  } else if (s->camera_id == CAMERA_ID_S5K3P8SP) {
    csiphy_params.lane_cnt = 4;
    csiphy_params.settle_cnt = 24;
    csiphy_params.lane_mask = 0x1f;
    csiphy_params.csid_core = 0;
  } else if (s->camera_id == CAMERA_ID_IMX179) {
    csiphy_params.lane_cnt = 4;
    csiphy_params.settle_cnt = 11;
    csiphy_params.lane_mask = 0x1f;
    csiphy_params.csid_core = 2;
  } else if (s->camera_id == CAMERA_ID_OV8865) {
    // guess!
    csiphy_params.lane_cnt = 4;
    csiphy_params.settle_cnt = 24;
    csiphy_params.lane_mask = 0x1f;
    csiphy_params.csid_core = 2;
  }
  csiphy_cfg_data.cfgtype = CSIPHY_CFG;
  csiphy_cfg_data.cfg.csiphy_params = &csiphy_params;
  err = ioctl(s->csiphy_fd, VIDIOC_MSM_CSIPHY_IO_CFG, &csiphy_cfg_data);
  LOG("csiphy configure: %d", err);

  // CSID: configure csid
  csid_params.lane_cnt = 4;
  csid_params.lane_assign = 0x4320;
  if (rear) {
    csid_params.phy_sel = 0;
  } else {
    csid_params.phy_sel = 2;
  }
  csid_params.lut_params.num_cid = rear ? 3 : 1;

#define CSI_STATS 0x35
#define CSI_PD 0x36

  csid_params.lut_params.vc_cfg_a[0].cid = 0;
  csid_params.lut_params.vc_cfg_a[0].dt = CSI_RAW10;
  csid_params.lut_params.vc_cfg_a[0].decode_format = CSI_DECODE_10BIT;
  csid_params.lut_params.vc_cfg_a[1].cid = 1;
  csid_params.lut_params.vc_cfg_a[1].dt = CSI_PD;
  csid_params.lut_params.vc_cfg_a[1].decode_format = CSI_DECODE_10BIT;
  csid_params.lut_params.vc_cfg_a[2].cid = 2;
  csid_params.lut_params.vc_cfg_a[2].dt = CSI_STATS;
  csid_params.lut_params.vc_cfg_a[2].decode_format = CSI_DECODE_10BIT;

  csid_params.lut_params.vc_cfg[0] = &csid_params.lut_params.vc_cfg_a[0];
  csid_params.lut_params.vc_cfg[1] = &csid_params.lut_params.vc_cfg_a[1];
  csid_params.lut_params.vc_cfg[2] = &csid_params.lut_params.vc_cfg_a[2];

  csid_cfg_data.cfgtype = CSID_CFG;
  csid_cfg_data.cfg.csid_params = &csid_params;
  err = ioctl(s->csid_fd, VIDIOC_MSM_CSID_IO_CFG, &csid_cfg_data);
  LOG("csid configure: %d", err);

  // ISP: SMMU_ATTACH
  struct msm_vfe_smmu_attach_cmd smmu_attach_cmd = {
    .security_mode = 0,
    .iommu_attach_mode = IOMMU_ATTACH
  };
  err = ioctl(s->isp_fd, VIDIOC_MSM_ISP_SMMU_ATTACH, &smmu_attach_cmd);
  LOG("isp smmu attach: %d", err);

  // ******************* STREAM RAW *****************************

  // configure QMET input
  for (int i = 0; i < (rear ? 3 : 1); i++) {
    StreamState *ss = &s->ss[i];

    memset(&input_cfg, 0, sizeof(struct msm_vfe_input_cfg));
    input_cfg.input_src = (msm_vfe_input_src)(VFE_RAW_0+i);
    input_cfg.input_pix_clk = s->pixel_clock;
    input_cfg.d.rdi_cfg.cid = i;
    input_cfg.d.rdi_cfg.frame_based = 1;
    err = ioctl(s->isp_fd, VIDIOC_MSM_ISP_INPUT_CFG, &input_cfg);
    LOG("configure input(%d): %d", i, err);

    // ISP: REQUEST_STREAM
    ss->stream_req.axi_stream_handle = 0;
    if (rear) {
      ss->stream_req.session_id = 2;
      ss->stream_req.stream_id = /*ISP_META_CHANNEL_BIT | */ISP_NATIVE_BUF_BIT | (1+i);
    } else {
      ss->stream_req.session_id = 3;
      ss->stream_req.stream_id = ISP_NATIVE_BUF_BIT | 1;
    }

    if (i == 0) {
      ss->stream_req.output_format = v4l2_fourcc('R', 'G', '1', '0');
    } else {
      ss->stream_req.output_format = v4l2_fourcc('Q', 'M', 'E', 'T');
    }
    ss->stream_req.stream_src = (msm_vfe_axi_stream_src)(RDI_INTF_0+i);

#ifdef HIGH_FPS
    if (rear) {
      ss->stream_req.frame_skip_pattern = EVERY_3FRAME;
    }
#endif

    ss->stream_req.frame_base = 1;
    ss->stream_req.buf_divert = 1; //i == 0;

    // setup stream plane. doesn't even matter?
    /*s->stream_req.plane_cfg[0].output_plane_format = Y_PLANE;
    s->stream_req.plane_cfg[0].output_width = s->ci.frame_width;
    s->stream_req.plane_cfg[0].output_height = s->ci.frame_height;
    s->stream_req.plane_cfg[0].output_stride = s->ci.frame_width;
    s->stream_req.plane_cfg[0].output_scan_lines = s->ci.frame_height;
    s->stream_req.plane_cfg[0].rdi_cid = 0;*/

    err = ioctl(s->isp_fd, VIDIOC_MSM_ISP_REQUEST_STREAM, &ss->stream_req);
    LOG("isp request stream: %d -> 0x%x", err, ss->stream_req.axi_stream_handle);

    // ISP: REQUEST_BUF
    ss->buf_request.session_id = ss->stream_req.session_id;
    ss->buf_request.stream_id = ss->stream_req.stream_id;
    ss->buf_request.num_buf = FRAME_BUF_COUNT;
    ss->buf_request.buf_type = ISP_PRIVATE_BUF;
    ss->buf_request.handle = 0;
    err = ioctl(s->isp_fd, VIDIOC_MSM_ISP_REQUEST_BUF, &ss->buf_request);
    LOG("isp request buf: %d", err);
    LOG("got buf handle: 0x%x", ss->buf_request.handle);

    // ENQUEUE all buffers
    for (int j = 0; j < ss->buf_request.num_buf; j++) {
      ss->qbuf_info[j].handle = ss->buf_request.handle;
      ss->qbuf_info[j].buf_idx = j;
      ss->qbuf_info[j].buffer.num_planes = 1;
      ss->qbuf_info[j].buffer.planes[0].addr = ss->bufs[j].fd;
      ss->qbuf_info[j].buffer.planes[0].length = ss->bufs[j].len;
      err = ioctl(s->isp_fd, VIDIOC_MSM_ISP_ENQUEUE_BUF, &ss->qbuf_info[j]);
    }

    // ISP: UPDATE_STREAM
    update_cmd.num_streams = 1;
    update_cmd.update_info[0].user_stream_id = ss->stream_req.stream_id;
    update_cmd.update_info[0].stream_handle = ss->stream_req.axi_stream_handle;
    update_cmd.update_type = UPDATE_STREAM_ADD_BUFQ;
    err = ioctl(s->isp_fd, VIDIOC_MSM_ISP_UPDATE_STREAM, &update_cmd);
    LOG("isp update stream: %d", err);
  }

  LOG("******** START STREAMS ********");

  sub.id = 0;
  sub.type = 0x1ff;
  err = ioctl(s->isp_fd, VIDIOC_SUBSCRIBE_EVENT, &sub);
  LOG("isp subscribe: %d", err);

  // ISP: START_STREAM
  s->stream_cfg.cmd = START_STREAM;
  s->stream_cfg.num_streams = rear ? 3 : 1;
  for (int i = 0; i < s->stream_cfg.num_streams; i++) {
    s->stream_cfg.stream_handle[i] = s->ss[i].stream_req.axi_stream_handle;
  }
  err = ioctl(s->isp_fd, VIDIOC_MSM_ISP_CFG_STREAM, &s->stream_cfg);
  LOG("isp start stream: %d", err);
}


static struct damping_params_t actuator_ringing_params = {
  .damping_step = 1023,
  .damping_delay = 15000,
  .hw_params = 0x0000e422,
};

static void rear_start(CameraState *s) {
  int err;

  struct msm_actuator_cfg_data actuator_cfg_data = {0};

  set_exposure(s, 1.0, 1.0);

  err = sensor_write_regs(s, start_reg_array, ARRAYSIZE(start_reg_array), MSM_CAMERA_I2C_BYTE_DATA);
  LOG("sensor start regs: %d", err);

  // focus on infinity assuming phone is perpendicular
  int inf_step;

  if (s->device != DEVICE_LP3) {
    imx298_ois_calibration(s->ois_fd, s->eeprom);
    inf_step = 332 - s->infinity_dac;

    // initial guess
    s->lens_true_pos = 300;
  } else {
    // default is OP3, this is for LeEco
    actuator_ringing_params.damping_step = 1023;
    actuator_ringing_params.damping_delay = 20000;
    actuator_ringing_params.hw_params = 13;

    inf_step = 512 - s->infinity_dac;

    // initial guess
    s->lens_true_pos = 400;
  }

  // reset lens position
  memset(&actuator_cfg_data, 0, sizeof(actuator_cfg_data));
  actuator_cfg_data.cfgtype = CFG_SET_POSITION;
  actuator_cfg_data.cfg.setpos = (struct msm_actuator_set_position_t){
    .number_of_steps = 1,
    .hw_params = (uint32_t)((s->device != DEVICE_LP3) ? 0x0000e424 : 7),
    .pos = {s->infinity_dac, 0},
    .delay = {0,}
  };
  err = ioctl(s->actuator_fd, VIDIOC_MSM_ACTUATOR_CFG, &actuator_cfg_data);
  LOG("actuator set pos: %d", err);

  // TODO: confirm this isn't needed
  /*memset(&actuator_cfg_data, 0, sizeof(actuator_cfg_data));
  actuator_cfg_data.cfgtype = CFG_MOVE_FOCUS;
  actuator_cfg_data.cfg.move = (struct msm_actuator_move_params_t){
    .dir = 0,
    .sign_dir = 1,
    .dest_step_pos = inf_step,
    .num_steps = inf_step,
    .curr_lens_pos = 0,
    .ringing_params = &actuator_ringing_params,
  };
  err = ioctl(s->actuator_fd, VIDIOC_MSM_ACTUATOR_CFG, &actuator_cfg_data); // should be ~332 at startup ?
  LOG("init actuator move focus: %d", err);*/
  //actuator_cfg_data.cfg.move.curr_lens_pos;

  s->cur_lens_pos = 0;
  s->cur_step_pos = inf_step;

  actuator_move(s, s->cur_lens_pos);

  LOG("init lens pos: %d", s->cur_lens_pos);
}

void actuator_move(CameraState *s, uint16_t target) {
  int err;

  int step = target - s->cur_lens_pos;
  // LP3 moves only on even positions. TODO: use proper sensor params
  if (s->device == DEVICE_LP3) {
    step /= 2;
  }

  int dest_step_pos = s->cur_step_pos + step;
  dest_step_pos = std::clamp(dest_step_pos, 0, 255);

  struct msm_actuator_cfg_data actuator_cfg_data = {0};
  actuator_cfg_data.cfgtype = CFG_MOVE_FOCUS;
  actuator_cfg_data.cfg.move = (struct msm_actuator_move_params_t){
    .dir = (int8_t)((step > 0) ? 0 : 1),
    .sign_dir = (int8_t)((step > 0) ? 1 : -1),
    .dest_step_pos = (int16_t)dest_step_pos,
    .num_steps = abs(step),
    .curr_lens_pos = s->cur_lens_pos,
    .ringing_params = &actuator_ringing_params,
  };
  err = ioctl(s->actuator_fd, VIDIOC_MSM_ACTUATOR_CFG, &actuator_cfg_data);
  //LOGD("actuator move focus: %d", err);

  s->cur_step_pos = dest_step_pos;
  s->cur_lens_pos = actuator_cfg_data.cfg.move.curr_lens_pos;

  //LOGD("step %d   target: %d  lens pos: %d", dest_step_pos, target, s->cur_lens_pos);
}

static void parse_autofocus(CameraState *s, uint8_t *d) {
  int good_count = 0;
  int16_t max_focus = -32767;
  int avg_focus = 0;

  /*printf("FOCUS: ");
  for (int i = 0; i < 0x10; i++) {
    printf("%2.2X ", d[i]);
  }*/

  for (int i = 0; i < NUM_FOCUS; i++) {
    int doff = i*5+5;
    s->confidence[i] = d[doff];
    // this should just be a 10-bit signed int instead of 11
    // TODO: write it in a nicer way
    int16_t focus_t = (d[doff+1] << 3) | (d[doff+2] >> 5);
    if (focus_t >= 1024) focus_t = -(2048-focus_t);
    s->focus[i] = focus_t;
    //printf("%x->%d ", d[doff], focus_t);
    if (s->confidence[i] > 0x20) {
      good_count++;
      max_focus = std::max(max_focus, s->focus[i]);
      avg_focus += s->focus[i];
    }
  }
  // self recover override
  if (s->self_recover > 1) {
    s->focus_err = 200 * ((s->self_recover % 2 == 0) ? 1:-1); // far for even numbers, close for odd
    s->self_recover -= 2;
    return;
  }

  if (good_count < 4) {
    s->focus_err = nan("");
    return;
  }

  avg_focus /= good_count;

  // outlier rejection
  if (abs(avg_focus - max_focus) > 200) {
    s->focus_err = nan("");
    return;
  }

  s->focus_err = max_focus*1.0;
}

static void do_autofocus(CameraState *s) {
  // params for focus PI controller
  const float focus_kp = 0.005;

  float err = s->focus_err;
  float sag = (s->last_sag_acc_z/9.8) * 128;

  const int dac_up = s->device == DEVICE_LP3? LP3_AF_DAC_UP:OP3T_AF_DAC_UP;
  const int dac_down = s->device == DEVICE_LP3? LP3_AF_DAC_DOWN:OP3T_AF_DAC_DOWN;

  float lens_true_pos = s->lens_true_pos;
  if (!isnan(err))  {
    // learn lens_true_pos
    lens_true_pos -= err*focus_kp;
  }

  // stay off the walls
  lens_true_pos = std::clamp(lens_true_pos, float(dac_down), float(dac_up));
  int target = std::clamp(lens_true_pos - sag, float(dac_down), float(dac_up));
  s->lens_true_pos = lens_true_pos;

  /*char debug[4096];
  char *pdebug = debug;
  pdebug += sprintf(pdebug, "focus ");
  for (int i = 0; i < NUM_FOCUS; i++) pdebug += sprintf(pdebug, "%2x(%4d) ", s->confidence[i], s->focus[i]);
  pdebug += sprintf(pdebug, "  err: %7.2f  offset: %6.2f sag: %6.2f lens_true_pos: %6.2f  cur_lens_pos: %4d->%4d", err * focus_kp, offset, sag, s->lens_true_pos, s->cur_lens_pos, target);
  LOGD(debug);*/

  actuator_move(s, target);
}

void camera_autoexposure(CameraState *s, float grey_frac) {
  if (s->camera_num == 0) {
    CameraExpInfo tmp = rear_exp.load();
    tmp.op_id++;
    tmp.grey_frac = grey_frac;
    rear_exp.store(tmp);
  } else {
    CameraExpInfo tmp = front_exp.load();
    tmp.op_id++;
    tmp.grey_frac = grey_frac;
    front_exp.store(tmp);
  }
}

static void front_start(CameraState *s) {
  int err;

  set_exposure(s, 1.0, 1.0);

  err = sensor_write_regs(s, start_reg_array, ARRAYSIZE(start_reg_array), MSM_CAMERA_I2C_BYTE_DATA);
  LOG("sensor start regs: %d", err);
}

void cameras_open(MultiCameraState *s) {
  int err;
  struct ispif_cfg_data ispif_cfg_data = {};
  struct msm_ispif_param_data ispif_params = {};
  ispif_params.num = 4;
  // rear camera
  ispif_params.entries[0].vfe_intf = VFE0;
  ispif_params.entries[0].intftype = RDI0;
  ispif_params.entries[0].num_cids = 1;
  ispif_params.entries[0].cids[0] = CID0;
  ispif_params.entries[0].csid = CSID0;
  // front camera
  ispif_params.entries[1].vfe_intf = VFE1;
  ispif_params.entries[1].intftype = RDI0;
  ispif_params.entries[1].num_cids = 1;
  ispif_params.entries[1].cids[0] = CID0;
  ispif_params.entries[1].csid = CSID2;
  // rear camera (focus)
  ispif_params.entries[2].vfe_intf = VFE0;
  ispif_params.entries[2].intftype = RDI1;
  ispif_params.entries[2].num_cids = CID1;
  ispif_params.entries[2].cids[0] = CID1;
  ispif_params.entries[2].csid = CSID0;
  // rear camera (stats, for AE)
  ispif_params.entries[3].vfe_intf = VFE0;
  ispif_params.entries[3].intftype = RDI2;
  ispif_params.entries[3].num_cids = 1;
  ispif_params.entries[3].cids[0] = CID2;
  ispif_params.entries[3].csid = CSID0;

  s->msmcfg_fd = open("/dev/media0", O_RDWR | O_NONBLOCK);
  assert(s->msmcfg_fd >= 0);

  sensors_init(s);

  s->v4l_fd = open("/dev/video0", O_RDWR | O_NONBLOCK);
  assert(s->v4l_fd >= 0);

  if (s->device == DEVICE_LP3) {
    s->ispif_fd = open("/dev/v4l-subdev15", O_RDWR | O_NONBLOCK);
  } else {
    s->ispif_fd = open("/dev/v4l-subdev16", O_RDWR | O_NONBLOCK);
  }
  assert(s->ispif_fd >= 0);

  // ISPIF: stop
  // memset(&ispif_cfg_data, 0, sizeof(ispif_cfg_data));
  // ispif_cfg_data.cfg_type = ISPIF_STOP_FRAME_BOUNDARY;
  // ispif_cfg_data.params = ispif_params;
  // err = ioctl(s->ispif_fd, VIDIOC_MSM_ISPIF_CFG, &ispif_cfg_data);
  // LOG("ispif stop: %d", err);

  LOG("*** open front ***");
  s->front.ss[0].bufs = s->front.buf.camera_bufs.get();
  camera_open(&s->front, false);

  LOG("*** open rear ***");
  s->rear.ss[0].bufs = s->rear.buf.camera_bufs.get();
  s->rear.ss[1].bufs = s->focus_bufs;
  s->rear.ss[2].bufs = s->stats_bufs;
  camera_open(&s->rear, true);

  if (getenv("CAMERA_TEST")) {
    cameras_close(s);
    exit(0);
  }

  // ISPIF: set vfe info
  memset(&ispif_cfg_data, 0, sizeof(ispif_cfg_data));
  ispif_cfg_data.cfg_type = ISPIF_SET_VFE_INFO;
  ispif_cfg_data.vfe_info.num_vfe = 2;
  err = ioctl(s->ispif_fd, VIDIOC_MSM_ISPIF_CFG, &ispif_cfg_data);
  LOG("ispif set vfe info: %d", err);

  // ISPIF: setup
  memset(&ispif_cfg_data, 0, sizeof(ispif_cfg_data));
  ispif_cfg_data.cfg_type = ISPIF_INIT;
  ispif_cfg_data.csid_version = 0x30050000; //CSID_VERSION_V35
  err = ioctl(s->ispif_fd, VIDIOC_MSM_ISPIF_CFG, &ispif_cfg_data);
  LOG("ispif setup: %d", err);

  memset(&ispif_cfg_data, 0, sizeof(ispif_cfg_data));
  ispif_cfg_data.cfg_type = ISPIF_CFG;
  ispif_cfg_data.params = ispif_params;

  err = ioctl(s->ispif_fd, VIDIOC_MSM_ISPIF_CFG, &ispif_cfg_data);
  LOG("ispif cfg: %d", err);

  ispif_cfg_data.cfg_type = ISPIF_START_FRAME_BOUNDARY;
  err = ioctl(s->ispif_fd, VIDIOC_MSM_ISPIF_CFG, &ispif_cfg_data);
  LOG("ispif start_frame_boundary: %d", err);

  front_start(&s->front);
  rear_start(&s->rear);
}


static void camera_close(CameraState *s) {
  int err;

  s->buf.stop();

  // ISP: STOP_STREAM
  s->stream_cfg.cmd = STOP_STREAM;
  err = ioctl(s->isp_fd, VIDIOC_MSM_ISP_CFG_STREAM, &s->stream_cfg);
  LOG("isp stop stream: %d", err);

  for (int i = 0; i < 3; i++) {
    StreamState *ss = &s->ss[i];
    if (ss->stream_req.axi_stream_handle != 0) {
      err = ioctl(s->isp_fd, VIDIOC_MSM_ISP_RELEASE_BUF, &ss->buf_request);
      LOG("isp release buf: %d", err);

      struct msm_vfe_axi_stream_release_cmd stream_release = {
        .stream_handle = ss->stream_req.axi_stream_handle,
      };
      err = ioctl(s->isp_fd, VIDIOC_MSM_ISP_RELEASE_STREAM, &stream_release);
      LOG("isp release stream: %d", err);
    }
  }

  free(s->eeprom);
}


const char* get_isp_event_name(unsigned int type) {
  switch (type) {
  case ISP_EVENT_REG_UPDATE: return "ISP_EVENT_REG_UPDATE";
  case ISP_EVENT_EPOCH_0: return "ISP_EVENT_EPOCH_0";
  case ISP_EVENT_EPOCH_1: return "ISP_EVENT_EPOCH_1";
  case ISP_EVENT_START_ACK: return "ISP_EVENT_START_ACK";
  case ISP_EVENT_STOP_ACK: return "ISP_EVENT_STOP_ACK";
  case ISP_EVENT_IRQ_VIOLATION: return "ISP_EVENT_IRQ_VIOLATION";
  case ISP_EVENT_STATS_OVERFLOW: return "ISP_EVENT_STATS_OVERFLOW";
  case ISP_EVENT_ERROR: return "ISP_EVENT_ERROR";
  case ISP_EVENT_SOF: return "ISP_EVENT_SOF";
  case ISP_EVENT_EOF: return "ISP_EVENT_EOF";
  case ISP_EVENT_BUF_DONE: return "ISP_EVENT_BUF_DONE";
  case ISP_EVENT_BUF_DIVERT: return "ISP_EVENT_BUF_DIVERT";
  case ISP_EVENT_STATS_NOTIFY: return "ISP_EVENT_STATS_NOTIFY";
  case ISP_EVENT_COMP_STATS_NOTIFY: return "ISP_EVENT_COMP_STATS_NOTIFY";
  case ISP_EVENT_FE_READ_DONE: return "ISP_EVENT_FE_READ_DONE";
  case ISP_EVENT_IOMMU_P_FAULT: return "ISP_EVENT_IOMMU_P_FAULT";
  case ISP_EVENT_HW_FATAL_ERROR: return "ISP_EVENT_HW_FATAL_ERROR";
  case ISP_EVENT_PING_PONG_MISMATCH: return "ISP_EVENT_PING_PONG_MISMATCH";
  case ISP_EVENT_REG_UPDATE_MISSING: return "ISP_EVENT_REG_UPDATE_MISSING";
  case ISP_EVENT_BUF_FATAL_ERROR: return "ISP_EVENT_BUF_FATAL_ERROR";
  case ISP_EVENT_STREAM_UPDATE_DONE: return "ISP_EVENT_STREAM_UPDATE_DONE";
  default: return "unknown";
  }
}

static FrameMetadata get_frame_metadata(CameraState *s, uint32_t frame_id) {
  pthread_mutex_lock(&s->frame_info_lock);
  for (int i=0; i<METADATA_BUF_COUNT; i++) {
    if (s->frame_metadata[i].frame_id == frame_id) {
      pthread_mutex_unlock(&s->frame_info_lock);
      return s->frame_metadata[i];
    }
  }
  pthread_mutex_unlock(&s->frame_info_lock);

  // should never happen
  return (FrameMetadata){
    .frame_id = (uint32_t)-1,
  };
}

static void* ops_thread(void* arg) {
  MultiCameraState *s = (MultiCameraState*)arg;

  int rear_op_id_last = 0;
  int front_op_id_last = 0;

  CameraExpInfo rear_op;
  CameraExpInfo front_op;

  set_thread_name("camera_settings");

  while(!do_exit) {
    rear_op = rear_exp.load();
    if (rear_op.op_id != rear_op_id_last) {
      do_autoexposure(&s->rear, rear_op.grey_frac);
      do_autofocus(&s->rear);
      rear_op_id_last = rear_op.op_id;
    }

    front_op = front_exp.load();
    if (front_op.op_id != front_op_id_last) {
      do_autoexposure(&s->front, front_op.grey_frac);
      front_op_id_last = front_op.op_id;
    }

    usleep(50000);
  }

  return NULL;
}

void camera_process_front(MultiCameraState *s, CameraState *c, int cnt) {
  common_camera_process_front(s->sm_driver, s->pm, c, cnt);
}

// called by processing_thread
void camera_process_frame(MultiCameraState *s, CameraState *c, int cnt) {
  const CameraBuf *b = &c->buf;
  // cache rgb roi and write to cl

  // gz compensation
  s->sm_sensor->update(0);
  if (s->sm_sensor->updated("sensorEvents")) {
    float vals[3] = {0.0};
    bool got_accel = false;
    auto sensor_events = (*(s->sm_sensor))["sensorEvents"].getSensorEvents();
    for (auto sensor_event : sensor_events) {
      if (sensor_event.which() == cereal::SensorEventData::ACCELERATION) {
        auto v = sensor_event.getAcceleration().getV();
        if (v.size() < 3) {
          continue;
        }
        for (int j = 0; j < 3; j++) {
          vals[j] = v[j];
        }
        got_accel = true;
        break;
      }
    }
    uint64_t ts = nanos_since_boot();
    if (got_accel && ts - s->rear.last_sag_ts > 10000000) { // 10 ms
      s->rear.last_sag_ts = ts;
      s->rear.last_sag_acc_z = -vals[2];
    }
  }

  // sharpness scores
  int roi_id = cnt % ARRAYSIZE(s->lapres);  // rolling roi
  int roi_x_offset = roi_id % (ROI_X_MAX-ROI_X_MIN+1);
  int roi_y_offset = roi_id / (ROI_X_MAX-ROI_X_MIN+1);

  for (int r=0;r<(b->rgb_height/NUM_SEGMENTS_Y);r++) {
    memcpy(s->rgb_roi_buf.get() + r * (b->rgb_width/NUM_SEGMENTS_X) * 3,
            (uint8_t *) b->cur_rgb_buf->addr + \
              (ROI_Y_MIN + roi_y_offset) * b->rgb_height/NUM_SEGMENTS_Y * FULL_STRIDE_X * 3 + \
              (ROI_X_MIN + roi_x_offset) * b->rgb_width/NUM_SEGMENTS_X * 3 + r * FULL_STRIDE_X * 3,
            b->rgb_width/NUM_SEGMENTS_X * 3);
  }

  assert(clEnqueueWriteBuffer(b->q, s->rgb_conv_roi_cl, true, 0,
                              b->rgb_width / NUM_SEGMENTS_X * b->rgb_height / NUM_SEGMENTS_Y * 3 * sizeof(uint8_t), s->rgb_roi_buf.get(), 0, 0, 0) == 0);
  assert(clSetKernelArg(s->krnl_rgb_laplacian, 0, sizeof(cl_mem), (void *)&s->rgb_conv_roi_cl) == 0);
  assert(clSetKernelArg(s->krnl_rgb_laplacian, 1, sizeof(cl_mem), (void *)&s->rgb_conv_result_cl) == 0);
  assert(clSetKernelArg(s->krnl_rgb_laplacian, 2, sizeof(cl_mem), (void *)&s->rgb_conv_filter_cl) == 0);
  assert(clSetKernelArg(s->krnl_rgb_laplacian, 3, s->conv_cl_localMemSize, 0) == 0);
  cl_event conv_event;
  assert(clEnqueueNDRangeKernel(b->q, s->krnl_rgb_laplacian, 2, NULL,
                                s->conv_cl_globalWorkSize, s->conv_cl_localWorkSize, 0, 0, &conv_event) == 0);
  clWaitForEvents(1, &conv_event);
  clReleaseEvent(conv_event);

  assert(clEnqueueReadBuffer(b->q, s->rgb_conv_result_cl, true, 0,
                             b->rgb_width / NUM_SEGMENTS_X * b->rgb_height / NUM_SEGMENTS_Y * sizeof(int16_t), s->conv_result.get(), 0, 0, 0) == 0);

  get_lapmap_one(s->conv_result.get(), &s->lapres[roi_id], b->rgb_width / NUM_SEGMENTS_X, b->rgb_height / NUM_SEGMENTS_Y);

  // setup self recover
  const float lens_true_pos = s->rear.lens_true_pos;
  std::atomic<int>& self_recover = s->rear.self_recover;
  if (is_blur(&s->lapres[0]) &&
      (lens_true_pos < (s->rear.device == DEVICE_LP3 ? LP3_AF_DAC_DOWN : OP3T_AF_DAC_DOWN) + 1 ||
       lens_true_pos > (s->rear.device == DEVICE_LP3 ? LP3_AF_DAC_UP : OP3T_AF_DAC_UP) - 1) &&
      self_recover < 2) {
    // truly stuck, needs help
    self_recover -= 1;
    if (self_recover < -FOCUS_RECOVER_PATIENCE) {
      LOGD("rear camera bad state detected. attempting recovery from %.1f, recover state is %d",
           lens_true_pos, self_recover.load());
      self_recover = FOCUS_RECOVER_STEPS + ((lens_true_pos < (s->rear.device == DEVICE_LP3 ? LP3_AF_DAC_M : OP3T_AF_DAC_M)) ? 1 : 0);  // parity determined by which end is stuck at
    }
  } else if ((lens_true_pos < (s->rear.device == DEVICE_LP3 ? LP3_AF_DAC_M - LP3_AF_DAC_3SIG : OP3T_AF_DAC_M - OP3T_AF_DAC_3SIG) ||
              lens_true_pos > (s->rear.device == DEVICE_LP3 ? LP3_AF_DAC_M + LP3_AF_DAC_3SIG : OP3T_AF_DAC_M + OP3T_AF_DAC_3SIG)) &&
             self_recover < 2) {
    // in suboptimal position with high prob, but may still recover by itself
    self_recover -= 1;
    if (self_recover < -(FOCUS_RECOVER_PATIENCE * 3)) {
      self_recover = FOCUS_RECOVER_STEPS / 2 + ((lens_true_pos < (s->rear.device == DEVICE_LP3 ? LP3_AF_DAC_M : OP3T_AF_DAC_M)) ? 1 : 0);
    }
  } else if (self_recover < 0) {
    self_recover += 1;  // reset if fine
  }

  {
    MessageBuilder msg;
    auto framed = msg.initEvent().initFrame();
    fill_frame_data(framed, b->cur_frame_data, cnt);
    framed.setFocusVal(kj::ArrayPtr<const int16_t>(&s->rear.focus[0], NUM_FOCUS));
    framed.setFocusConf(kj::ArrayPtr<const uint8_t>(&s->rear.confidence[0], NUM_FOCUS));
    framed.setSharpnessScore(kj::ArrayPtr<const uint16_t>(&s->lapres[0], ARRAYSIZE(s->lapres)));
    framed.setRecoverState(self_recover);
    framed.setTransform(kj::ArrayPtr<const float>(&b->yuv_transform.v[0], 9));
    s->pm->send("frame", msg);
  }

  if (cnt % 100 == 3) {
    create_thumbnail(s, c, (uint8_t*)b->cur_rgb_buf->addr);
  }

  const int exposure_x = 290;
  const int exposure_y = 322;
  const int exposure_width = 560;
  const int exposure_height = 314;
  const int skip = 1;
  if (cnt % 3 == 0) {
    set_exposure_target(c, (const uint8_t *)b->yuv_bufs[b->cur_yuv_idx].y, exposure_x, exposure_x + exposure_width, skip, exposure_y, exposure_y + exposure_height, skip);
  }
}

void cameras_run(MultiCameraState *s) {
  int err;

  pthread_t ops_thread_handle;
  err = pthread_create(&ops_thread_handle, NULL,
                       ops_thread, s);
  assert(err == 0);
  std::vector<std::thread> threads;
  threads.push_back(start_process_thread(s, "processing", &s->rear, 51, camera_process_frame));
  threads.push_back(start_process_thread(s, "frontview", &s->front, 51, camera_process_front));

  CameraState* cameras[2] = {&s->rear, &s->front};

  while (!do_exit) {
    struct pollfd fds[2] = {{0}};

    fds[0].fd = cameras[0]->isp_fd;
    fds[0].events = POLLPRI;

    fds[1].fd = cameras[1]->isp_fd;
    fds[1].events = POLLPRI;

    int ret = poll(fds, ARRAYSIZE(fds), 1000);
    if (ret <= 0) {
      if (errno == EINTR || errno == EAGAIN) continue;
      LOGE("poll failed (%d - %d)", ret, errno);
      break;
    }

    // process cameras
    for (int i=0; i<2; i++) {
      if (!fds[i].revents) continue;

      CameraState *c = cameras[i];

      struct v4l2_event ev;
      ret = ioctl(c->isp_fd, VIDIOC_DQEVENT, &ev);
      struct msm_isp_event_data *isp_event_data = (struct msm_isp_event_data *)ev.u.data;
      unsigned int event_type = ev.type;

      uint64_t timestamp = (isp_event_data->mono_timestamp.tv_sec*1000000000ULL
                            + isp_event_data->mono_timestamp.tv_usec*1000);

      int buf_idx = isp_event_data->u.buf_done.buf_idx;
      int stream_id = isp_event_data->u.buf_done.stream_id;
      int buffer = (stream_id&0xFFFF) - 1;

      uint64_t t = nanos_since_boot();

      /*if (i == 1) {
        printf("%10.2f: VIDIOC_DQEVENT: %d  type:%X (%s)\n", t*1.0/1e6, ret, event_type, get_isp_event_name(event_type));
      }*/

      // printf("%d: %s\n", i, get_isp_event_name(event_type));

      switch (event_type) {
      case ISP_EVENT_BUF_DIVERT:

        /*if (c->is_samsung) {
          printf("write %d\n", c->frame_size);
          FILE *f = fopen("/tmp/test", "wb");
          fwrite((void*)c->camera_bufs[i].addr, 1, c->frame_size, f);
          fclose(f);
        }*/
        //printf("divert: %d %d %d\n", i, buffer, buf_idx);

        if (buffer == 0) {
          c->buf.camera_bufs_metadata[buf_idx] = get_frame_metadata(c, isp_event_data->frame_id);
          tbuffer_dispatch(&c->buf.camera_tb, buf_idx);
        } else {
          uint8_t *d = (uint8_t*)(c->ss[buffer].bufs[buf_idx].addr);
          if (buffer == 1) {
            parse_autofocus(c, d);
          }
          c->ss[buffer].qbuf_info[buf_idx].dirty_buf = 1;
          ioctl(c->isp_fd, VIDIOC_MSM_ISP_ENQUEUE_BUF, &c->ss[buffer].qbuf_info[buf_idx]);
        }
        break;
      case ISP_EVENT_EOF:
        // printf("ISP_EVENT_EOF delta %f\n", (t-last_t)/1e6);
        c->last_t = t;

        pthread_mutex_lock(&c->frame_info_lock);
        c->frame_metadata[c->frame_metadata_idx] = (FrameMetadata){
          .frame_id = isp_event_data->frame_id,
          .timestamp_eof = timestamp,
          .frame_length = (unsigned int)c->cur_frame_length,
          .integ_lines = (unsigned int)c->cur_integ_lines,
          .global_gain = (unsigned int)c->cur_gain,
          .lens_pos = c->cur_lens_pos,
          .lens_sag = c->last_sag_acc_z,
          .lens_err = c->focus_err,
          .lens_true_pos = c->lens_true_pos,
          .gain_frac = c->cur_gain_frac,
        };
        c->frame_metadata_idx = (c->frame_metadata_idx+1)%METADATA_BUF_COUNT;
        pthread_mutex_unlock(&c->frame_info_lock);

        break;
      case ISP_EVENT_ERROR:
        LOGE("ISP_EVENT_ERROR! err type: 0x%08x", isp_event_data->u.error_info.err_type);
        break;
      }
    }
  }

  LOG(" ************** STOPPING **************");

  err = pthread_join(ops_thread_handle, NULL);
  assert(err == 0);

  cameras_close(s);

  for (auto &t : threads) t.join();
}

void cameras_close(MultiCameraState *s) {
  camera_close(&s->rear);
  camera_close(&s->front);
  for (int i = 0; i < FRAME_BUF_COUNT; i++) {
    visionbuf_free(&s->focus_bufs[i]);
    visionbuf_free(&s->stats_bufs[i]);
  }
  clReleaseMemObject(s->rgb_conv_roi_cl);
  clReleaseMemObject(s->rgb_conv_result_cl);
  clReleaseMemObject(s->rgb_conv_filter_cl);

  clReleaseProgram(s->prg_rgb_laplacian);
  clReleaseKernel(s->krnl_rgb_laplacian);
  delete s->sm_driver;
  delete s->sm_sensor;
  delete s->pm;
}
