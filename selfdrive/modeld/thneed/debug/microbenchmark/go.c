#include <stdio.h>
#include <CL/cl.h>
#include <assert.h>
#include <time.h>

/*
0x7f7e8a6380                 convolution_horizontal_reduced_reads_1x1 --   88    4    8  --    4    4    8
  image2d_t input = 0x7f7f490b00 image 8448 x 8 rp 67840
  short startPackedInputChannel = 0
  short numPackedInputChannelsForGroup = 528
  short totalNumPackedInputChannels = 528
  short packedOuputChannelOffset = 0
  short totalNumPackedOutputChannels = 88
  image2d_t weights = 0x7f7f52fb80 image 2112 x 88 rp 16896
  float* biases = 0x7f7f564d80 buffer 1408
  short filterSizeX = 1
  short filterSizeY = 1
  image2d_t output = 0x7f7f490e80 image 1408 x 8 rp 11264
  short paddingX = 0
  short paddingY = 0
  short strideX = 1
  short strideY = 1
  short neuron = 0
  float a = 1.000000
  float b = 1.000000
  float min_clamp = 0.000000
  float max_clamp = 0.000000
  float* parameters = 0x0
  float* batchNormBiases = 0x0
  short numOutputColumns = 16
*/

static inline uint64_t nanos_since_boot() {
  struct timespec t;
  clock_gettime(CLOCK_BOOTTIME, &t);
  return t.tv_sec * 1000000000ULL + t.tv_nsec;
}

int main(int argc, char *argv[]) {
  cl_int err;

  // cl init
  cl_device_id device_id;
  cl_context context;
  cl_command_queue q;
  {
    cl_platform_id platform_id[2];
    cl_uint num_devices;
    cl_uint num_platforms;

    err = clGetPlatformIDs(sizeof(platform_id)/sizeof(cl_platform_id), platform_id, &num_platforms);
    assert(err == 0);

    err = clGetDeviceIDs(platform_id[0], CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &num_devices);
    assert(err == 0);

    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    assert(err == 0);

    q = clCreateCommandQueue(context, device_id, 0, &err);
    assert(err == 0);
  }
  printf("cl ready\n");

  char tmp[0x10000];
  memset(tmp, 0, sizeof(tmp));
  FILE *f = fopen(argv[1], "rb");
  fread(tmp, 1, sizeof(tmp), f);
  fclose(f);

  const char *strings[1];
  size_t lengths[1];
  strings[0] = tmp;
  lengths[0] = strlen(tmp);

  cl_program prog = clCreateProgramWithSource(context, 1, strings, lengths, &err);
  assert(err == 0);
  printf("creating program\n");

  err = clBuildProgram(prog, 1, &device_id, "-D AVANTE_IS_GPU_A530_64", NULL, NULL);
  assert(err == 0);
  printf("built program\n");

  cl_kernel kern = clCreateKernel(prog, "convolution_horizontal_reduced_reads_1x1", &err);
  assert(err == 0);
  printf("creating kernel\n");

  /*
    image2d_t input = 0x7f7f490b00 image 8448 x 8 rp 67840
    image2d_t weights = 0x7f7f52fb80 image 2112 x 88 rp 16896
    float* biases = 0x7f7f564d80 buffer 1408
    image2d_t output = 0x7f7f490e80 image 1408 x 8 rp 11264
  */

  cl_mem input;
  cl_mem weights;
  cl_mem biases;
  cl_mem outputs;

  cl_image_format fmt;
  fmt.image_channel_order = CL_RGBA;
  fmt.image_channel_data_type = CL_HALF_FLOAT;

  cl_image_desc desc;
  desc.image_type = CL_MEM_OBJECT_IMAGE2D;
  desc.image_depth = 0; desc.image_slice_pitch = 0; desc.num_mip_levels = 0; desc.num_samples = 0;
  desc.buffer = NULL;

  biases = clCreateBuffer(context, CL_MEM_READ_WRITE, 1408, NULL, &err);
  assert(err == 0);

  desc.image_width = 8448; desc.image_height = 8; desc.image_row_pitch = 67840;
  desc.buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, desc.image_height * desc.image_row_pitch, NULL, &err);
  assert(err == 0);
  input = clCreateImage(context, CL_MEM_READ_WRITE, &fmt, &desc, NULL, &err);
  assert(err == 0);

  desc.image_width = 2112; desc.image_height = 88; desc.image_row_pitch = 16896;
  desc.buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, desc.image_height * desc.image_row_pitch, NULL, &err);
  assert(err == 0);
  weights = clCreateImage(context, CL_MEM_READ_WRITE, &fmt, &desc, NULL, &err);
  assert(err == 0);

  desc.image_width = 1408; desc.image_height = 8; desc.image_row_pitch = 11264;
  desc.buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, desc.image_height * desc.image_row_pitch, NULL, &err);
  assert(err == 0);
  outputs = clCreateImage(context, CL_MEM_READ_WRITE, &fmt, &desc, NULL, &err);
  assert(err == 0);

  void *n = NULL;
  uint16_t v;
  float fl;

  clSetKernelArg(kern, 0, sizeof(cl_mem), &input);
  v = 0; clSetKernelArg(kern, 1, sizeof(v), &v);
  v = 528; clSetKernelArg(kern, 2, sizeof(v), &v);
  v = 528; clSetKernelArg(kern, 3, sizeof(v), &v);
  v = 0; clSetKernelArg(kern, 4, sizeof(v), &v);
  v = 88; clSetKernelArg(kern, 5, sizeof(v), &v);
  clSetKernelArg(kern, 6, sizeof(cl_mem), &weights);
  clSetKernelArg(kern, 7, sizeof(cl_mem), &biases);
  v = 1; clSetKernelArg(kern, 8, sizeof(v), &v);
  v = 1; clSetKernelArg(kern, 9, sizeof(v), &v);
  clSetKernelArg(kern, 10, sizeof(cl_mem), &outputs);
  v = 0; clSetKernelArg(kern, 11, sizeof(v), &v);
  v = 0; clSetKernelArg(kern, 12, sizeof(v), &v);
  v = 1; clSetKernelArg(kern, 13, sizeof(v), &v);
  v = 1; clSetKernelArg(kern, 14, sizeof(v), &v);
  v = 0; clSetKernelArg(kern, 15, sizeof(v), &v);
  fl = 1.0; clSetKernelArg(kern, 16, sizeof(fl), &fl);
  fl = 0.0; clSetKernelArg(kern, 17, sizeof(fl), &fl);
  fl = 0.0; clSetKernelArg(kern, 18, sizeof(fl), &fl);
  fl = 0.0; clSetKernelArg(kern, 19, sizeof(fl), &fl);
  clSetKernelArg(kern, 20, sizeof(n), &n);
  clSetKernelArg(kern, 21, sizeof(n), &n);
  v = 16; clSetKernelArg(kern, 22, sizeof(v), &v);
  
  size_t global_work_size[3] = {88, 4, 8};
  size_t local_work_size[3] = {4, 4, 8};

  for (int i = 0; i < 20; i++) {
    cl_event event;
    clEnqueueNDRangeKernel(q, kern, 3, NULL, global_work_size, local_work_size, 0, NULL, &event);

    uint64_t tb = nanos_since_boot();
    clWaitForEvents(1, &event);
    uint64_t te = nanos_since_boot();
    printf("%2d: wait %lu us\n", i, (te-tb)/1000);
  }

  return 0;
}

