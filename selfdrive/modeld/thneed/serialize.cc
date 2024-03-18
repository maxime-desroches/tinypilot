#include <cassert>
#include <set>
#include "tobiaslocker/base64.hpp"
#include "nlohmann/json.hpp"
#include "common/util.h"
#include "common/clutil.h"
#include "common/swaglog.h"
#include "selfdrive/modeld/thneed/thneed.h"
using namespace nlohmann;

extern map<cl_program, string> g_program_source;

void Thneed::load(const char *filename) {
  LOGD("Thneed::load: loading from %s\n", filename);

  string buf = util::read_file(filename);
  int jsz = *(int *)buf.data();
  string jsonerr;
  string jj(buf.data() + sizeof(int), jsz);
  json jdat = json::parse(jj);

  map<cl_mem, cl_mem> real_mem;
  real_mem[NULL] = NULL;

  int ptr = sizeof(int)+jsz;
  for (auto &obj : jdat["objects"]) {
    auto mobj = obj;
    int sz = mobj["size"].template get<int>();
    cl_mem clbuf = NULL;
    if (mobj["buffer_id"].template get<std::string>().size() > 0) {
      // image buffer must already be allocated
      clbuf = real_mem[*(cl_mem*)(base64::decode_into<std::vector<std::uint8_t>>(mobj["buffer_id"].template get<std::string>()).data())];
      assert(mobj["needs_load"].template get<bool>() == false);
    } else {
      if (mobj["needs_load"].template get<bool>()) {
        clbuf = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, sz, &buf[ptr], NULL);
        if (debug >= 1) printf("loading %p %d @ 0x%X\n", clbuf, sz, ptr);
        ptr += sz;
      } else {
        // TODO: is there a faster way to init zeroed out buffers?
        void *host_zeros = calloc(sz, 1);
        clbuf = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, sz, host_zeros, NULL);
        free(host_zeros);
      }
    }
    assert(clbuf != NULL);

    if (mobj["arg_type"] == "image2d_t" || mobj["arg_type"] == "image1d_t") {
      cl_image_desc desc = {0};
      desc.image_type = (mobj["arg_type"] == "image2d_t") ? CL_MEM_OBJECT_IMAGE2D : CL_MEM_OBJECT_IMAGE1D_BUFFER;
      desc.image_width = mobj["width"].template get<int>();
      desc.image_height = mobj["height"].template get<int>();
      desc.image_row_pitch = mobj["row_pitch"].template get<int>();
      assert(sz == desc.image_height*desc.image_row_pitch);
#ifdef QCOM2
      desc.buffer = clbuf;
#else
      // TODO: we are creating unused buffers on PC
      clReleaseMemObject(clbuf);
#endif
      cl_image_format format = {0};
      format.image_channel_order = CL_RGBA;
      format.image_channel_data_type = mobj["float32"].template get<bool>() ? CL_FLOAT : CL_HALF_FLOAT;

      cl_int errcode;

#ifndef QCOM2
      if (mobj["needs_load"].template get<bool>()) {
        clbuf = clCreateImage(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, &format, &desc, &buf[ptr-sz], &errcode);
      } else {
        clbuf = clCreateImage(context, CL_MEM_READ_WRITE, &format, &desc, NULL, &errcode);
      }
#else
      clbuf = clCreateImage(context, CL_MEM_READ_WRITE, &format, &desc, NULL, &errcode);
#endif
      if (clbuf == NULL) {
        LOGE("clError: %s create image %zux%zu rp %zu with buffer %p\n", cl_get_error_string(errcode),
             desc.image_width, desc.image_height, desc.image_row_pitch, desc.buffer);
      }
      assert(clbuf != NULL);
    }

    real_mem[*(cl_mem*)(base64::decode_into<std::vector<std::uint8_t>>(mobj["id"].template get<std::string>()).data())] = clbuf;
  }

  map<string, cl_program> g_programs;
  for (const auto &[name, source] : jdat["programs"].items()) {
    if (debug >= 1) printf("building %s with size %zu\n", name.c_str(), source.template get<std::string>().size());
    g_programs[name] = cl_program_from_source(context, device_id, source.template get<std::string>());
  }

  for (auto &obj : jdat["inputs"]) {
    auto mobj = obj;
    int sz = mobj["size"].template get<int>();
    cl_mem aa = real_mem[*(cl_mem*)(base64::decode_into<std::vector<std::uint8_t>>(mobj["buffer_id"].template get<std::string>()).data())];
    input_clmem.push_back(aa);
    input_sizes.push_back(sz);
    LOGD("Thneed::load: adding input %s with size %d\n", mobj["name"].template get<std::string>().data(), sz);

    cl_int cl_err;
    void *ret = clEnqueueMapBuffer(command_queue, aa, CL_TRUE, CL_MAP_WRITE, 0, sz, 0, NULL, NULL, &cl_err);
    if (cl_err != CL_SUCCESS) LOGE("clError: %s map %p %d\n", cl_get_error_string(cl_err), aa, sz);
    assert(cl_err == CL_SUCCESS);
    inputs.push_back(ret);
  }

  for (auto &obj : jdat["outputs"]) {
    auto mobj = obj;
    int sz = mobj["size"].template get<int>();
    LOGD("Thneed::save: adding output with size %d\n", sz);
    // TODO: support multiple outputs
    output = real_mem[*(cl_mem*)(base64::decode_into<std::vector<std::uint8_t>>(mobj["buffer_id"].template get<std::string>()).data())];
    assert(output != NULL);
  }

  for (auto &obj : jdat["binaries"]) {
    string name = obj["name"].template get<std::string>();
    size_t length = obj["length"].template get<int>();
    if (debug >= 1) printf("binary %s with size %zu\n", name.c_str(), length);
    g_programs[name] = cl_program_from_binary(context, device_id, (const uint8_t*)&buf[ptr], length);
    ptr += length;
  }

  for (auto &obj : jdat["kernels"]) {
    auto gws = obj["global_work_size"];
    auto lws = obj["local_work_size"];
    auto kk = shared_ptr<CLQueuedKernel>(new CLQueuedKernel(this));

    kk->name = obj["name"].template get<std::string>();
    kk->program = g_programs[kk->name];
    kk->work_dim = obj["work_dim"].template get<int>();
    for (int i = 0; i < kk->work_dim; i++) {
      kk->global_work_size[i] = gws[i].template get<int>();
      kk->local_work_size[i] = lws[i].template get<int>();
    }
    kk->num_args = obj["num_args"].template get<int>();
    for (int i = 0; i < kk->num_args; i++) {
      string arg = obj["args"][i].template get<std::string>();
      int arg_size = obj["args_size"][i].template get<int>();
      kk->args_size.push_back(arg_size);
      if (arg_size == 8) {
        cl_mem val = *(cl_mem*)(base64::decode_into<std::vector<std::uint8_t>>(arg).data());
        val = real_mem[val];
        kk->args.push_back(string((char*)&val, sizeof(val)));
      } else {
        kk->args.push_back(arg);
      }
    }
    kq.push_back(kk);
  }

  clFinish(command_queue);
}
