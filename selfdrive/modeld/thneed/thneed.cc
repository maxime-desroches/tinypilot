#include <cassert>
#include <sys/mman.h>
#include <dlfcn.h>
#include <map>
#include <string>
#include <string.h>
#include <errno.h>
#include "thneed.h"

//#define SAVE_KERNELS
//#define SAVE_LOG

//#define RUN_DISASSEMBLER
//#define RUN_OPTIMIZER

Thneed *g_thneed = NULL;
int g_fd = -1;
map<pair<cl_kernel, int>, string> g_args;

static inline uint64_t nanos_since_boot() {
  struct timespec t;
  clock_gettime(CLOCK_BOOTTIME, &t);
  return t.tv_sec * 1000000000ULL + t.tv_nsec;
}

void hexdump(uint32_t *d, int len) {
  assert((len%4) == 0);
  printf("  dumping %p len 0x%x\n", d, len);
  for (int i = 0; i < len/4; i++) {
    if (i != 0 && (i%0x10) == 0) printf("\n");
    printf("%8x ", d[i]);
  }
  printf("\n");
}

extern "C" {

int (*my_ioctl)(int filedes, unsigned long request, void *argp) = NULL;
#undef ioctl
int ioctl(int filedes, unsigned long request, void *argp) {
  request &= 0xFFFFFFFF;  // needed on QCOM2
  if (my_ioctl == NULL) my_ioctl = reinterpret_cast<decltype(my_ioctl)>(dlsym(RTLD_NEXT, "ioctl"));
  Thneed *thneed = g_thneed;

  // save the fd
  if (request == IOCTL_KGSL_GPUOBJ_ALLOC) g_fd = filedes;

  // note that this runs always, even without a thneed object
  if (request == IOCTL_KGSL_DRAWCTXT_CREATE) {
    struct kgsl_drawctxt_create *create = (struct kgsl_drawctxt_create *)argp;
    create->flags &= ~KGSL_CONTEXT_PRIORITY_MASK;
    create->flags |= 1 << KGSL_CONTEXT_PRIORITY_SHIFT;   // priority from 1-15, 1 is max priority
    printf("IOCTL_KGSL_DRAWCTXT_CREATE: creating context with flags 0x%x\n", create->flags);
  }

  if (thneed != NULL) {
    if (request == IOCTL_KGSL_GPU_COMMAND) {
      struct kgsl_gpu_command *cmd = (struct kgsl_gpu_command *)argp;
      if (thneed->record & THNEED_RECORD) {
        thneed->timestamp = cmd->timestamp;
        thneed->context_id = cmd->context_id;
        thneed->cmds.push_back(unique_ptr<CachedCommand>(new CachedCommand(thneed, cmd)));
      }
      if (thneed->record & THNEED_DEBUG) {
        printf("IOCTL_KGSL_GPU_COMMAND(%2zu): flags: 0x%lx    context_id: %u  timestamp: %u\n",
            thneed->cmds.size(),
            cmd->flags,
            cmd->context_id, cmd->timestamp);
      }
    } else if (request == IOCTL_KGSL_GPUOBJ_SYNC) {
      struct kgsl_gpuobj_sync *cmd = (struct kgsl_gpuobj_sync *)argp;
      struct kgsl_gpuobj_sync_obj *objs = (struct kgsl_gpuobj_sync_obj *)(cmd->objs);

      if (thneed->record & THNEED_DEBUG) {
        printf("IOCTL_KGSL_GPUOBJ_SYNC count:%d ", cmd->count);
        for (int i = 0; i < cmd->count; i++) {
          printf(" -- offset:0x%lx len:0x%lx id:%d op:%d  ", objs[i].offset, objs[i].length, objs[i].id, objs[i].op);
        }
        printf("\n");
      }

      if (thneed->record & THNEED_RECORD) {
        thneed->syncobjs.push_back(string((char *)objs, sizeof(struct kgsl_gpuobj_sync_obj)*cmd->count));
      }
    } else if (request == IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID) {
      struct kgsl_device_waittimestamp_ctxtid *cmd = (struct kgsl_device_waittimestamp_ctxtid *)argp;
      if (thneed->record & THNEED_DEBUG) {
        printf("IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID: context_id: %d  timestamp: %d  timeout: %d\n",
            cmd->context_id, cmd->timestamp, cmd->timeout);
      }
    } else if (request == IOCTL_KGSL_SETPROPERTY) {
      if (thneed->record & THNEED_DEBUG) {
        struct kgsl_device_getproperty *prop = (struct kgsl_device_getproperty *)argp;
        printf("IOCTL_KGSL_SETPROPERTY: 0x%x sizebytes:%zu\n", prop->type, prop->sizebytes);
        if (thneed->record & THNEED_VERBOSE_DEBUG) {
          hexdump((uint32_t *)prop->value, prop->sizebytes);
          if (prop->type == KGSL_PROP_PWR_CONSTRAINT) {
            struct kgsl_device_constraint *constraint = (struct kgsl_device_constraint *)prop->value;
            hexdump((uint32_t *)constraint->data, constraint->size);
          }
        }
      }
    }
  }

  int ret = my_ioctl(filedes, request, argp);
  if (ret != 0) printf("ioctl returned %d with errno %d\n", ret, errno);
  return ret;
}

}

GPUMalloc::GPUMalloc(int size, int fd) {
  struct kgsl_gpuobj_alloc alloc;
  memset(&alloc, 0, sizeof(alloc));
  alloc.size = size;
  alloc.flags = 0x10000a00;
  ioctl(fd, IOCTL_KGSL_GPUOBJ_ALLOC, &alloc);
  void *addr = mmap64(NULL, alloc.mmapsize, 0x3, 0x1, fd, alloc.id*0x1000);
  assert(addr != MAP_FAILED);

  base = (uint64_t)addr;
  remaining = size;
}

GPUMalloc::~GPUMalloc() {
  // TODO: free the GPU malloced area
}

void *GPUMalloc::alloc(int size) {
  if (size > remaining) return NULL;
  remaining -= size;
  void *ret = (void*)base;
  base += (size+0xff) & (~0xFF);
  return ret;
}

CachedCommand::CachedCommand(Thneed *lthneed, struct kgsl_gpu_command *cmd) {
  thneed = lthneed;
  // Are these asserts needed?
  //assert(cmd->numcmds == 2);
  //assert(cmd->numobjs == 1);
  assert(cmd->numsyncs == 0);

  memcpy(cmds, (void *)cmd->cmdlist, sizeof(struct kgsl_command_object)*cmd->numcmds);
  memcpy(objs, (void *)cmd->objlist, sizeof(struct kgsl_command_object)*cmd->numobjs);

  memcpy(&cache, cmd, sizeof(cache));
  cache.cmdlist = (uint64_t)cmds;
  cache.objlist = (uint64_t)objs;

  for (int i = 0; i < cmd->numcmds; i++) {
    void *nn = thneed->ram->alloc(cmds[i].size);
    memcpy(nn, (void*)cmds[i].gpuaddr, cmds[i].size);
    cmds[i].gpuaddr = (uint64_t)nn;
  }

  for (int i = 0; i < cmd->numobjs; i++) {
    void *nn = thneed->ram->alloc(objs[i].size);
    memset(nn, 0, objs[i].size);
    objs[i].gpuaddr = (uint64_t)nn;
  }

  info = thneed->info;
  thneed->info.clear();
}

void CachedCommand::exec(bool wait) {
  cache.timestamp = ++thneed->timestamp;
  int ret = ioctl(thneed->fd, IOCTL_KGSL_GPU_COMMAND, &cache);

  if (wait) {
    struct kgsl_device_waittimestamp_ctxtid wait;
    wait.context_id = cache.context_id;
    wait.timestamp = cache.timestamp;
    wait.timeout = -1;

    uint64_t tb = nanos_since_boot();
    int wret = ioctl(thneed->fd, IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID, &wait);
    uint64_t te = nanos_since_boot();

    if (thneed->record & THNEED_DEBUG) printf("exec %d wait %d after %lu us\n", ret, wret, (te-tb)/1000);
  } else {
    if (thneed->record & THNEED_DEBUG) printf("CachedCommand::exec got %d\n", ret);
  }

  if (thneed->record & THNEED_VERBOSE_DEBUG) {
    for (auto it : info) {
      printf("%s\n", it.c_str());
    }
    #ifdef RUN_DISASSEMBLER
      disassemble();
    #endif
  }

  assert(ret == 0);
}

Thneed::Thneed() {
  assert(g_fd != -1);
  fd = g_fd;
  ram = make_unique<GPUMalloc>(0x40000, fd);
  record = THNEED_RECORD;
  timestamp = -1;
  g_thneed = this;
}

void Thneed::stop() {
  record = 0;
}

void Thneed::execute(float **finputs, float *foutput, bool slow) {
  uint64_t tb, te;
  if (record & THNEED_DEBUG) tb = nanos_since_boot();

  #ifdef SAVE_LOG
    char fn[0x100];
    snprintf(fn, sizeof(fn), "/tmp/thneed_log_%d", timestamp);
    FILE *f = fopen(fn, "wb");
  #endif

  // ****** copy inputs
  for (int idx = 0; idx < inputs.size(); ++idx) {
    size_t sz;
    clGetMemObjectInfo(inputs[idx], CL_MEM_SIZE, sizeof(sz), &sz, NULL);

    #ifdef SAVE_LOG
      fwrite(&sz, 1, sizeof(sz), f);
      fwrite(finputs[idx], 1, sz, f);
    #endif

    if (record & THNEED_DEBUG) printf("copying %lu -- %p -> %p\n", sz, finputs[idx], inputs[idx]);
    clEnqueueWriteBuffer(command_queue, inputs[idx], CL_TRUE, 0, sz, finputs[idx], 0, NULL, NULL);
  }

  // ****** set power constraint
  struct kgsl_device_constraint_pwrlevel pwrlevel;
  pwrlevel.level = KGSL_CONSTRAINT_PWR_MAX;

  struct kgsl_device_constraint constraint;
  constraint.type = KGSL_CONSTRAINT_PWRLEVEL;
  constraint.context_id = context_id;
  constraint.data = (void*)&pwrlevel;
  constraint.size = sizeof(pwrlevel);

  struct kgsl_device_getproperty prop;
  prop.type = KGSL_PROP_PWR_CONSTRAINT;
  prop.value = (void*)&constraint;
  prop.sizebytes = sizeof(constraint);
  int ret = ioctl(fd, IOCTL_KGSL_SETPROPERTY, &prop);
  assert(ret == 0);

  // ****** run commands
  int i = 0;
  for (auto it = cmds.begin(); it != cmds.end(); ++it) {
    ++i;
    if (record & THNEED_DEBUG) printf("run %2d @ %7lu us: ", i, (nanos_since_boot()-tb)/1000);
    (*it)->exec((i == cmds.size()) || slow);
  }

  // ****** sync objects
  for (auto it = syncobjs.begin(); it != syncobjs.end(); ++it) {
    struct kgsl_gpuobj_sync cmd;

    cmd.objs = (uint64_t)it->data();
    cmd.obj_len = it->length();
    cmd.count = it->length() / sizeof(struct kgsl_gpuobj_sync_obj);

    ret = ioctl(fd, IOCTL_KGSL_GPUOBJ_SYNC, &cmd);
    assert(ret == 0);
  }

  // ****** copy outputs
  size_t sz;
  clGetMemObjectInfo(output, CL_MEM_SIZE, sizeof(sz), &sz, NULL);
  if (record & THNEED_DEBUG) printf("copying %lu for output %p -> %p\n", sz, output, foutput);
  clEnqueueReadBuffer(command_queue, output, CL_TRUE, 0, sz, foutput, 0, NULL, NULL);

  #ifdef SAVE_LOG
    fwrite(&sz, 1, sizeof(sz), f);
    fwrite(foutput, 1, sz, f);
    fclose(f);
  #endif

  // ****** unset power constraint
  constraint.type = KGSL_CONSTRAINT_NONE;
  constraint.data = NULL;
  constraint.size = 0;

  ret = ioctl(fd, IOCTL_KGSL_SETPROPERTY, &prop);
  assert(ret == 0);

  if (record & THNEED_DEBUG) {
    te = nanos_since_boot();
    printf("model exec in %lu us\n", (te-tb)/1000);
  }
}

// TODO: with a different way of getting the input and output buffers, we don't have to intercept CL at all

cl_int (*my_clSetKernelArg)(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value) = NULL;
cl_int thneed_clSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value) {
  if (my_clSetKernelArg == NULL) my_clSetKernelArg = reinterpret_cast<decltype(my_clSetKernelArg)>(dlsym(RTLD_NEXT, "REAL_clSetKernelArg"));
  if (arg_value != NULL) {
    g_args[make_pair(kernel, arg_index)] = string((char*)arg_value, arg_size);
  } else {
    g_args[make_pair(kernel, arg_index)] = string("");
  }
  cl_int ret = my_clSetKernelArg(kernel, arg_index, arg_size, arg_value);
  return ret;
}

cl_int (*my_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *) = NULL;
cl_int thneed_clEnqueueNDRangeKernel(cl_command_queue command_queue,
  cl_kernel kernel,
  cl_uint work_dim,
  const size_t *global_work_offset,
  const size_t *global_work_size,
  const size_t *local_work_size,
  cl_uint num_events_in_wait_list,
  const cl_event *event_wait_list,
  cl_event *event) {

  if (my_clEnqueueNDRangeKernel == NULL) my_clEnqueueNDRangeKernel = reinterpret_cast<decltype(my_clEnqueueNDRangeKernel)>(dlsym(RTLD_NEXT, "REAL_clEnqueueNDRangeKernel"));
  Thneed *thneed = g_thneed;

  // SNPE doesn't use these
  assert(num_events_in_wait_list == 0);
  assert(global_work_offset == NULL);
  assert(event_wait_list == NULL);

  cl_int ret = 0;
  if (thneed != NULL && thneed->record & THNEED_RECORD) {
    thneed->command_queue = command_queue;
    clGetKernelInfo(kernel, CL_KERNEL_CONTEXT, sizeof(thneed->context), &thneed->context, NULL);

    // if we are recording, we don't actually enqueue the kernel
    thneed->kq.push_back(unique_ptr<CLQueuedKernel>(new CLQueuedKernel(thneed, kernel, work_dim, global_work_size, local_work_size)));
    *event = NULL;
  } else {
    ret = my_clEnqueueNDRangeKernel(command_queue, kernel, work_dim,
      global_work_offset, global_work_size, local_work_size,
      num_events_in_wait_list, event_wait_list, event);
  }

  /*uint64_t tb = nanos_since_boot();
  clWaitForEvents(1, event);
  uint64_t te = nanos_since_boot();
  if (thneed != NULL && thneed->record & THNEED_DEBUG) {
    printf("  wait %lu us\n", (te-tb)/1000);
  }*/

  return ret;
}

cl_int (*my_clFinish)(cl_command_queue) = NULL;
cl_int thneed_clFinish(cl_command_queue command_queue) {
  if (my_clFinish == NULL) my_clFinish = reinterpret_cast<decltype(my_clFinish)>(dlsym(RTLD_NEXT, "REAL_clFinish"));
  Thneed *thneed = g_thneed;

  if (thneed != NULL && thneed->record & THNEED_RECORD) {
    #ifdef RUN_OPTIMIZER
      thneed->optimize();
    #endif
    printf("clFinish: running queued kernels\n");
    for (auto &k : thneed->kq) {
      k->exec();
    }
    thneed->kq.clear();
  }

  return my_clFinish(command_queue);
}

#ifdef SAVE_KERNELS
map<cl_program, string> program_source;

cl_program (*my_clCreateProgramWithSource)(cl_context context, cl_uint count, const char **strings, const size_t *lengths, cl_int *errcode_ret) = NULL;
cl_program thneed_clCreateProgramWithSource(cl_context context, cl_uint count, const char **strings, const size_t *lengths, cl_int *errcode_ret) {
  if (my_clCreateProgramWithSource == NULL) my_clCreateProgramWithSource = reinterpret_cast<decltype(my_clCreateProgramWithSource)>(dlsym(RTLD_NEXT, "REAL_clCreateProgramWithSource"));
  assert(count == 1);
  size_t my_lengths[1];
  my_lengths[0] = lengths[0];

  char fn[0x100];
  snprintf(fn, sizeof(fn), "/tmp/program_%zu.cl", strlen(strings[0]));
  FILE *f = fopen(fn, "wb");
  fprintf(f, "%s", strings[0]);
  fclose(f);

  char tmp[0x10000];
  memset(tmp, 0, sizeof(tmp));
  snprintf(fn, sizeof(fn), "/tmp/patched_%zu.cl", strlen(strings[0]));
  FILE *g = fopen(fn, "rb");
  if (g != NULL) {
    printf("LOADING PATCHED PROGRAM %s\n", fn);
    fread(tmp, 1, sizeof(tmp), g);
    fclose(g);
    strings[0] = tmp;
    my_lengths[0] = strlen(tmp);
  }

  cl_program ret = my_clCreateProgramWithSource(context, count, strings, my_lengths, errcode_ret);
  program_source[ret] = strings[0];
  return ret;
}
#endif

void *dlsym(void *handle, const char *symbol) {
  // TODO: Find dlsym in a better way. Currently this is hand looked up in libdl.so
#if defined QCOM
  void *(*my_dlsym)(void *handle, const char *symbol) = (void *(*)(void *handle, const char *symbol))((uintptr_t)dlopen-0x2d4);
#elif defined QCOM2
  void *(*my_dlsym)(void *handle, const char *symbol) = (void *(*)(void *handle, const char *symbol))((uintptr_t)dlopen+0x138);
#else
  #error "Unsupported platform for thneed"
#endif
  if (memcmp("REAL_", symbol, 5) == 0) {
    return my_dlsym(handle, symbol+5);
  } else if (strcmp("clFinish", symbol) == 0) {
    return (void*)thneed_clFinish;
  } else if (strcmp("clEnqueueNDRangeKernel", symbol) == 0) {
    return (void*)thneed_clEnqueueNDRangeKernel;
  } else if (strcmp("clSetKernelArg", symbol) == 0) {
    return (void*)thneed_clSetKernelArg;
#ifdef SAVE_KERNELS
  } else if (strcmp("clCreateProgramWithSource", symbol) == 0) {
    return (void*)thneed_clCreateProgramWithSource;
#endif
  } else {
    return my_dlsym(handle, symbol);
  }
}

CLQueuedKernel::CLQueuedKernel(Thneed *lthneed,
                               cl_kernel kernel,
                               cl_uint _work_dim,
                               const size_t *_global_work_size,
                               const size_t *_local_work_size) {
  thneed = lthneed;
  work_dim = _work_dim;
  for (int i = 0; i < work_dim; i++) {
    global_work_size[i] = _global_work_size[i];
    local_work_size[i] = _local_work_size[i];
  }

  char _name[0x100];
  clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, sizeof(_name), _name, NULL);
  name = string(_name);
  clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, sizeof(num_args), &num_args, NULL);

  // get args
  for (int i = 0; i < num_args; i++) {
    char arg_name[0x100];
    clGetKernelArgInfo(kernel, i, CL_KERNEL_ARG_NAME, sizeof(arg_name), arg_name, NULL);
    arg_names.push_back(string(arg_name));
    args.push_back(g_args[make_pair(kernel, i)]);
  }

  // get program
  clGetKernelInfo(kernel, CL_KERNEL_PROGRAM, sizeof(program), &program, NULL);
}

int CLQueuedKernel::get_arg_num(const char *search_arg_name) {
  for (int i = 0; i < num_args; i++) {
    if (arg_names[i] == search_arg_name) return i;
  }
  printf("failed to find %s in %s\n", search_arg_name, name.c_str());
  assert(false);
}

int CLQueuedKernel::exec() {
  // create the exec kernel
  cl_kernel kernel = clCreateKernel(program, name.c_str(), NULL);
  for (int j = 0; j < num_args; j++) {
    if (args[j].size() != 0) {
      thneed_clSetKernelArg(kernel, j, args[j].size(), args[j].data());
    } else {
      thneed_clSetKernelArg(kernel, j, 0, NULL);
    }
  }

  // check for global inputs/outputs
  for (int i = 0; i < num_args; i++) {
    if (name == "zero_pad_image_float" && arg_names[i] == "input") {
      cl_mem mem;
      memcpy(&mem, (void*)args[i].data(), sizeof(mem));
      thneed->inputs.push_back(mem);
    }

    if (name == "image2d_to_buffer_float" && arg_names[i] == "output") {
      cl_mem mem;
      memcpy(&mem, (void*)args[i].data(), sizeof(mem));
      thneed->output = mem;
    }
  }

  // debug
  if (thneed != NULL && thneed->record & THNEED_DEBUG) {
    char tbuf[0x100];
    char *buf = tbuf;
    buf += sprintf(buf, "%p %56s -- ", kernel, name.c_str());
    for (int i = 0; i < work_dim; i++) {
      buf += sprintf(buf, "%4zu ", global_work_size[i]);
    }
    buf += sprintf(buf, " -- ");
    for (int i = 0; i < work_dim; i++) {
      buf += sprintf(buf, "%4zu ", local_work_size[i]);
    }
    thneed->info.push_back(tbuf);
    printf("%s\n", tbuf);
  }

  // verbose debug
  if (thneed != NULL && thneed->record & THNEED_VERBOSE_DEBUG) {
    for (int i = 0; i < num_args; i++) {
      char arg_type[0x100];
      clGetKernelArgInfo(kernel, i, CL_KERNEL_ARG_TYPE_NAME, sizeof(arg_type), arg_type, NULL);
      string arg = args[i];
      printf("  %s %s", arg_type, arg_names[i].c_str());
      void *arg_value = (void*)arg.data();
      int arg_size = arg.size();
      if (arg_size == 1) {
        printf(" = %d", *((char*)arg_value));
      } else if (arg_size == 2) {
        printf(" = %d", *((short*)arg_value));
      } else if (arg_size == 4) {
        if (strcmp(arg_type, "float") == 0) {
          printf(" = %f", *((float*)arg_value));
        } else {
          printf(" = %d", *((int*)arg_value));
        }
      } else if (arg_size == 8) {
        cl_mem val = (cl_mem)(*((uintptr_t*)arg_value));
        printf(" = %p", val);
        if (val != NULL) {
          if (strcmp("image2d_t", arg_type) == 0 || strcmp("image1d_t", arg_type) == 0) {
            cl_image_format format;
            size_t width, height, depth, array_size, row_pitch, slice_pitch;
            cl_mem buf;
            clGetImageInfo(val, CL_IMAGE_FORMAT, sizeof(format), &format, NULL);
            assert(format.image_channel_order == CL_RGBA);
            assert(format.image_channel_data_type == CL_HALF_FLOAT);
            clGetImageInfo(val, CL_IMAGE_WIDTH, sizeof(width), &width, NULL);
            clGetImageInfo(val, CL_IMAGE_HEIGHT, sizeof(height), &height, NULL);
            clGetImageInfo(val, CL_IMAGE_ROW_PITCH, sizeof(row_pitch), &row_pitch, NULL);
            clGetImageInfo(val, CL_IMAGE_DEPTH, sizeof(depth), &depth, NULL);
            clGetImageInfo(val, CL_IMAGE_ARRAY_SIZE, sizeof(array_size), &array_size, NULL);
            clGetImageInfo(val, CL_IMAGE_SLICE_PITCH, sizeof(slice_pitch), &slice_pitch, NULL);
            assert(depth == 0);
            assert(array_size == 0);
            assert(slice_pitch == 0);

            clGetImageInfo(val, CL_IMAGE_BUFFER, sizeof(buf), &buf, NULL);
            size_t sz;
            clGetMemObjectInfo(buf, CL_MEM_SIZE, sizeof(sz), &sz, NULL);
            printf(" image %zu x %zu rp %zu @ %p buffer %zu", width, height, row_pitch, buf, sz);
          } else {
            size_t sz;
            clGetMemObjectInfo(val, CL_MEM_SIZE, sizeof(sz), &sz, NULL);
            printf(" buffer %zu", sz);
          }
        }
      }
      printf("\n");
    }
  }

  int ret = my_clEnqueueNDRangeKernel(thneed->command_queue,
    kernel, work_dim, NULL, global_work_size, local_work_size, 0, NULL, NULL);

  return ret;
}

