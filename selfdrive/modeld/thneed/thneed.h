#pragma once

#include <stdint.h>
#include "include/msm_kgsl.h"
#include <vector>

class Thneed;

class GPUMalloc {
  public:
    GPUMalloc(int size, int fd);
    void *alloc(int size);
  private:
    uint64_t base;
    int remaining;
};

class CachedCommand {
  public:
    CachedCommand(Thneed *lthneed, struct kgsl_gpu_command *cmd);
    void exec(bool wait);
  private:
    struct kgsl_gpu_command cache;
    struct kgsl_command_object cmds[2];
    struct kgsl_command_object objs[1];
    Thneed *thneed;
};

class Thneed {
  public:
    Thneed();
    void stop();
    void execute();

    // protected?
    int record;
    int timestamp;
    GPUMalloc *ram;
    std::vector<CachedCommand *> cmds;
    std::vector<std::pair<int, struct kgsl_gpuobj_sync_obj *> > syncobjs;
    int fd;
};

