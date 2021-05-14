#include "selfdrive/common/util.h"

#include <errno.h>

#include <atomic>
#include <cassert>
#include <csignal>
#include <cstring>
#include <fstream>
#include <sstream>

#ifdef __linux__
#include <sys/prctl.h>
#include <sys/syscall.h>
#ifndef __USE_GNU
#define __USE_GNU
#endif
#include <sched.h>
#endif  // __linux__

void set_thread_name(const char* name) {
#ifdef __linux__
  // pthread_setname_np is dumb (fails instead of truncates)
  prctl(PR_SET_NAME, (unsigned long)name, 0, 0, 0);
#endif
}

int set_realtime_priority(int level) {
#ifdef __linux__
  long tid = syscall(SYS_gettid);

  // should match python using chrt
  struct sched_param sa;
  memset(&sa, 0, sizeof(sa));
  sa.sched_priority = level;
  return sched_setscheduler(tid, SCHED_FIFO, &sa);
#else
  return -1;
#endif
}

int set_core_affinity(int core) {
#ifdef __linux__
  long tid = syscall(SYS_gettid);
  cpu_set_t rt_cpu;

  CPU_ZERO(&rt_cpu);
  CPU_SET(core, &rt_cpu);
  return sched_setaffinity(tid, sizeof(rt_cpu), &rt_cpu);
#else
  return -1;
#endif
}

namespace util {

std::string read_file(const std::string& fn) {
  std::ifstream ifs(fn, std::ios::binary | std::ios::ate);
  if (ifs) {
    std::ifstream::pos_type pos = ifs.tellg();
    if (pos != std::ios::beg) {
      std::string result;
      result.resize(pos);
      ifs.seekg(0, std::ios::beg);
      ifs.read(result.data(), pos);
      if (ifs) {
        return result;
      }
    }
  }
  ifs.close();

  // fallback for files created on read, e.g. procfs
  std::ifstream f(fn);
  std::stringstream buffer;
  buffer << f.rdbuf();
  return buffer.str();
}

int read_files_in_dir(std::string path, std::map<std::string, std::string> *contents) {
  DIR *d = opendir(path.c_str());
  if (!d) return -1;

  struct dirent *de = NULL;
  while ((de = readdir(d))) {
    if (isalnum(de->d_name[0])) {
      (*contents)[de->d_name] = util::read_file(path + "/" + de->d_name);
    }
  }

  closedir(d);
  return 0;
}

int write_file(const char* path, const void* data, size_t size, int flags, mode_t mode) {
  int fd = open(path, flags, mode);
  if (fd == -1) {
    return -1;
  }
  ssize_t n = write(fd, data, size);
  close(fd);
  return (n >= 0 && (size_t)n == size) ? 0 : -1;
}

std::string readlink(const std::string &path) {
  char buff[4096];
  ssize_t len = ::readlink(path.c_str(), buff, sizeof(buff)-1);
  if (len != -1) {
    buff[len] = '\0';
    return std::string(buff);
  }
  return "";
}

bool file_exists(const std::string& fn) {
  std::ifstream f(fn);
  return f.good();
}

std::string getenv_default(const char* env_var, const char * suffix, const char* default_val) {
  const char* env_val = getenv(env_var);
  if (env_val != NULL){
    return std::string(env_val) + std::string(suffix);
  } else {
    return std::string(default_val);
  }
}

std::string tohex(const uint8_t *buf, size_t buf_size) {
  std::unique_ptr<char[]> hexbuf(new char[buf_size * 2 + 1]);
  for (size_t i = 0; i < buf_size; i++) {
    sprintf(&hexbuf[i * 2], "%02x", buf[i]);
  }
  hexbuf[buf_size * 2] = 0;
  return std::string(hexbuf.get(), hexbuf.get() + buf_size * 2);
}

std::string base_name(std::string const &path) {
  size_t pos = path.find_last_of("/");
  if (pos == std::string::npos) return path;
  return path.substr(pos + 1);
}

std::string dir_name(std::string const &path) {
  size_t pos = path.find_last_of("/");
  if (pos == std::string::npos) return "";
  return path.substr(0, pos);
}

}  // namespace util


// class ExitHandler

struct ExitHandleHelper {
  static void set_do_exit(int sig) {
#ifndef __APPLE__
    power_failure = (sig == SIGPWR);
#endif
    do_exit = true;
  }
  inline static std::atomic<bool> do_exit = false;
  inline static std::atomic<bool> power_failure = false;
};

ExitHandler::ExitHandler() {
  std::signal(SIGINT, (sighandler_t)ExitHandleHelper::set_do_exit);
  std::signal(SIGTERM, (sighandler_t)ExitHandleHelper::set_do_exit);
#ifndef __APPLE__
  std::signal(SIGPWR, (sighandler_t)ExitHandleHelper::set_do_exit);
#endif
}

ExitHandler::operator bool() { 
  return ExitHandleHelper::do_exit; 
}

ExitHandler& ExitHandler::operator=(bool v) {
  ExitHandleHelper::do_exit = v;
  return *this;
}

bool ExitHandler::powerFailure() {
  return ExitHandleHelper::power_failure;
}
