#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <csignal>
#include <android/log.h>

//#include <log/log.h>
#include <log/logger.h>
#include <log/logprint.h>

#include "common/timing.h"
#include "messaging.hpp"

volatile sig_atomic_t do_exit = 0;
static void set_do_exit(int sig) {
  do_exit = 1;
}

int main() {
  int err;

  // setup signal handlers
  signal(SIGINT, (sighandler_t)set_do_exit);
  signal(SIGTERM, (sighandler_t)set_do_exit);

  struct logger_list *logger_list = android_logger_list_alloc(ANDROID_LOG_RDONLY, 0, 0);
  assert(logger_list);
  struct logger *main_logger = android_logger_open(logger_list, LOG_ID_MAIN);
  assert(main_logger);
  struct logger *radio_logger = android_logger_open(logger_list, LOG_ID_RADIO);
  assert(radio_logger);
  struct logger *system_logger = android_logger_open(logger_list, LOG_ID_SYSTEM);
  assert(system_logger);
  struct logger *crash_logger = android_logger_open(logger_list, LOG_ID_CRASH);
  assert(crash_logger);
  struct logger *kernel_logger = android_logger_open(logger_list, (log_id_t)5); // LOG_ID_KERNEL
  assert(kernel_logger);
  PubMaster pm({"androidLog"});

  while (!do_exit) {
    log_msg log_msg;
    err = android_logger_list_read(logger_list, &log_msg);
    if (err <= 0) {
      break;
    }

    AndroidLogEntry entry;
    err = android_log_processLogBuffer(&log_msg.entry_v1, &entry);
    if (err < 0) {
      continue;
    }

    MessageBuilder msg;
    auto androidEntry = msg.initEvent().initAndroidLog();
    androidEntry.setId(log_msg.id());
    androidEntry.setTs(entry.tv_sec * 1000000000ULL + entry.tv_nsec);
    androidEntry.setPriority(entry.priority);
    androidEntry.setPid(entry.pid);
    androidEntry.setTid(entry.tid);
    androidEntry.setTag(entry.tag);
    androidEntry.setMessage(entry.message);

    pm.send("androidLog", msg);
  }

  android_logger_list_close(logger_list);
  return 0;
}
