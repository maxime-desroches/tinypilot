#include <zmq.h>

#include <iostream>

#include "catch2/catch.hpp"
#include "common/swaglog.h"
#include "common/util.h"
#include "common/version.h"
#include "system/hardware/hw.h"
#include "nlohmann/json.hpp"

std::string daemon_name = "testy";
std::string dongle_id = "test_dongle_id";
int LINE_NO = 0;

void log_thread(int thread_id, int msg_cnt) {
  for (int i = 0; i < msg_cnt; ++i) {
    LOGD("%d", thread_id);
    LINE_NO = __LINE__ - 1;
    usleep(1);
  }
}

void recv_log(int thread_cnt, int thread_msg_cnt) {
  void *zctx = zmq_ctx_new();
  void *sock = zmq_socket(zctx, ZMQ_PULL);
  zmq_bind(sock, Path::swaglog_ipc().c_str());
  std::vector<int> thread_msgs(thread_cnt);
  int total_count = 0;

  for (auto start = std::chrono::steady_clock::now(), now = start;
       now < start + std::chrono::seconds{1} && total_count < (thread_cnt * thread_msg_cnt);
       now = std::chrono::steady_clock::now()) {
    char buf[4096] = {};
    if (zmq_recv(sock, buf, sizeof(buf), ZMQ_DONTWAIT) <= 0) {
      if (errno == EAGAIN || errno == EINTR || errno == EFSM) continue;
      break;
    }

    REQUIRE(buf[0] == CLOUDLOG_DEBUG);
    std::string err;
    nlohmann::json msg = nlohmann::json::parse(buf + 1);

    REQUIRE(msg["levelnum"].template get<int>() == CLOUDLOG_DEBUG);
    REQUIRE_THAT(msg["filename"].template get<std::string>(), Catch::Contains("test_swaglog.cc"));
    REQUIRE(msg["funcname"].template get<std::string>() == "log_thread");
    REQUIRE(msg["lineno"].template get<int>() == LINE_NO);

    auto ctx = msg["ctx"];

    REQUIRE(ctx["daemon"].template get<std::string>() == daemon_name);
    REQUIRE(ctx["dongle_id"].template get<std::string>() == dongle_id);
    REQUIRE(ctx["dirty"].template get<bool>() == true);

    REQUIRE(ctx["version"].template get<std::string>() == COMMA_VERSION);

    std::string device = Hardware::get_name();
    REQUIRE(ctx["device"].template get<std::string>() == device);

    int thread_id = atoi(msg["msg"].template get<std::string>().c_str());
    REQUIRE((thread_id >= 0 && thread_id < thread_cnt));
    thread_msgs[thread_id]++;
    total_count++;
  }
  for (int i = 0; i < thread_cnt; ++i) {
    INFO("thread :" << i);
    REQUIRE(thread_msgs[i] == thread_msg_cnt);
  }
  zmq_close(sock);
  zmq_ctx_destroy(zctx);
}

TEST_CASE("swaglog") {
  setenv("MANAGER_DAEMON", daemon_name.c_str(), 1);
  setenv("DONGLE_ID", dongle_id.c_str(), 1);
  setenv("dirty", "1", 1);
  const int thread_cnt = 5;
  const int thread_msg_cnt = 100;

  std::vector<std::thread> log_threads;
  for (int i = 0; i < thread_cnt; ++i) {
    log_threads.push_back(std::thread(log_thread, i, thread_msg_cnt));
  }
  for (auto &t : log_threads) t.join();

  recv_log(thread_cnt, thread_msg_cnt);
}
