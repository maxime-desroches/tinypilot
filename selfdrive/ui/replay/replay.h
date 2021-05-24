#pragma once

#include <iostream>
#include <termios.h>
#include <set>

#include <QJsonArray>
#include <QReadWriteLock>
#include <QThread>

#include <capnp/dynamic.h>

#include "cereal/visionipc/visionipc_server.h"
#include "selfdrive/common/util.h"
#include "selfdrive/ui/qt/api.h"
#include "selfdrive/ui/replay/filereader.h"
#include "selfdrive/ui/replay/framereader.h"


constexpr int FORWARD_SEGS = 2;
constexpr int BACKWARD_SEGS = 2;


class Replay : public QObject {
  Q_OBJECT

public:
  Replay(QString route, SubMaster *sm = nullptr, QObject *parent = 0);
  ~Replay();
  void start();
  void addSegment(int n);
  void removeSegment(int n);
  void seekTime(int ts);

public slots:
  void stream();
  void keyboardThread();
  void segmentQueueThread();
  void parseResponse(const QString &response);

private:
  float last_print = 0;
  uint64_t route_start_ts;
  std::atomic<int> seek_ts = 0;
  std::atomic<int> current_ts = 0;
  std::atomic<int> current_segment = 0;

  QThread *thread;
  QThread *kb_thread;
  QThread *queue_thread;

  HttpRequest *http;
  QJsonArray camera_paths;
  QJsonArray log_paths;
  QMap<int, LogReader*> lrs;
  QMap<int, FrameReader*> frs;

  // messaging
  SubMaster *sm;
  PubMaster *pm;
  std::set<std::string> socks;
  VisionIpcServer *vipc_server = nullptr;
};
