#pragma once

#include <unordered_map>
#include <vector>

#include <QNetworkAccessManager>
#include <QString>
#include <QThread>

#include <capnp/serialize.h>

#include "cereal/gen/cpp/log.capnp.h"

#include "selfdrive/camerad/cameras/camera_common.h"

class FileReader : public QObject {
  Q_OBJECT

public:
  FileReader(const QString &fn, QObject *parent = nullptr);
  void read();
  void abort();

signals:
  void finished(const QByteArray &dat);
  void failed(const QString &err);

private:
  void startHttpRequest();
  QNetworkReply *reply_ = nullptr;
  QUrl url_;
};

const CameraType ALL_CAMERAS[] = {RoadCam, DriverCam, WideRoadCam};
const int MAX_CAMERAS = std::size(ALL_CAMERAS);

struct EncodeIdx {
  int segmentNum;
  uint32_t frameEncodeId;
};

class Event {
public:
  Event(cereal::Event::Which which, uint64_t mono_time) : reader(kj::ArrayPtr<capnp::word>{}) {
    // construct a dummy Event for binary search, e.g std::upper_bound
    this->which = which;
    this->mono_time = mono_time;
  }

  Event(const kj::ArrayPtr<const capnp::word> &amsg) : reader(amsg) {
    words = kj::ArrayPtr<const capnp::word>(amsg.begin(), reader.getEnd());
    event = reader.getRoot<cereal::Event>();
    which = event.which();
    mono_time = event.getLogMonoTime();
  }
  inline kj::ArrayPtr<const capnp::byte> bytes() const { return words.asBytes(); }

  struct lessThan {
    inline bool operator()(const Event *l, const Event *r) {
      return l->mono_time < r->mono_time || (l->mono_time == r->mono_time && l->which < r->which);
    }
  };

  uint64_t mono_time;
  cereal::Event::Which which;
  cereal::Event::Reader event;
  capnp::FlatArrayMessageReader reader;
  kj::ArrayPtr<const capnp::word> words;
};

class LogReader : public QObject {
  Q_OBJECT

public:
  LogReader(const QString &file, QObject *parent = nullptr);
  ~LogReader();
  inline bool valid() const { return valid_; }

  std::vector<Event*> events;
  std::unordered_map<uint32_t, EncodeIdx> eidx[MAX_CAMERAS] = {};

signals:
  void finished(bool success);

private:
  void parseEvents(const QByteArray &dat);

  std::atomic<bool> exit_ = false;
  std::atomic<bool> valid_ = false;
  std::vector<uint8_t> raw_;

  FileReader *file_reader_ = nullptr;
  QThread thread_;
};
