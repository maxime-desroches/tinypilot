#pragma once

#include <vector>
#include <unordered_map>

#include <QElapsedTimer>
#include <QMultiMap>
#include <QNetworkAccessManager>
#include <QString>
#include <QThread>

#include <capnp/serialize.h>

class FileReader : public QObject {
  Q_OBJECT

public:
  void startRequest(const QUrl &url);
  
signals:
  void ready(const QByteArray &dat);

protected:
  void readyRead();
  void httpFinished();
  QNetworkReply *reply;

private:
  QNetworkAccessManager *qnam;
  QElapsedTimer timer;
};

struct EncodeIdx {
  int segmentNum;
  uint32_t segmentId;
};

enum FrameType {
  RoadCamFrame = 0,
  DriverCamFrame,
  WideRoadCamFrame
};

typedef QMultiMap<uint64_t, capnp::FlatArrayMessageReader *> Events;
typedef std::unordered_map<int, EncodeIdx> EncodeIdxMap;

class LogReader : public QThread {
  Q_OBJECT

public:
  LogReader(const QString &file, QObject *parent);
  ~LogReader();
  void run() override;
  bool ready() const { return ready_; }
  const Events &events() const { return events_; }
  const EncodeIdx *getFrameEncodeIdx(FrameType type, uint32_t frame_id) const;

signals:
  void done();

protected:
  void readyRead(const QByteArray &dat);
  void parseEvents(kj::ArrayPtr<const capnp::word> amsg);

  std::vector<uint8_t> raw_;
  Events events_;
  std::atomic<bool> ready_ = false;
  EncodeIdxMap encoderIdx_[WideRoadCamFrame + 1];
  QString file;
  std::atomic<bool> exit_;
};
