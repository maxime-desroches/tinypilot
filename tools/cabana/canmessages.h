#pragma once

#include <atomic>

#include <QColor>
#include <QHash>
#include <QApplication>

#include "opendbc/can/common_dbc.h"
#include "tools/cabana/settings.h"
#include "tools/cabana/util.h"
#include "tools/replay/replay.h"

struct CanData {
  double ts = 0.;
  uint8_t src = 0;
  uint32_t address = 0;
  uint32_t count = 0;
  uint32_t freq = 0;
  QByteArray dat;
  QVector<QColor> colors;
};

class CANMessages : public QObject {
  Q_OBJECT

public:
  CANMessages(QObject *parent);
  ~CANMessages();
  bool loadRoute(const QString &route, const QString &data_dir, uint32_t replay_flags = REPLAY_FLAG_NONE);
  void seekTo(double ts);
  bool eventFilter(const Event *event);

  inline QString routeName() const { return replay->route()->name(); }
  inline QString carFingerprint() const { return replay->carFingerprint().c_str(); }
  inline VisionStreamType visionStreamType() const { return replay->hasFlag(REPLAY_FLAG_ECAM) ? VISION_STREAM_WIDE_ROAD : VISION_STREAM_ROAD; }
  inline double totalSeconds() const { return replay->totalSeconds(); }
  inline double routeStartTime() const { return replay->routeStartTime() / (double)1e9; }
  inline double currentSec() const { return replay->currentSeconds(); }
  inline QDateTime currentDateTime() const { return replay->currentDateTime(); }
  inline const CanData &lastMessage(const QString &id) { return can_msgs[id]; }

  inline const Route* route() const { return replay->route(); }
  inline const std::vector<Event *> *events() const { return replay->events(); }
  inline void setSpeed(float speed) { replay->setSpeed(speed); }
  inline bool isPaused() const { return replay->isPaused(); }
  void pause(bool pause);
  inline const std::vector<std::tuple<int, int, TimelineType>> getTimeline() { return replay->getTimeline(); }

signals:
  void paused();
  void resume();
  void seekedTo(double sec);
  void streamStarted();
  void eventsMerged();
  void updated();
  void msgsReceived(const QHash<QString, CanData> *);
  void received(QHash<QString, CanData> *);

public:
  QMap<QString, CanData> can_msgs;

protected:
  void process(QHash<QString, CanData> *);
  void settingChanged();

  Replay *replay = nullptr;
  std::atomic<double> counters_begin_sec = 0;
  std::atomic<bool> processing = false;
  QHash<QString, uint32_t> counters;
};

// A global pointer referring to the unique CANMessages object
extern CANMessages *can;
