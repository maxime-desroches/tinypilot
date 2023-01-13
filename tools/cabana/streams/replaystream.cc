#include "tools/cabana/streams/replaystream.h"

#include "tools/cabana/dbcmanager.h"

ReplayStream::ReplayStream(QObject *parent) : AbstractStream(parent, false) {
  QObject::connect(&settings, &Settings::changed, this, &ReplayStream::settingChanged);
}

ReplayStream::~ReplayStream() {
  replay->stop();
}

static bool event_filter(const Event *e, void *opaque) {
  ReplayStream *c = (ReplayStream *)opaque;
  return c->eventFilter(e);
}

bool ReplayStream::loadRoute(const QString &route, const QString &data_dir, uint32_t replay_flags) {
  replay = new Replay(route, {"can", "roadEncodeIdx", "wideRoadEncodeIdx", "carParams"}, {}, nullptr, replay_flags, data_dir, this);
  replay->setSegmentCacheLimit(settings.cached_segment_limit);
  replay->installEventFilter(event_filter, this);
  QObject::connect(replay, &Replay::seekedTo, this, &AbstractStream::seekedTo);
  QObject::connect(replay, &Replay::segmentsMerged, this, &AbstractStream::eventsMerged);
  QObject::connect(replay, &Replay::streamStarted, this, &AbstractStream::streamStarted);
  if (replay->load()) {
    const auto &segments = replay->route()->segments();
    if (std::none_of(segments.begin(), segments.end(), [](auto &s) { return s.second.rlog.length() > 0; })) {
      qWarning() << "no rlogs in route" << route;
      return false;
    }
    replay->start();
    return true;
  }
  return false;
}

bool ReplayStream::eventFilter(const Event *event) {
  if (event->which == cereal::Event::Which::CAN) {
    updateEvent(event);
  }
  return true;
}

void ReplayStream::seekTo(double ts) {
  replay->seekTo(std::max(double(0), ts), false);
  counters_begin_sec = 0;
  emit updated();
}

void ReplayStream::pause(bool pause) {
  replay->pause(pause);
  emit(pause ? paused() : resume());
}

void ReplayStream::settingChanged() {
  replay->setSegmentCacheLimit(settings.cached_segment_limit);
}
