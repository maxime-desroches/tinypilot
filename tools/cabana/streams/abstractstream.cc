#include "tools/cabana/streams/abstractstream.h"

#include <utility>

#include <QDebug>
#include <QTimer>

static const int EVENT_NEXT_BUFFER_SIZE = 6 * 1024 * 1024;  // 6MB

AbstractStream *can = nullptr;

StreamNotifier *StreamNotifier::instance() {
  static StreamNotifier notifier;
  return &notifier;
}

AbstractStream::AbstractStream(QObject *parent) : QObject(parent) {
  assert(parent != nullptr);
  event_buffer_ = std::make_unique<MonotonicBuffer>(EVENT_NEXT_BUFFER_SIZE);

  QObject::connect(this, &AbstractStream::lastMsgsChanged, this, &AbstractStream::updateLastMessages, Qt::QueuedConnection);
  QObject::connect(this, &AbstractStream::seekedTo, this, &AbstractStream::updateLastMsgsTo);
  QObject::connect(dbc(), &DBCManager::changed, this, &AbstractStream::updateMasks);
  QObject::connect(this, &AbstractStream::streamStarted, [this]() {
    emit StreamNotifier::instance()->changingStream();
    delete can;
    can = this;
    emit StreamNotifier::instance()->streamStarted();
  });
}

void AbstractStream::updateMasks() {
  std::lock_guard lk(mutex_);
  masks_.clear();
  if (!settings.suppress_defined_signals) return;

  for (auto s : sources_) {
    for (const auto &[address, m] : dbc()->getMessages(s)) {
      masks_[{.source = (uint8_t)s, .address = address}] = m.mask;
    }
  }
  // clear bit change counts
  for (auto &[id, m] : msgs_) {
    auto &mask = masks_[id];
    const int size = std::min(mask.size(), m.last_changes.size());
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < 8; ++j) {
        if (((mask[i] >> (7 - j)) & 1) != 0) m.last_changes[i].bit_change_counts[j] = 0;
      }
    }
  }
}

void AbstractStream::suppressDefinedSignals(bool suppress) {
  std::lock_guard lk(mutex_);
  settings.suppress_defined_signals = suppress;
  updateMasks();
}

size_t AbstractStream::suppressHighlighted() {
  std::lock_guard lk(mutex_);
  size_t cnt = 0;
  const uint64_t cur_ts = currentMonoTime();
  for (auto &[_, m] : msgs_) {
    for (auto &last_change : m.last_changes) {
      const double dt = int64_t(cur_ts - last_change.mono_time) / 1e9;
      if (dt < 2.0) {
        last_change.suppressed = true;
      }
      // clear bit change counts
      last_change.bit_change_counts.fill(0);
      cnt += last_change.suppressed;
    }
  }
  return cnt;
}

void AbstractStream::clearSuppressed() {
  std::lock_guard lk(mutex_);
  for (auto &[_, m] : msgs_) {
    std::for_each(m.last_changes.begin(), m.last_changes.end(), [](auto &c) { c.suppressed = false; });
  }
}

void AbstractStream::updateLastMessages() {
  auto prev_src_size = sources_.size();
  auto prev_msg_size = last_msgs_.size();
  std::set<MessageId> messages;
  {
    std::lock_guard lk(mutex_);
    for (const auto &id : new_msgs_) {
      last_msgs_[id] = msgs_[id];
      sources_.insert(id.source);
    }
    messages = std::move(new_msgs_);
  }

  if (!messages.empty()) {
    if (sources_.size() != prev_src_size) {
      updateMasks();
      emit sourcesUpdated(sources_);
    }
    emit msgsReceived(&messages, prev_msg_size != last_msgs_.size());
  }
}

void AbstractStream::updateEvent(const MessageId &id, uint64_t ts, const uint8_t *data, uint8_t size) {
  std::lock_guard lk(mutex_);
  msgs_[id].update(id, data, size, ts, getSpeed(), masks_[id]);
  new_msgs_.insert(id);
}

const std::vector<const CanEvent *> &AbstractStream::events(const MessageId &id) const {
  static std::vector<const CanEvent *> empty_events;
  auto it = events_.find(id);
  return it != events_.end() ? it->second : empty_events;
}

const CanData &AbstractStream::lastMessage(const MessageId &id) {
  static CanData empty_data = {};
  auto it = last_msgs_.find(id);
  return it != last_msgs_.end() ? it->second : empty_data;
}

// it is thread safe to update data in updateLastMsgsTo.
// updateLastMsgsTo is always called in UI thread.
void AbstractStream::updateLastMsgsTo(double sec) {
  new_msgs_.clear();
  msgs_.clear();

  uint64_t last_ts = toMonoTime(sec);
  for (auto &[id, ev] : events_) {
    auto it = std::lower_bound(ev.crbegin(), ev.crend(), last_ts, [](auto e, uint64_t ts) {
      return e->mono_time > ts;
    });
    if (it != ev.crend()) {
      auto &m = msgs_[id];
      m.update(id, (*it)->dat, (*it)->size, (*it)->mono_time, getSpeed(), {});
      m.count = std::distance(it, ev.crend());
    }
  }

  bool changed = msgs_.size() != last_msgs_.size() ||
                 std::any_of(msgs_.begin(), msgs_.end(), [this](auto &m) { return !last_msgs_.count(m.first); });
  last_msgs_ = msgs_;
  // use a timer to prevent recursive calls
  QTimer::singleShot(0, this, [this, changed]() { emit msgsReceived(nullptr, changed); });
}

const CanEvent *AbstractStream::newEvent(uint64_t mono_time, const cereal::CanData::Reader &c) {
  auto dat = c.getDat();
  CanEvent *e = (CanEvent *)event_buffer_->allocate(sizeof(CanEvent) + sizeof(uint8_t) * dat.size());
  e->src = c.getSrc();
  e->address = c.getAddress();
  e->mono_time = mono_time;
  e->size = dat.size();
  memcpy(e->dat, (uint8_t *)dat.begin(), e->size);
  return e;
}

void AbstractStream::mergeEvents(const std::vector<const CanEvent *> &events) {
  static CanEventsMap events_map;
  std::for_each(events_map.begin(), events_map.end(), [](auto &e) { e.second.clear(); });
  for (auto e : events) {
    events_map[{.source = e->src, .address = e->address}].push_back(e);
  }

  if (!events.empty()) {
    for (auto &[id, new_e] : events_map) {
      if (!new_e.empty()) {
        auto &e = events_[id];
        auto insert_pos = std::upper_bound(e.begin(), e.end(), new_e.front()->mono_time, CompareCanEvent());
        e.insert(insert_pos, new_e.begin(), new_e.end());
      }
    }
    auto insert_pos = std::upper_bound(all_events_.begin(), all_events_.end(), events.front()->mono_time, CompareCanEvent());
    all_events_.insert(insert_pos, events.begin(), events.end());

    emit eventsMerged(events_map);
  }
}

// CanData

namespace {

constexpr int periodic_threshold = 10;
constexpr int start_alpha = 128;
constexpr float fade_time = 2.0;
const QColor CYAN = QColor(0, 187, 255, start_alpha);
const QColor RED = QColor(255, 0, 0, start_alpha);
const QColor GREYISH_BLUE = QColor(102, 86, 169, start_alpha / 2);
const QColor CYAN_LIGHTER = QColor(0, 187, 255, start_alpha).lighter(135);
const QColor RED_LIGHTER = QColor(255, 0, 0, start_alpha).lighter(135);
const QColor GREYISH_BLUE_LIGHTER = QColor(102, 86, 169, start_alpha / 2).lighter(135);

inline QColor blend(const QColor &a, const QColor &b) {
  return QColor((a.red() + b.red()) / 2, (a.green() + b.green()) / 2, (a.blue() + b.blue()) / 2, (a.alpha() + b.alpha()) / 2);
}

// Calculate the frequency of the past minute.
double calc_freq(const MessageId &msg_id, uint64_t current_ts) {
  const auto &events = can->events(msg_id);
  uint64_t first_ts = current_ts - std::min<int64_t>(current_ts, 59 * 1e9);
  auto first = std::lower_bound(events.begin(), events.end(), first_ts, CompareCanEvent());
  auto second = std::lower_bound(first, events.end(), current_ts, CompareCanEvent());
  if (first != events.end() && second != events.end()) {
    double duration = ((*second)->mono_time - (*first)->mono_time) / 1e9;
    uint32_t count = std::distance(first, second);
    return count / std::max(1.0, duration);
  }
  return 0;
}

}  // namespace

void CanData::update(const MessageId &msg_id, const uint8_t *can_data, const int size, uint64_t current_ts,
                      double playback_speed, const std::vector<uint8_t> &mask) {
  mono_time = current_ts;
  ++count;

  if (current_ts < last_freq_update_ts || (current_ts - last_freq_update_ts) >= 1.0) {
    last_freq_update_ts = current_ts;
    freq = calc_freq(msg_id, current_ts);
  }

  if (dat.size() != size) {
    dat.resize(size);
    colors.assign(size, QColor(0, 0, 0, 0));
    last_changes.resize(size);
    std::for_each(last_changes.begin(), last_changes.end(), [current_ts](auto &c) { c.mono_time = current_ts; });
  } else {
    bool lighter = settings.theme == DARK_THEME;
    const QColor &cyan = !lighter ? CYAN : CYAN_LIGHTER;
    const QColor &red = !lighter ? RED : RED_LIGHTER;
    const QColor &greyish_blue = !lighter ? GREYISH_BLUE : GREYISH_BLUE_LIGHTER;
    const float alpha_delta = 1.0 / (freq + 1) / (fade_time * playback_speed);

    for (int i = 0; i < size; ++i) {
      auto &last_change = last_changes[i];

      uint8_t mask_byte = last_change.suppressed ? 0x00 : 0xFF;
      if (i < mask.size()) mask_byte &= ~(mask[i]);

      const uint8_t last = dat[i] & mask_byte;
      const uint8_t cur = can_data[i] & mask_byte;
      const int delta = cur - last;

      if (last != cur) {
        double delta_t = (mono_time - last_change.mono_time) / 1e9;

        // Keep track if signal is changing randomly, or mostly moving in the same direction
        if (std::signbit(delta) == std::signbit(last_change.delta)) {
          last_change.same_delta_counter = std::min(16, last_change.same_delta_counter + 1);
        } else {
          last_change.same_delta_counter = std::max(0, last_change.same_delta_counter - 4);
        }

        // Mostly moves in the same direction, color based on delta up/down
        if (delta_t * freq > periodic_threshold || last_change.same_delta_counter > 8) {
          // Last change was while ago, choose color based on delta up or down
          colors[i] = (cur > last) ? cyan : red;
        } else {
          // Periodic changes
          colors[i] = blend(colors[i], greyish_blue);
        }

        // Track bit level changes
        const uint8_t tmp = (cur ^ last);
        for (int bit = 0; bit < 8; bit++) {
          if (tmp & (1 << (7 - bit))) {
            last_change.bit_change_counts[bit] += 1;
          }
        }

        last_change.mono_time = mono_time;
        last_change.delta = delta;
      } else {
        // Fade out
        colors[i].setAlphaF(std::max(0.0, colors[i].alphaF() - alpha_delta));
      }
    }
  }
  memcpy(dat.data(), can_data, size);
}
