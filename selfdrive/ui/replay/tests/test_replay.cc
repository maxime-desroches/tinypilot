#include <QCryptographicHash>
#include <QDebug>
#include <QEventLoop>
#include <QString>
#include <future>

#include "catch2/catch.hpp"
#include "selfdrive/common/util.h"
#include "selfdrive/ui/replay/replay.h"

const char *stream_url = "https://commadataci.blob.core.windows.net/openpilotci/0c94aa1e1296d7c6/2021-05-05--19-48-37/0/fcamera.hevc";

// TEST_CASE("FrameReader") {
//   SECTION("process&get") {
//     FrameReader fr;
//     REQUIRE(fr.load(stream_url) == true);
//     REQUIRE(fr.valid() == true);
//     REQUIRE(fr.getFrameCount() == 1200);
//     // random get 50 frames
//     // srand(time(NULL));
//     // for (int i = 0; i < 50; ++i) {
//     //   int idx = rand() % (fr.getFrameCount() - 1);
//     //   REQUIRE(fr.get(idx) != nullptr);
//     // }
//     // sequence get 50 frames {
//     for (int i = 0; i < 50; ++i) {
//       REQUIRE(fr.get(i) != nullptr);
//     }
//   }
// }

// std::string sha_256(const QString &dat) {
//   return QString(QCryptographicHash::hash(dat.toUtf8(), QCryptographicHash::Sha256).toHex()).toStdString();
// }

// TEST_CASE("httpMultiPartDownload") {
//   char filename[] = "/tmp/XXXXXX";
//   int fd = mkstemp(filename);
//   REQUIRE(fd != -1);
//   close(fd);

//   SECTION("http 200") {
//     REQUIRE(httpMultiPartDownload(stream_url, filename, 5));
//     std::string content = util::read_file(filename);
//     REQUIRE(content.size() == 37495242);
//     std::string checksum = sha_256(QString::fromStdString(content));
//     REQUIRE(checksum == "d8ff81560ce7ed6f16d5fb5a6d6dd13aba06c8080c62cfe768327914318744c4");
//   }
//   SECTION("http 404") {
//     REQUIRE(httpMultiPartDownload(util::string_format("%s_abc", stream_url), filename, 5) == false);
//   }
// }

int random_int(int min, int max) {
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<std::mt19937::result_type> dist(min, max);
  return dist(rng);
}

bool is_events_ordered(const std::vector<Event *> &events) {
  REQUIRE(events.size() > 0);
  uint64_t prev_mono_time = 0;
  cereal::Event::Which prev_which = cereal::Event::INIT_DATA;
  for (auto event : events) {
    if (event->mono_time < prev_mono_time || (event->mono_time == prev_mono_time && event->which < prev_which)) {
      return false;
    }
    prev_mono_time = event->mono_time;
    prev_which = event->which;
  }
  return true;
}

const QString DEMO_ROUTE = "3533c53bb29502d1|2019-12-10--01-13-27";
// Route demo_route(DEMO_ROUTE);

// TEST_CASE("Segment") {
//   REQUIRE(demo_route.load());
//   REQUIRE(demo_route.size() == 121);

//   QEventLoop loop;
//   Segment segment(0, demo_route.at(0), false, false);
//   REQUIRE(segment.isValid() == true);
//   REQUIRE(segment.isLoaded() == false);
//   QObject::connect(&segment, &Segment::loadFinished, [&]() {
//     REQUIRE(segment.isLoaded() == true);
//     REQUIRE(segment.log != nullptr);
//     REQUIRE(segment.log->events.size() > 0);
//     REQUIRE(is_events_ordered(segment.log->events));
//     REQUIRE(segment.frames[RoadCam] != nullptr);
//     REQUIRE(segment.frames[RoadCam]->getFrameCount() > 0);
//     REQUIRE(segment.frames[DriverCam] == nullptr);
//     REQUIRE(segment.frames[WideRoadCam] == nullptr);
//     loop.quit();
//   });
//   loop.exec();
// }

// helper class for unit tests
class TestReplay : public Replay {
 public:
  TestReplay(const QString &route) : Replay(route, {}, {}) {
  }
  void test_seek();

 protected:
  void testSeekTo(int seek_to, int invalid_segment = -1);
};

void TestReplay::testSeekTo(int seek_to, int invalid_segment) {
  seekTo(seek_to);

  // wait for seek finish
  std::unique_lock lk(lock_);
  stream_cv_.wait(lk, [=]() { return events_updated_ == true; });
  events_updated_ = false;

  // verify result
  INFO("seek to [" << seek_to << "s segment " << seek_to / 60 << "]");
  INFO("events size " << events_->size());
  // INFO("mono" << (cur_mono_time_ - route_start_ts_) / 1e9);
  REQUIRE(is_events_ordered(*events_));
  REQUIRE(uint64_t(route_start_ts_ + seek_to * 1e9) == cur_mono_time_);
  REQUIRE(!events_->empty());
  Event cur_event(cereal::Event::Which::INIT_DATA, cur_mono_time_);
  auto eit = std::upper_bound(events_->begin(), events_->end(), &cur_event, Event::lessThan());
  
  REQUIRE(eit != events_->end());
  const int seek_to_segment = seek_to / 60;
  const int real_segment = int(((*eit)->mono_time - route_start_ts_) / 60 / 1e9);
  INFO("event [" << ((*eit)->mono_time - route_start_ts_) / 1e9 << "s segment " << real_segment << "]");
  REQUIRE((*eit)->mono_time >= seek_to * 1e9 + route_start_ts_);
  if (seek_to_segment != invalid_segment) {
    REQUIRE(real_segment == seek_to_segment); // in the same segment
  } else {
    // skipped invalid_segment
    REQUIRE(real_segment == seek_to_segment + 1); 

  }
}

void TestReplay::test_seek() {
  QEventLoop loop;

  REQUIRE(load());
  // // limit the segment count to 5
  // REQUIRE(route_->size() >= 5);
  // segments_.resize(5);
  
  std::thread thread = std::thread([&]() {
    // random seek 200 times in first 3 good segments
    for (int i = 0; i < 200; ++i) {
      testSeekTo(random_int(0, 60 * 3 - 1));
    }

    // // make segment 1 invalid
    // segments_[1]->valid_ = segments_[1]->loaded_ = false;
    // queueSegment();
    // // random seek 200 times
    // for (int i = 0; i < 200; ++i) {
    //   testSeekTo(random_int(0, 60 * 3 - 1), 1);
    // }

    loop.quit();
  });

  loop.exec();
  thread.join();
}

TEST_CASE("Replay") {
  TestReplay replay(DEMO_ROUTE);
  REQUIRE(replay.load());
  // modify the route
  replay.test_seek();
}
