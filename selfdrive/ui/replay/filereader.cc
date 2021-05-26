#include "selfdrive/ui/replay/filereader.h"

#include <bzlib.h>
#include <QtNetwork>
#include "cereal/gen/cpp/log.capnp.h"

static bool decompressBZ2(std::vector<uint8_t> &dest, const char srcData[], size_t srcSize,
                          size_t outputSizeIncrement = 0x100000U) {
  bz_stream strm = {};
  int ret = BZ2_bzDecompressInit(&strm, 0, 0);
  assert(ret == BZ_OK);

  strm.next_in = const_cast<char *>(srcData);
  strm.avail_in = srcSize;
  do {
    strm.next_out = (char *)&dest[strm.total_out_lo32];
    strm.avail_out = dest.size() - strm.total_out_lo32;
    ret = BZ2_bzDecompress(&strm);
    if (ret == BZ_OK && strm.avail_in > 0 && strm.avail_out == 0) {
      dest.resize(dest.size() + outputSizeIncrement);
    }
  } while (ret == BZ_OK);

  BZ2_bzDecompressEnd(&strm);
  dest.resize(strm.total_out_lo32);
  return ret == BZ_STREAM_END;
}

void FileReader::startRequest(const QUrl &url) {
  timer.start();

  qnam = new QNetworkAccessManager;
  reply = qnam->get(QNetworkRequest(url));
  connect(reply, &QNetworkReply::finished, this, &FileReader::httpFinished);
  connect(reply, &QIODevice::readyRead, this, &FileReader::readyRead);
  qDebug() << "requesting" << url;
}

void FileReader::httpFinished() {
  if (reply->error()) {
    qWarning() << reply->errorString();
  }

  const QVariant redirectionTarget = reply->attribute(QNetworkRequest::RedirectionTargetAttribute);
  if (!redirectionTarget.isNull()) {
    const QUrl redirectedUrl = redirectionTarget.toUrl();
    //qDebug() << "redirected to" << redirectedUrl;
    startRequest(redirectedUrl);
  } else {
    qDebug() << "done in" << timer.elapsed() << "ms";
  }
}

void FileReader::readyRead() {
  QByteArray dat = reply->readAll();
  emit ready(dat);
}

LogReader::LogReader(const QString &file, QObject *parent) : file(file),  QThread(parent) {
  // start with 64MB buffer
  raw_.resize(1024 * 1024 * 64);
}

LogReader::~LogReader() {
  // wait thread exit
  exit_ = true;
  wait();

  for (auto e : events_) {
    delete e;
  }
}

void LogReader::run() {
  QEventLoop loop;
  FileReader reader;
  connect(&reader, &FileReader::ready, [&](const QByteArray &dat) {
    if (!decompressBZ2(raw_, dat.data(), dat.size())) {
      qWarning() << "bz2 decompress failed";
    }
    loop.exit();
  });
  reader.startRequest(file);
  loop.exec();

  parseEvents({(const capnp::word *)raw_.data(), raw_.size() / sizeof(capnp::word)});
}

void LogReader::parseEvents(kj::ArrayPtr<const capnp::word> amsg) {
  size_t offset = 0;
  while (!exit_ && offset < amsg.size()) {
    try {
      std::unique_ptr<capnp::FlatArrayMessageReader> reader =
          std::make_unique<capnp::FlatArrayMessageReader>(amsg.slice(offset, amsg.size()));

      cereal::Event::Reader event = reader->getRoot<cereal::Event>();
      offset = reader->getEnd() - amsg.begin();

      // hack
      // TODO: rewrite with callback
      if (event.which() == cereal::Event::ROAD_ENCODE_IDX) {
        auto ee = event.getRoadEncodeIdx();
        encoderIdx_[RoadCamFrame][ee.getFrameId()] = {ee.getSegmentNum(), ee.getSegmentId()};
      } else if (event.which() == cereal::Event::DRIVER_ENCODE_IDX) {
        auto ee = event.getDriverEncodeIdx();
        encoderIdx_[DriverCamFrame][ee.getFrameId()] = {ee.getSegmentNum(), ee.getSegmentId()};
      } else if (event.which() == cereal::Event::WIDE_ROAD_ENCODE_IDX) {
        auto ee = event.getWideRoadEncodeIdx();
        encoderIdx_[WideRoadCamFrame][ee.getFrameId()] = {ee.getSegmentNum(), ee.getSegmentId()};
      }

      events_.insert(event.getLogMonoTime(), reader.release());
    } catch (const kj::Exception &e) {
      // partial messages trigger this
      // qDebug() << e.getDescription().cStr();
      break;
    }
  }
  ready_ = true;
  emit done();
}

void LogReader::readyRead(const QByteArray &dat) {
  if (!decompressBZ2(raw_, dat.data(), dat.size())) {
    qWarning() << "bz2 decompress failed";
  }
  parseEvents({(const capnp::word *)raw_.data(), raw_.size() / sizeof(capnp::word)});
}

const EncodeIdx *LogReader::getFrameEncodeIdx(FrameType type, uint32_t frame_id) const {
  auto it = encoderIdx_[type].find(frame_id);
  return it != encoderIdx_[type].end() ? &(it->second) : nullptr;
}
