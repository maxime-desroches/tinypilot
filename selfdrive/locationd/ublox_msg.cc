#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <cmath>
#include <ctime>
#include <chrono>
#include <iostream>
#include <cassert>
#include <unordered_map>

#include "common/swaglog.h"

#include "ublox_msg.h"

#define UBLOX_MSG_SIZE(hdr) (*(uint16_t *)&hdr[4])

inline static bool bit_to_bool(uint8_t val, int shifts) {
  return (bool)(val & (1 << shifts));
}

inline int UbloxMsgParser::needed_bytes() {
  // Msg header incomplete?
  if(bytes_in_parse_buf < ublox::UBLOX_HEADER_SIZE)
    return ublox::UBLOX_HEADER_SIZE + ublox::UBLOX_CHECKSUM_SIZE - bytes_in_parse_buf;
  uint16_t needed = UBLOX_MSG_SIZE(msg_parse_buf) + ublox::UBLOX_HEADER_SIZE + ublox::UBLOX_CHECKSUM_SIZE;
  // too much data
  if(needed < (uint16_t)bytes_in_parse_buf)
    return -1;
  return needed - (uint16_t)bytes_in_parse_buf;
}

inline bool UbloxMsgParser::valid_cheksum() {
  uint8_t ck_a = 0, ck_b = 0;
  for(int i = 2; i < bytes_in_parse_buf - ublox::UBLOX_CHECKSUM_SIZE;i++) {
    ck_a = (ck_a + msg_parse_buf[i]) & 0xFF;
    ck_b = (ck_b + ck_a) & 0xFF;
  }
  if(ck_a != msg_parse_buf[bytes_in_parse_buf - 2]) {
    LOGD("Checksum a mismtach: %02X, %02X", ck_a, msg_parse_buf[6]);
    return false;
  }
  if(ck_b != msg_parse_buf[bytes_in_parse_buf - 1]) {
    LOGD("Checksum b mismtach: %02X, %02X", ck_b, msg_parse_buf[7]);
    return false;
  }
  return true;
}

inline bool UbloxMsgParser::valid() {
  return bytes_in_parse_buf >= ublox::UBLOX_HEADER_SIZE + ublox::UBLOX_CHECKSUM_SIZE &&
         needed_bytes() == 0 && valid_cheksum();
}

inline bool UbloxMsgParser::valid_so_far() {
  if(bytes_in_parse_buf > 0 && msg_parse_buf[0] != ublox::PREAMBLE1) {
    return false;
  }
  if(bytes_in_parse_buf > 1 && msg_parse_buf[1] != ublox::PREAMBLE2) {
    return false;
  }
  if(needed_bytes() == 0 && !valid()) {
    return false;
  }
  return true;
}


bool UbloxMsgParser::add_data(const uint8_t *incoming_data, uint32_t incoming_data_len, size_t &bytes_consumed) {
  int needed = needed_bytes();
  if(needed > 0) {
    bytes_consumed = std::min((uint32_t)needed, incoming_data_len );
    // Add data to buffer
    memcpy(msg_parse_buf + bytes_in_parse_buf, incoming_data, bytes_consumed);
    bytes_in_parse_buf += bytes_consumed;
  } else {
    bytes_consumed = incoming_data_len;
  }

  // Validate msg format, detect invalid header and invalid checksum.
  while(!valid_so_far() && bytes_in_parse_buf != 0) {
    //LOGD("Drop corrupt data, remained in buf: %u", bytes_in_parse_buf);
    // Corrupted msg, drop a byte.
    bytes_in_parse_buf -= 1;
    if(bytes_in_parse_buf > 0)
      memmove(&msg_parse_buf[0], &msg_parse_buf[1], bytes_in_parse_buf);
  }

  // There is redundant data at the end of buffer, reset the buffer.
  if(needed_bytes() == -1) {
    bytes_in_parse_buf = 0;
  }
  return valid();
}


std::pair<std::string, kj::Array<capnp::word>> UbloxMsgParser::gen_msg() {
  std::string dat = data();
  kaitai::kstream stream(dat);

  ubx_t ubx_message(&stream);
  auto body = ubx_message.body();

  switch (ubx_message.msg_type())
  {
  case 0x0107:
    return {"gpsLocationExternal", gen_nav_pvt(static_cast<ubx_t::nav_pvt_t*>(body))};
    break;
  case 0x0213:
    return {"ubloxGnss", gen_rxm_sfrbx(static_cast<ubx_t::rxm_sfrbx_t*>(body))};
    break;
  case 0x0215:
    return {"ubloxGnss", gen_rxm_rawx(static_cast<ubx_t::rxm_rawx_t*>(body))};
    break;
  case 0x0a09:
    return {"ubloxGnss", gen_mon_hw(static_cast<ubx_t::mon_hw_t*>(body))};
    break;
  case 0x0a0b:
    return {"ubloxGnss", gen_mon_hw2(static_cast<ubx_t::mon_hw2_t*>(body))};
    break;
  default:
    LOGE("Unkown message type %x", ubx_message.msg_type());
    return {"ubloxGnss", kj::Array<capnp::word>()};
    break;
  }
}


kj::Array<capnp::word> UbloxMsgParser::gen_nav_pvt(ubx_t::nav_pvt_t *msg) {
  MessageBuilder msg_builder;
  auto gpsLoc = msg_builder.initEvent().initGpsLocationExternal();
  gpsLoc.setSource(cereal::GpsLocationData::SensorSource::UBLOX);
  gpsLoc.setFlags(msg->flags());
  gpsLoc.setLatitude(msg->lat() * 1e-07);
  gpsLoc.setLongitude(msg->lon() * 1e-07);
  gpsLoc.setAltitude(msg->height() * 1e-03);
  gpsLoc.setSpeed(msg->g_speed() * 1e-03);
  gpsLoc.setBearingDeg(msg->head_mot() * 1e-5);
  gpsLoc.setAccuracy(msg->h_acc() * 1e-03);
  std::tm timeinfo = std::tm();
  timeinfo.tm_year = msg->year() - 1900;
  timeinfo.tm_mon = msg->month() - 1;
  timeinfo.tm_mday = msg->day();
  timeinfo.tm_hour = msg->hour();
  timeinfo.tm_min = msg->min();
  timeinfo.tm_sec = msg->sec();

  std::time_t utc_tt = timegm(&timeinfo);
  gpsLoc.setTimestamp(utc_tt * 1e+03 + msg->nano() * 1e-06);
  float f[] = { msg->vel_n() * 1e-03f, msg->vel_e() * 1e-03f, msg->vel_d() * 1e-03f };
  gpsLoc.setVNED(f);
  gpsLoc.setVerticalAccuracy(msg->v_acc() * 1e-03);
  gpsLoc.setSpeedAccuracy(msg->s_acc() * 1e-03);
  gpsLoc.setBearingAccuracyDeg(msg->head_acc() * 1e-05);
  return capnp::messageToFlatArray(msg_builder);

}


kj::Array<capnp::word> UbloxMsgParser::gen_rxm_sfrbx(ubx_t::rxm_sfrbx_t *msg) {
  auto body = *msg->body();

  if (msg->gnss_id() == ubx_t::gnss_type_t::GNSS_TYPE_GPS) {

    // GPS subframes are packed into 10x 4 bytes, each containing 3 actual bytes
    // We will first need to separate the data from the padding and parity
    assert(body.size() == 10);

    std::string subframe_data;
    subframe_data.reserve(30);
    for (uint32_t word : body) {
      word = word >> 6; // TODO: Verify parity
      subframe_data.push_back(word >> 16);
      subframe_data.push_back(word >> 8);
      subframe_data.push_back(word >> 0);
    }

    kaitai::kstream stream(subframe_data);
    gps_t subframe(&stream);
    int subframe_id = subframe.how()->subframe_id();

    switch (subframe_id) {
    case 1:
      break;
    case 2:
      break;
    case 3:
      break;
    case 4:
      break;
    case 5:
      break;
    default:
      LOGE("Unknow GPS subframe id %d", subframe_id);
      break;
    }
  }
  return kj::Array<capnp::word>();
}

kj::Array<capnp::word> UbloxMsgParser::gen_rxm_rawx(ubx_t::rxm_rawx_t *msg) {
  MessageBuilder msg_builder;
  auto mr = msg_builder.initEvent().initUbloxGnss().initMeasurementReport();
  mr.setRcvTow(msg->rcv_tow());
  mr.setGpsWeek(msg->week());
  mr.setLeapSeconds(msg->leap_s());
  mr.setGpsWeek(msg->week());

  auto mb = mr.initMeasurements(msg->num_meas());
  auto measurements = *msg->measurements();
  for(int8_t i = 0; i < msg->num_meas(); i++) {
    mb[i].setSvId(measurements[i]->sv_id());
    // mb[i].setSigId(measurements[i]->sig_id()); // reserved2
    mb[i].setPseudorange(measurements[i]->pr_mes());
    mb[i].setCarrierCycles(measurements[i]->cp_mes());
    mb[i].setDoppler(measurements[i]->do_mes());
    mb[i].setGnssId(measurements[i]->gnss_id());
    mb[i].setGlonassFrequencyIndex(measurements[i]->freq_id());
    mb[i].setLocktime(measurements[i]->lock_time());
    mb[i].setCno(measurements[i]->cno());
    mb[i].setPseudorangeStdev(0.01*(pow(2, (measurements[i]->pr_stdev() & 15)))); // weird scaling, might be wrong
    mb[i].setCarrierPhaseStdev(0.004*(measurements[i]->cp_stdev() & 15));
    mb[i].setDopplerStdev(0.002*(pow(2, (measurements[i]->do_stdev() & 15)))); // weird scaling, might be wrong

    auto ts = mb[i].initTrackingStatus();
    auto trk_stat = measurements[i]->trk_stat();
    ts.setPseudorangeValid(bit_to_bool(trk_stat, 0));
    ts.setCarrierPhaseValid(bit_to_bool(trk_stat, 1));
    ts.setHalfCycleValid(bit_to_bool(trk_stat, 2));
    ts.setHalfCycleSubtracted(bit_to_bool(trk_stat, 3));
  }

  mr.setNumMeas(msg->num_meas());
  auto rs = mr.initReceiverStatus();
  rs.setLeapSecValid(bit_to_bool(msg->rec_stat(), 0));
  rs.setClkReset(bit_to_bool(msg->rec_stat(), 2));
  return capnp::messageToFlatArray(msg_builder);
}

kj::Array<capnp::word> UbloxMsgParser::gen_mon_hw(ubx_t::mon_hw_t *msg) {
  MessageBuilder msg_builder;
  auto hwStatus = msg_builder.initEvent().initUbloxGnss().initHwStatus();
  hwStatus.setNoisePerMS(msg->noise_per_ms());
  hwStatus.setAgcCnt(msg->agc_cnt());
  hwStatus.setAStatus((cereal::UbloxGnss::HwStatus::AntennaSupervisorState) msg->a_status());
  hwStatus.setAPower((cereal::UbloxGnss::HwStatus::AntennaPowerStatus) msg->a_power());
  hwStatus.setJamInd(msg->jam_ind());
  return capnp::messageToFlatArray(msg_builder);
}

kj::Array<capnp::word> UbloxMsgParser::gen_mon_hw2(ubx_t::mon_hw2_t *msg) {
  MessageBuilder msg_builder;
  auto hwStatus = msg_builder.initEvent().initUbloxGnss().initHwStatus2();
  hwStatus.setOfsI(msg->ofs_i());
  hwStatus.setMagI(msg->mag_i());
  hwStatus.setOfsQ(msg->ofs_q());
  hwStatus.setMagQ(msg->mag_q());

  switch (msg->cfg_source()) {
    case ubx_t::mon_hw2_t::config_source_t::CONFIG_SOURCE_ROM:
      hwStatus.setCfgSource(cereal::UbloxGnss::HwStatus2::ConfigSource::ROM);
      break;
    case ubx_t::mon_hw2_t::config_source_t::CONFIG_SOURCE_OTP:
      hwStatus.setCfgSource(cereal::UbloxGnss::HwStatus2::ConfigSource::OTP);
      break;
    case ubx_t::mon_hw2_t::config_source_t::CONFIG_SOURCE_CONFIG_PINS:
      hwStatus.setCfgSource(cereal::UbloxGnss::HwStatus2::ConfigSource::CONFIGPINS);
      break;
    case ubx_t::mon_hw2_t::config_source_t::CONFIG_SOURCE_FLASH:
      hwStatus.setCfgSource(cereal::UbloxGnss::HwStatus2::ConfigSource::FLASH);
      break;
    default:
      hwStatus.setCfgSource(cereal::UbloxGnss::HwStatus2::ConfigSource::UNDEFINED);
      break;
  }

  hwStatus.setLowLevCfg(msg->low_lev_cfg());
  hwStatus.setPostStatus(msg->post_status());

  return capnp::messageToFlatArray(msg_builder);
}
