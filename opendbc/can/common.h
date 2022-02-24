#pragma once

#include <vector>
#include <map>
#include <unordered_map>

#include "common_dbc.h"
#include <capnp/dynamic.h>
#include <capnp/serialize.h>

#ifndef DYNAMIC_CAPNP
#include "cereal/gen/cpp/log.capnp.h"
#endif

#define MAX_BAD_COUNTER 5

// Helper functions
unsigned int honda_checksum(unsigned int address, uint64_t d, int l);
unsigned int toyota_checksum(unsigned int address, uint64_t d, int l);
unsigned int subaru_checksum(unsigned int address, uint64_t d, int l);
unsigned int chrysler_checksum(unsigned int address, uint64_t d, int l);
void init_crc_lookup_tables();
unsigned int volkswagen_crc(unsigned int address, uint64_t d, int l);
unsigned int pedal_checksum(uint64_t d, int l);
uint64_t read_u64_be(const uint8_t* v);
uint64_t read_u64_le(const uint8_t* v);

class MessageState {
public:
  uint32_t address;
  unsigned int size;

  std::vector<Signal> parse_sigs;
  std::vector<double> vals;
  std::vector<std::vector<double>> all_vals;

  uint64_t seen;
  uint64_t check_threshold;

  uint8_t counter;
  uint8_t counter_fail;

  bool ignore_checksum = false;
  bool ignore_counter = false;

  bool parse(uint64_t sec, uint8_t * dat);
  bool update_counter_generic(int64_t v, int cnt_size);
};

class CANParser {
private:
  const int bus;
  kj::Array<capnp::word> aligned_buf;

  const DBC *dbc = NULL;
  std::unordered_map<uint32_t, MessageState> message_states;

public:
  bool can_valid = false;
  uint64_t last_sec = 0;

  CANParser(int abus, const std::string& dbc_name,
            const std::vector<MessageParseOptions> &options,
            const std::vector<SignalParseOptions> &sigoptions);
  CANParser(int abus, const std::string& dbc_name, bool ignore_checksum, bool ignore_counter);
  #ifndef DYNAMIC_CAPNP
  void update_string(const std::string &data, bool sendcan);
  void UpdateCans(uint64_t sec, const capnp::List<cereal::CanData>::Reader& cans);
  #endif
  void UpdateCans(uint64_t sec, const capnp::DynamicStruct::Reader& cans);
  void UpdateValid(uint64_t sec);
  std::vector<SignalValue> query_latest();
};

class CANPacker {
private:
  const DBC *dbc = NULL;
  std::map<std::pair<uint32_t, std::string>, Signal> signal_lookup;
  std::map<uint32_t, Msg> message_lookup;

public:
  CANPacker(const std::string& dbc_name);
  uint64_t pack(uint32_t address, const std::vector<SignalPackValue> &values, int counter);
  Msg* lookup_message(uint32_t address);
};
