#pragma once

#include <atomic>
#include <cstdint>
#include <ctime>
#include <functional>
#include <list>
#include <mutex>
#include <optional>
#include <vector>

#include "cereal/gen/cpp/car.capnp.h"
#include "cereal/gen/cpp/log.capnp.h"
#include "panda/board/health.h"
#include "selfdrive/boardd/panda_comms.h"


class Panda {
private:
  PandaCommsHandle *handle = NULL;
  std::vector<uint8_t> recv_buf;

public:
  Panda(std::string serial="", uint32_t bus_offset=0);
  ~Panda();

  std::string hw_serial;
  std::atomic<bool> connected = true;
  std::atomic<bool> comms_healthy = true;
  cereal::PandaState::PandaType hw_type = cereal::PandaState::PandaType::UNKNOWN;
  bool has_rtc = false;
  const uint32_t bus_offset;

  // Static functions
  static std::vector<std::string> list();

  // Panda functionality
  cereal::PandaState::PandaType get_hw_type();
  void set_safety_model(cereal::CarParams::SafetyModel safety_model, uint16_t safety_param=0U);
  void set_alternative_experience(uint16_t alternative_experience);
  void set_rtc(struct tm sys_time);
  struct tm get_rtc();
  void set_fan_speed(uint16_t fan_speed);
  uint16_t get_fan_speed();
  void set_ir_pwr(uint16_t ir_pwr);
  std::optional<health_t> get_state();
  std::optional<can_health_t> get_can_state(uint16_t can_number);
  void set_loopback(bool loopback);
  std::optional<std::vector<uint8_t>> get_firmware_version();
  std::optional<std::string> get_serial();
  void set_power_saving(bool power_saving);
  void enable_deepsleep();
  void send_heartbeat(bool engaged);
  void set_can_speed_kbps(uint16_t bus, uint16_t speed);
  void set_data_speed_kbps(uint16_t bus, uint16_t speed);
  void set_canfd_non_iso(uint16_t bus, bool non_iso);
  void can_send(capnp::List<cereal::CanData>::Reader can_data_list);
  bool can_receive(std::vector<can_frame>& out_vec);

protected:
  // for unit tests
  Panda(uint32_t bus_offset) : bus_offset(bus_offset) {}
  void pack_can_buffer(const capnp::List<cereal::CanData>::Reader &can_data_list,
                         std::function<void(uint8_t *, size_t)> write_func);
  bool unpack_can_buffer(uint8_t *data, int size, std::vector<can_frame> &out_vec);
};
