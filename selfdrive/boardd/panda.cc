#include "selfdrive/boardd/panda.h"

#include <unistd.h>

#include <cassert>
#include <stdexcept>
#include <vector>

#include <sys/ioctl.h>
#include <linux/spi/spidev.h>

#include "cereal/messaging/messaging.h"
#include "panda/board/dlc_to_len.h"
#include "common/gpio.h"
#include "common/swaglog.h"
#include "common/util.h"

static int init_usb_ctx(libusb_context **context) {
  assert(context != nullptr);

  int err = libusb_init(context);
  if (err != 0) {
    LOGE("libusb initialization error");
    return err;
  }

#if LIBUSB_API_VERSION >= 0x01000106
  libusb_set_option(*context, LIBUSB_OPTION_LOG_LEVEL, LIBUSB_LOG_LEVEL_INFO);
#else
  libusb_set_debug(*context, 3);
#endif

  return err;
}


Panda::Panda(std::string serial, uint32_t bus_offset) : bus_offset(bus_offset) {
  int ret, err;
  libusb_device **dev_list = NULL;

  if (serial.find("spidev") != std::string::npos) {
    spi_comms = true;
    hw_serial = serial;

    LOGD("opening SPI panda: %s", serial.c_str());
    spi_fd = open(serial.c_str(), O_RDWR);

    // SPI settings
    uint32_t spi_mode = SPI_MODE_0;
    err = ioctl(spi_fd, SPI_IOC_WR_MODE, &spi_mode);
    if (err < 0) { LOGE("failed setting SPI mode"); goto fail; }

    uint32_t spi_speed = 30000000;
    err = ioctl(spi_fd, SPI_IOC_WR_MAX_SPEED_HZ, &spi_speed);
    if (err < 0) { LOGE("failed setting SPI speed"); goto fail; }

    uint8_t spi_bits_per_word = 8;
    err = ioctl(spi_fd, SPI_IOC_WR_BITS_PER_WORD, &spi_bits_per_word);
    if (err < 0) { LOGE("failed setting SPI bits per word"); goto fail; }
   
  } else {
    spi_comms = false;

    // init libusb
    ssize_t num_devices;
    err = init_usb_ctx(&ctx);
    if (err != 0) { goto fail; }

    // connect by serial
    num_devices = libusb_get_device_list(ctx, &dev_list);
    if (num_devices < 0) { goto fail; }
    for (size_t i = 0; i < num_devices; ++i) {
      libusb_device_descriptor desc;
      libusb_get_device_descriptor(dev_list[i], &desc);
      if (desc.idVendor == 0xbbaa && desc.idProduct == 0xddcc) {
        ret = libusb_open(dev_list[i], &dev_handle);
        if (dev_handle == NULL || ret < 0) { goto fail; }

        unsigned char desc_serial[26] = { 0 };
        ret = libusb_get_string_descriptor_ascii(dev_handle, desc.iSerialNumber, desc_serial, std::size(desc_serial));
        if (ret < 0) { goto fail; }

        hw_serial = std::string((char *)desc_serial, ret).c_str();
        if (serial.empty() || serial == hw_serial) {
          break;
        }
        libusb_close(dev_handle);
        dev_handle = NULL;
      }
    }
    if (dev_handle == NULL) goto fail;
    libusb_free_device_list(dev_list, 1);
    dev_list = nullptr;

    if (libusb_kernel_driver_active(dev_handle, 0) == 1) {
      libusb_detach_kernel_driver(dev_handle, 0);
    }

    err = libusb_set_configuration(dev_handle, 1);
    if (err != 0) { goto fail; }

    err = libusb_claim_interface(dev_handle, 0);
    if (err != 0) { goto fail; }
  }

  hw_type = get_hw_type();

  assert((hw_type != cereal::PandaState::PandaType::WHITE_PANDA) &&
         (hw_type != cereal::PandaState::PandaType::GREY_PANDA));

  has_rtc = (hw_type == cereal::PandaState::PandaType::UNO) ||
            (hw_type == cereal::PandaState::PandaType::DOS);

  return;

fail:
  if (dev_list != NULL) {
    libusb_free_device_list(dev_list, 1);
  }
  cleanup();
  throw std::runtime_error("Error connecting to panda");
}

Panda::~Panda() {
  std::lock_guard lk(hw_lock);
  cleanup();
  connected = false;
}

void Panda::cleanup() {
  if (spi_fd != -1) {
    close(spi_fd);
    spi_fd = -1;
  }

  if (dev_handle) {
    libusb_release_interface(dev_handle, 0);
    libusb_close(dev_handle);
  }

  if (ctx) {
    libusb_exit(ctx);
  }
}

std::vector<std::string> Panda::list() {
  // init libusb
  ssize_t num_devices;
  libusb_context *context = NULL;
  libusb_device **dev_list = NULL;
  std::vector<std::string> serials;

  int err = init_usb_ctx(&context);
  if (err != 0) { return serials; }

  num_devices = libusb_get_device_list(context, &dev_list);
  if (num_devices < 0) {
    LOGE("libusb can't get device list");
    goto finish;
  }
  for (size_t i = 0; i < num_devices; ++i) {
    libusb_device *device = dev_list[i];
    libusb_device_descriptor desc;
    libusb_get_device_descriptor(device, &desc);
    if (desc.idVendor == 0xbbaa && desc.idProduct == 0xddcc) {
      libusb_device_handle *handle = NULL;
      int ret = libusb_open(device, &handle);
      if (ret < 0) { goto finish; }

      unsigned char desc_serial[26] = { 0 };
      ret = libusb_get_string_descriptor_ascii(handle, desc.iSerialNumber, desc_serial, std::size(desc_serial));
      libusb_close(handle);
      if (ret < 0) { goto finish; }

      serials.push_back(std::string((char *)desc_serial, ret).c_str());
    }
  }

finish:
  if (dev_list != NULL) {
    libusb_free_device_list(dev_list, 1);
  }
  if (context) {
    libusb_exit(context);
  }
  return serials;
}

void add_checksum(uint8_t *data, int data_len) {
  data[data_len] = SPI_CHECKSUM_START;
  for (int i=0; i < data_len; i++) {
    data[data_len] ^= data[i];
  }
}

int Panda::spi_transfer(uint8_t endpoint, uint8_t *tx_data, uint16_t tx_len, uint8_t *rx_data, uint16_t max_rx_len) {
  int ret = 0, rx_data_len, i = 0;
  uint8_t tx_buf[SPI_BUF_SIZE], rx_buf[SPI_BUF_SIZE];

  // needs to be less, since we need to have space for the checksum
  assert(tx_len < SPI_BUF_SIZE);
  assert(max_rx_len < SPI_BUF_SIZE);

  spi_header header = {
    .sync = SPI_SYNC,
    .endpoint = endpoint,
    .tx_len = tx_len,
    .max_rx_len = max_rx_len
  };

  spi_ioc_transfer transfer = {
    .tx_buf = (uint64_t) tx_buf,
    .rx_buf = (uint64_t) rx_buf
  };

  // Send header
  memcpy(tx_buf, &header, sizeof(header));
  add_checksum(tx_buf, sizeof(header));
  transfer.len = sizeof(header) + 1;
  ret = ioctl(spi_fd, SPI_IOC_MESSAGE(1), &transfer);
  if (ret < 0) { LOGE("SPI: failed to send header"); goto transfer_fail; }

  // Wait for (N)ACK
  tx_buf[0] = 0x12;
  transfer.len = 1;
  while (true) {
    ret = ioctl(spi_fd, SPI_IOC_MESSAGE(1), &transfer);
    if (ret < 0) { LOGE("SPI: failed to send ACK request"); goto transfer_fail; }

    if (rx_buf[0] == SPI_HACK) {
      break;
    } else if (rx_buf[0] == SPI_NACK) {
      LOGW("SPI: got header NACK");
      goto transfer_fail;
    }
  }

  // Send data
  if (tx_data != NULL) {
    memcpy(tx_buf, tx_data, tx_len);
  }
  add_checksum(tx_buf, tx_len);
  transfer.len = tx_len + 1;
  ret = ioctl(spi_fd, SPI_IOC_MESSAGE(1), &transfer);
  if (ret < 0) { LOGE("SPI: failed to send data"); goto transfer_fail; }

  // Wait for (N)ACK
  tx_buf[0] = 0xab;
  transfer.len = 1;
  while (true) {
    ret = ioctl(spi_fd, SPI_IOC_MESSAGE(1), &transfer);
    if (ret < 0) { LOGE("SPI: failed to send ACK request"); goto transfer_fail; }

    i++;

    if (rx_buf[0] == SPI_DACK) {
      break;
    } else if (rx_buf[0] == SPI_NACK) {
      LOGW("SPI: got data NACK");
      goto transfer_fail;
    }
  }

  // Read data len
  transfer.len = 2;
  ret = ioctl(spi_fd, SPI_IOC_MESSAGE(1), &transfer);
  if (ret < 0) { LOGE("SPI: failed to read rx data len"); goto transfer_fail; }
  rx_data_len = *(uint16_t *)rx_buf;
  assert(rx_data_len < SPI_BUF_SIZE);

  // Read data
  transfer.len = rx_data_len + 1;
  ret = ioctl(spi_fd, SPI_IOC_MESSAGE(1), &transfer);
  if (ret < 0) { LOGE("SPI: failed to read rx data"); goto transfer_fail; }
  // TODO: check checksum

  if (rx_data != NULL) {
    memcpy(rx_data, rx_buf, rx_data_len);
  }
  ret = rx_data_len;

transfer_fail:
  return ret;
}

void Panda::handle_usb_issue(int err, const char func[]) {
  LOGE_100("usb error %d \"%s\" in %s", err, libusb_strerror((enum libusb_error)err), func);
  if (err == LIBUSB_ERROR_NO_DEVICE) {
    LOGE("lost connection");
    connected = false;
  }
  // TODO: check other errors, is simply retrying okay?
}

int Panda::hw_control_write(uint8_t request, uint16_t param1, uint16_t param2, unsigned int timeout) {
  int err;
  const uint8_t bmRequestType = LIBUSB_ENDPOINT_OUT | LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_RECIPIENT_DEVICE;

  if (!connected) {
    return LIBUSB_ERROR_NO_DEVICE;
  }

  std::lock_guard lk(hw_lock);
  do {
    if (spi_comms) {
      spi_control_packet packet = {
        .request = request,
        .param1 = param1,
        .param2 = param2,
        .length = 0
      };

      err = spi_transfer(0, (uint8_t *) &packet, sizeof(packet), NULL, 0);
      // TODO: handle error

    } else {
      err = libusb_control_transfer(dev_handle, bmRequestType, request, param1, param2, NULL, 0, timeout);
      if (err < 0) handle_usb_issue(err, __func__);
    }
  } while (err < 0 && connected);

  return err;
}

int Panda::hw_control_read(uint8_t request, uint16_t param1, uint16_t param2, unsigned char *data, uint16_t length, unsigned int timeout) {
  int err;
  const uint8_t bmRequestType = LIBUSB_ENDPOINT_IN | LIBUSB_REQUEST_TYPE_VENDOR | LIBUSB_RECIPIENT_DEVICE;

  if (!connected) {
    return LIBUSB_ERROR_NO_DEVICE;
  }

  std::lock_guard lk(hw_lock);
  do {
    if (spi_comms) {
      spi_control_packet packet = {
        .request = request,
        .param1 = param1,
        .param2 = param2,
        .length = length
      };

      err = spi_transfer(0, (uint8_t *) &packet, sizeof(packet), data, length);
      // TODO: handle error

    } else {
      err = libusb_control_transfer(dev_handle, bmRequestType, request, param1, param2, data, length, timeout);
      if (err < 0) handle_usb_issue(err, __func__);
    }
  } while (err < 0 && connected);

  return err;
}

int Panda::hw_bulk_write(unsigned char endpoint, unsigned char* data, int length, unsigned int timeout) {
  int err;
  int transferred = 0;

  if (!connected) {
    return 0;
  }

  std::lock_guard lk(hw_lock);
  do {
    if (spi_comms) {
      err = spi_transfer(endpoint, (uint8_t *) data, length, NULL, 0);
      // TODO: handle error
    } else {
      // Try sending can messages. If the receive buffer on the panda is full it will NAK
      // and libusb will try again. After 5ms, it will time out. We will drop the messages.
      err = libusb_bulk_transfer(dev_handle, endpoint, data, length, &transferred, timeout);

      if (err == LIBUSB_ERROR_TIMEOUT) {
        LOGW("Transmit buffer full");
        break;
      } else if (err != 0 || length != transferred) {
        handle_usb_issue(err, __func__);
      }
    }
  } while(err != 0 && connected);

  return transferred;
}

int Panda::hw_bulk_read(unsigned char endpoint, unsigned char* data, int length, unsigned int timeout) {
  int err, max_len;
  int transferred = 0;

  if (!connected) {
    return 0;
  }

  std::lock_guard lk(hw_lock);

  do {
    if (spi_comms) {
      max_len = std::min(length, SPI_BUF_SIZE - 1);
      transferred = spi_transfer(endpoint, NULL, 0, (uint8_t *) data + transferred, max_len);

      if (transferred < 0) {
        err = transferred;
      } else if(transferred <= max_len) {
        // read all currently available
        err = 0;
      }

      // TODO: handle error
    } else {
      err = libusb_bulk_transfer(dev_handle, endpoint, data, length, &transferred, timeout);

      if (err == LIBUSB_ERROR_TIMEOUT) {
        break; // timeout is okay to exit, recv still happened
      } else if (err == LIBUSB_ERROR_OVERFLOW) {
        comms_healthy = false;
        LOGE_100("overflow got 0x%x", transferred);
      } else if (err != 0) {
        handle_usb_issue(err, __func__);
      }
    }
  } while(err != 0 && connected);

  return transferred;
}

void Panda::set_safety_model(cereal::CarParams::SafetyModel safety_model, uint16_t safety_param) {
  hw_control_write(0xdc, (uint16_t)safety_model, safety_param);
}

void Panda::set_alternative_experience(uint16_t alternative_experience) {
  hw_control_write(0xdf, alternative_experience, 0);
}

cereal::PandaState::PandaType Panda::get_hw_type() {
  unsigned char hw_query[1] = {0};

  hw_control_read(0xc1, 0, 0, hw_query, 1);
  return (cereal::PandaState::PandaType)(hw_query[0]);
}

void Panda::set_rtc(struct tm sys_time) {
  // tm struct has year defined as years since 1900
  hw_control_write(0xa1, (uint16_t)(1900 + sys_time.tm_year), 0);
  hw_control_write(0xa2, (uint16_t)(1 + sys_time.tm_mon), 0);
  hw_control_write(0xa3, (uint16_t)sys_time.tm_mday, 0);
  // hw_control_write(0xa4, (uint16_t)(1 + sys_time.tm_wday), 0);
  hw_control_write(0xa5, (uint16_t)sys_time.tm_hour, 0);
  hw_control_write(0xa6, (uint16_t)sys_time.tm_min, 0);
  hw_control_write(0xa7, (uint16_t)sys_time.tm_sec, 0);
}

struct tm Panda::get_rtc() {
  struct __attribute__((packed)) timestamp_t {
    uint16_t year; // Starts at 0
    uint8_t month;
    uint8_t day;
    uint8_t weekday;
    uint8_t hour;
    uint8_t minute;
    uint8_t second;
  } rtc_time = {0};

  hw_control_read(0xa0, 0, 0, (unsigned char*)&rtc_time, sizeof(rtc_time));

  struct tm new_time = { 0 };
  new_time.tm_year = rtc_time.year - 1900; // tm struct has year defined as years since 1900
  new_time.tm_mon  = rtc_time.month - 1;
  new_time.tm_mday = rtc_time.day;
  new_time.tm_hour = rtc_time.hour;
  new_time.tm_min  = rtc_time.minute;
  new_time.tm_sec  = rtc_time.second;

  return new_time;
}

void Panda::set_fan_speed(uint16_t fan_speed) {
  hw_control_write(0xb1, fan_speed, 0);
}

uint16_t Panda::get_fan_speed() {
  uint16_t fan_speed_rpm = 0;
  hw_control_read(0xb2, 0, 0, (unsigned char*)&fan_speed_rpm, sizeof(fan_speed_rpm));
  return fan_speed_rpm;
}

void Panda::set_ir_pwr(uint16_t ir_pwr) {
  hw_control_write(0xb0, ir_pwr, 0);
}

std::optional<health_t> Panda::get_state() {
  health_t health {0};
  int err = hw_control_read(0xd2, 0, 0, (unsigned char*)&health, sizeof(health));
  return err >= 0 ? std::make_optional(health) : std::nullopt;
}

std::optional<can_health_t> Panda::get_can_state(uint16_t can_number) {
  can_health_t can_health {0};
  int err = usb_read(0xc2, can_number, 0, (unsigned char*)&can_health, sizeof(can_health));
  return err >= 0 ? std::make_optional(can_health) : std::nullopt;
}

void Panda::set_loopback(bool loopback) {
  hw_control_write(0xe5, loopback, 0);
}

std::optional<std::vector<uint8_t>> Panda::get_firmware_version() {
  std::vector<uint8_t> fw_sig_buf(128);
  int read_1 = hw_control_read(0xd3, 0, 0, &fw_sig_buf[0], 64);
  int read_2 = hw_control_read(0xd4, 0, 0, &fw_sig_buf[64], 64);
  return ((read_1 == 64) && (read_2 == 64)) ? std::make_optional(fw_sig_buf) : std::nullopt;
}

std::optional<std::string> Panda::get_serial() {
  char serial_buf[17] = {'\0'};
  int err = hw_control_read(0xd0, 0, 0, (uint8_t*)serial_buf, 16);
  return err >= 0 ? std::make_optional(serial_buf) : std::nullopt;
}

void Panda::set_power_saving(bool power_saving) {
  hw_control_write(0xe7, power_saving, 0);
}

void Panda::enable_deepsleep() {
  hw_control_write(0xfb, 0, 0);
}

void Panda::send_heartbeat(bool engaged) {
  hw_control_write(0xf3, engaged, 0);
}

void Panda::set_can_speed_kbps(uint16_t bus, uint16_t speed) {
  hw_control_write(0xde, bus, (speed * 10));
}

void Panda::set_data_speed_kbps(uint16_t bus, uint16_t speed) {
  hw_control_write(0xf9, bus, (speed * 10));
}

void Panda::set_canfd_non_iso(uint16_t bus, bool non_iso) {
  usb_write(0xfc, bus, non_iso);
}

static uint8_t len_to_dlc(uint8_t len) {
  if (len <= 8) {
    return len;
  }
  if (len <= 24) {
    return 8 + ((len - 8) / 4) + ((len % 4) ? 1 : 0);
  } else {
    return 11 + (len / 16) + ((len % 16) ? 1 : 0);
  }
}

static void write_packet(uint8_t *dest, int *write_pos, const uint8_t *src, size_t size) {
  for (int i = 0, &pos = *write_pos; i < size; ++i, ++pos) {
    // Insert counter every 64 bytes (first byte of 64 bytes USB packet)
    if (pos % USBPACKET_MAX_SIZE == 0) {
      dest[pos] = pos / USBPACKET_MAX_SIZE;
      pos++;
    }
    dest[pos] = src[i];
  }
}

void Panda::pack_can_buffer(const capnp::List<cereal::CanData>::Reader &can_data_list,
                            std::function<void(uint8_t *, size_t)> write_func) {
  int32_t pos = 0;
  uint8_t send_buf[2 * USB_TX_SOFT_LIMIT];

  for (auto cmsg : can_data_list) {
    // check if the message is intended for this panda
    uint8_t bus = cmsg.getSrc();
    if (bus < bus_offset || bus >= (bus_offset + PANDA_BUS_CNT)) {
      continue;
    }
    auto can_data = cmsg.getDat();
    uint8_t data_len_code = len_to_dlc(can_data.size());
    assert(can_data.size() <= 64);
    assert(can_data.size() == dlc_to_len[data_len_code]);

    can_header header;
    header.addr = cmsg.getAddress();
    header.extended = (cmsg.getAddress() >= 0x800) ? 1 : 0;
    header.data_len_code = data_len_code;
    header.bus = bus - bus_offset;

    write_packet(send_buf, &pos, (uint8_t *)&header, sizeof(can_header));
    write_packet(send_buf, &pos, (uint8_t *)can_data.begin(), can_data.size());
    if (pos >= USB_TX_SOFT_LIMIT) {
      write_func(send_buf, pos);
      pos = 0;
    }
  }

  // send remaining packets
  if (pos > 0) write_func(send_buf, pos);
}

void Panda::can_send(capnp::List<cereal::CanData>::Reader can_data_list) {
  pack_can_buffer(can_data_list, [=](uint8_t* data, size_t size) {
    hw_bulk_write(3, data, size, 5);
  });
}

bool Panda::can_receive(std::vector<can_frame>& out_vec) {
  uint8_t data[RECV_SIZE];
  int recv = hw_bulk_read(0x81, (uint8_t*)data, RECV_SIZE);
  LOGD("can recv %d", recv);
  if (!comms_healthy) {
    return false;
  }
  if (recv == RECV_SIZE) {
    LOGW("Panda receive buffer full");
  }

  return (recv <= 0) ? true : unpack_can_buffer(data, recv, out_vec);
}

bool Panda::unpack_can_buffer(uint8_t *data, int size, std::vector<can_frame> &out_vec) {
  recv_buf.clear();
  for (int i = 0; i < size; i += USBPACKET_MAX_SIZE) {
    // TODO: fix this!
    if (data[i] != i / USBPACKET_MAX_SIZE) {
      LOGE("CAN: MALFORMED USB RECV PACKET %d %d", data[i], i / USBPACKET_MAX_SIZE);
      comms_healthy = false;
      return false;
    }
    int chunk_len = std::min(USBPACKET_MAX_SIZE, (size - i));
    recv_buf.insert(recv_buf.end(), &data[i + 1], &data[i + chunk_len]);
  }

  int pos = 0;
  while (pos < recv_buf.size()) {
    can_header header;
    memcpy(&header, &recv_buf[pos], CANPACKET_HEAD_SIZE);

    can_frame &canData = out_vec.emplace_back();
    canData.busTime = 0;
    canData.address = header.addr;
    canData.src = header.bus + bus_offset;
    if (header.rejected) { canData.src += CANPACKET_REJECTED; }
    if (header.returned) { canData.src += CANPACKET_RETURNED; }

    const uint8_t data_len = dlc_to_len[header.data_len_code];
    canData.dat.assign((char *)&recv_buf[pos + CANPACKET_HEAD_SIZE], data_len);

    pos += CANPACKET_HEAD_SIZE + data_len;
  }
  return true;
}
