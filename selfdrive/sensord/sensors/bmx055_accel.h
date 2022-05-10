#pragma once

#include "selfdrive/sensord/sensors/i2c_sensor.h"

// Address of the chip on the bus
#define BMX055_ACCEL_I2C_ADDR       0x18

// Registers of the chip
#define BMX055_ACCEL_I2C_REG_ID     0x00
#define BMX055_ACCEL_I2C_REG_X_LSB  0x02
#define BMX055_ACCEL_I2C_REG_TEMP   0x08
#define BMX055_ACCEL_I2C_REG_BW     0x10
#define BMX055_ACCEL_I2C_REG_PMU    0x11
#define BMX055_ACCEL_I2C_REG_HBW    0x13
#define BMX055_ACCEL_I2C_REG_FIFO   0x3F

#define BMX055_ACCEL_I2C_REG_INT_EN_1      0x17
#define BMX055_ACCEL_I2C_REG_INT_MAP_1     0x1A
#define BMX055_ACCEL_I2C_REG_INT_SRC       0x1E


// Constants
#define BMX055_ACCEL_CHIP_ID        0xFA

#define BMX055_ACCEL_HBW_ENABLE       0b10000000
#define BMX055_ACCEL_HBW_DISABLE      0b00000000
#define BMX055_ACCEL_DEEP_SUSPEND     0b00100000
#define BMX055_ACCEL_NORMAL_MODE      0b00000000

#define BMX055_ACCEL_BW_7_81HZ  0b01000
#define BMX055_ACCEL_BW_15_63HZ 0b01001
#define BMX055_ACCEL_BW_31_25HZ 0b01010
#define BMX055_ACCEL_BW_62_5HZ  0b01011
#define BMX055_ACCEL_BW_125HZ   0b01100
#define BMX055_ACCEL_BW_250HZ   0b01101
#define BMX055_ACCEL_BW_500HZ   0b01110
#define BMX055_ACCEL_BW_1000HZ  0b01111

#define BMX055_ACCEL_DATA_EN          (1 << 4)
#define BMX055_ACCEL_DATA_TO_INT1     0b1
#define BMX055_ACCEL_INT_SRC_FILTERED 0b0


class BMX055_Accel : public I2CSensor {
  int gpio_nr;
  uint8_t get_device_address() {return BMX055_ACCEL_I2C_ADDR;}
public:
  BMX055_Accel(I2CBus *bus, int gpio_nr);
  int init();
  bool get_event(cereal::SensorEventData::Builder &event);
  int shutdown();
};
