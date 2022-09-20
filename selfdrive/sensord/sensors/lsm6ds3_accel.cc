#include "lsm6ds3_accel.h"

#include <cassert>
#include <cmath>

#include "common/swaglog.h"
#include "common/timing.h"
#include "common/util.h"

LSM6DS3_Accel::LSM6DS3_Accel(I2CBus *bus, int gpio_nr, bool shared_gpio) :
  I2CSensor(bus, gpio_nr, shared_gpio) {}

void LSM6DS3_Accel::wait_for_data_ready() {
  uint8_t drdy = 0;
  uint8_t buffer[6];

  do {
    read_register(LSM6DS3_ACCEL_I2C_REG_STAT_REG, &drdy, sizeof(drdy));
    drdy &= LSM6DS3_ACCEL_DRDY_XLDA;
  } while(drdy == 0);

  // read first values and discard
  int len = read_register(LSM6DS3_ACCEL_I2C_REG_OUTX_L_XL, buffer, sizeof(buffer));
  assert(len == sizeof(buffer));
}

void LSM6DS3_Accel::read_and_avg_data(float* out_buf) {
  uint8_t drdy = 0;
  uint8_t buffer[6];

  float scaleing = 0.061f;
  if (source == cereal::SensorEventData::SensorSource::LSM6DS3TRC) {
    scaleing = 0.122f;
  }

  for (int i = 0; i < 5; i++) {
    // check for new data
    do {
      read_register(LSM6DS3_ACCEL_I2C_REG_STAT_REG, &drdy, sizeof(drdy));
      drdy &= LSM6DS3_ACCEL_DRDY_XLDA;
    } while(drdy == 0);

    int len = read_register(LSM6DS3_ACCEL_I2C_REG_OUTX_L_XL, buffer, sizeof(buffer));
    assert(len == sizeof(buffer));

    for (int j = 0; j < 3; j++) {
      out_buf[j] += (float)read_16_bit(buffer[j*2], buffer[j*2+1]) * scaleing;
    }
  }

  // calculate the mg average values
  for (int i = 0; i < 3; i++) {
    out_buf[i] /= 5.0f;
  }
}

int LSM6DS3_Accel::perform_self_test(int test_type) {
  int i, ret = 0;
  float val_st_off[3] = {0};
  float val_st_on[3] = {0};
  float test_val[3] = {0};
  bool test_result = true;
  uint8_t ODR_FS_MO = LSM6DS3_ACCEL_ODR_52HZ; // full scale: +-2g, ODR: 52Hz

  // prepare sensor for self-test

  // enable block data update and automatic increment
  ret = set_register(LSM6DS3_ACCEL_I2C_REG_CTRL3_C, LSM6DS3_ACCEL_IF_INC_BDU);
  if (ret < 0) {
    goto fail;
  }

  if (source == cereal::SensorEventData::SensorSource::LSM6DS3TRC) {
    ODR_FS_MO = LSM6DS3_ACCEL_FS_4G | LSM6DS3_ACCEL_ODR_52HZ;
  }
  ret = set_register(LSM6DS3_ACCEL_I2C_REG_CTRL1_XL, ODR_FS_MO);
  if (ret < 0) {
    goto fail;
  }

  /*
  if (source == cereal::SensorEventData::SensorSource::LSM6DS3TRC) {
    // 0x18, 0x19 have different meaning on lsm6ds c and non c variant
    ret = set_register(LSM6DS3_ACCEL_I2C_REG_CTR9_XL, 0);
    if (ret < 0) {
      goto fail;
    }
  }*/

  // wait for stable output, and discard first values
  util::sleep_for(100);
  wait_for_data_ready();
  read_and_avg_data(val_st_off);

  // enable Self Test positive (or negative)
  ret = set_register(LSM6DS3_ACCEL_I2C_REG_CTRL5_C, test_type);
  if (ret < 0) {
    goto fail;
  }

  // wait for stable output, and discard first values
  util::sleep_for(100);
  wait_for_data_ready();
  read_and_avg_data(val_st_on);

  // calculate the mg values for self test
  for (i = 0; i < 3; i++) {
    test_val[i] = fabs((val_st_on[i] - val_st_off[i]));
  }

  // verify test result
  for (i = 0; i < 3; i++) {
    if ((LSM6DS3_ACCEL_MIN_ST_LIMIT_mg > test_val[i]) ||
        (test_val[i] > LSM6DS3_ACCEL_MAX_ST_LIMIT_mg)) {
      test_result = false;
    }
  }

  // disable sensor
  ret = set_register(LSM6DS3_ACCEL_I2C_REG_CTRL1_XL, 0);
  if (ret < 0) {
    goto fail;
  }

  // disable self test
  ret = set_register(LSM6DS3_ACCEL_I2C_REG_CTRL5_C, 0);
  if (ret < 0) {
    goto fail;
  }

  if (!test_result) {
    ret = -1;
  }

fail:
  return ret;
}

int LSM6DS3_Accel::init() {
  int ret = 0;
  uint8_t buffer[1];
  uint8_t value = 0;

  ret = read_register(LSM6DS3_ACCEL_I2C_REG_ID, buffer, 1);
  if(ret < 0) {
    LOGE("Reading chip ID failed: %d", ret);
    goto fail;
  }

  if(buffer[0] != LSM6DS3_ACCEL_CHIP_ID && buffer[0] != LSM6DS3TRC_ACCEL_CHIP_ID) {
    LOGE("Chip ID wrong. Got: %d, Expected %d", buffer[0], LSM6DS3_ACCEL_CHIP_ID);
    ret = -1;
    goto fail;
  }

  if (buffer[0] == LSM6DS3TRC_ACCEL_CHIP_ID) {
    source = cereal::SensorEventData::SensorSource::LSM6DS3TRC;
  }

  ret = perform_self_test(LSM6DS3_ACCEL_POSITIVE_TEST);
  if (ret < 0) {
    LOGE("LSM6DS3 accel positive self-test failed!");
    goto fail;
  }

  ret = perform_self_test(LSM6DS3_ACCEL_NEGATIVE_TEST);
  if (ret < 0) {
    LOGE("LSM6DS3 accel negative self-test failed!");
    goto fail;
  }

  ret = init_gpio();
  if (ret < 0) {
    goto fail;
  }

  // enable continuous update, and automatic increase
  ret = set_register(LSM6DS3_ACCEL_I2C_REG_CTRL3_C, LSM6DS3_ACCEL_IF_INC);
  if (ret < 0) {
    goto fail;
  }

  /*
  if (source == cereal::SensorEventData::SensorSource::LSM6DS3TRC) {
    // data enable for x, y, z axis
    ret = set_register(LSM6DS3_ACCEL_I2C_REG_CTR9_XL, LSM6DS3_ACCEL_XYZ_DEN);
    if (ret < 0) {
      goto fail;
    }
  }*/

  // TODO: set scale and bandwidth. Default is +- 2G, 50 Hz
  ret = set_register(LSM6DS3_ACCEL_I2C_REG_CTRL1_XL, LSM6DS3_ACCEL_ODR_104HZ);
  if (ret < 0) {
    goto fail;
  }

  ret = set_register(LSM6DS3_ACCEL_I2C_REG_DRDY_CFG, LSM6DS3_ACCEL_DRDY_PULSE_MODE);
  if (ret < 0) {
    goto fail;
  }

  // enable data ready interrupt for accel on INT1
  // (without resetting existing interrupts)
  ret = read_register(LSM6DS3_ACCEL_I2C_REG_INT1_CTRL, &value, 1);
  if (ret < 0) {
    goto fail;
  }

  value |= LSM6DS3_ACCEL_INT1_DRDY_XL;
  ret = set_register(LSM6DS3_ACCEL_I2C_REG_INT1_CTRL, value);

fail:
  return ret;
}

int LSM6DS3_Accel::shutdown() {
  int ret = 0;

  // disable data ready interrupt for accel on INT1
  uint8_t value = 0;
  ret = read_register(LSM6DS3_ACCEL_I2C_REG_INT1_CTRL, &value, 1);
  if (ret < 0) {
    goto fail;
  }

  value &= ~(LSM6DS3_ACCEL_INT1_DRDY_XL);
  ret = set_register(LSM6DS3_ACCEL_I2C_REG_INT1_CTRL, value);
  if (ret < 0) {
    LOGE("Could not disable lsm6ds3 acceleration interrupt!")
    goto fail;
  }

  // enable power-down mode
  value = 0;
  ret = read_register(LSM6DS3_ACCEL_I2C_REG_CTRL1_XL, &value, 1);
  if (ret < 0) {
    goto fail;
  }

  value &= 0x0F;
  ret = set_register(LSM6DS3_ACCEL_I2C_REG_CTRL1_XL, value);
  if (ret < 0) {
    LOGE("Could not power-down lsm6ds3 accelerometer!")
    goto fail;
  }

fail:
  return ret;
}

bool LSM6DS3_Accel::get_event(MessageBuilder &msg, uint64_t ts) {

  // INT1 shared with gyro, check STATUS_REG who triggered
  uint8_t status_reg = 0;
  read_register(LSM6DS3_ACCEL_I2C_REG_STAT_REG, &status_reg, sizeof(status_reg));
  if ((status_reg & LSM6DS3_ACCEL_DRDY_XLDA) == 0) {
    return false;
  }

  uint8_t buffer[6];
  int len = read_register(LSM6DS3_ACCEL_I2C_REG_OUTX_L_XL, buffer, sizeof(buffer));
  assert(len == sizeof(buffer));

  float scale = 9.81 * 2.0f / (1 << 15);
  float x = read_16_bit(buffer[0], buffer[1]) * scale;
  float y = read_16_bit(buffer[2], buffer[3]) * scale;
  float z = read_16_bit(buffer[4], buffer[5]) * scale;

  auto event = msg.initEvent().initAccelerometer();
  event.setSource(source);
  event.setVersion(1);
  event.setSensor(SENSOR_ACCELEROMETER);
  event.setType(SENSOR_TYPE_ACCELEROMETER);
  event.setTimestamp(ts);

  float xyz[] = {y, -x, z};
  auto svec = event.initAcceleration();
  svec.setV(xyz);
  svec.setStatus(true);

  return true;
}
