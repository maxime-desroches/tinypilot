#include "common_dbc.h"

namespace {

const Signal sigs_2[] = {
    {
      .name = "COUNTER",
      .b1 = 32,
      .b2 = 4,
      .bo = 28,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "STEER_ANGLE",
      .b1 = 0,
      .b2 = 16,
      .bo = 48,
      .is_signed = true,
      .factor = -0.1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "STEER_ANGLE_RATE",
      .b1 = 16,
      .b2 = 8,
      .bo = 40,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "SET_ME_X07",
      .b1 = 24,
      .b2 = 8,
      .bo = 32,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_42[] = {
    {
      .name = "unknown1",
      .b1 = 0,
      .b2 = 24,
      .bo = 40,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "unknown3",
      .b1 = 24,
      .b2 = 2,
      .bo = 38,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "SEATBELT_DRIVER_UNLATCHED",
      .b1 = 29,
      .b2 = 1,
      .bo = 34,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "SEATBELT_DRIVER_LATCHED",
      .b1 = 27,
      .b2 = 1,
      .bo = 36,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "unknown2",
      .b1 = 24,
      .b2 = 4,
      .bo = 36,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "unknown4",
      .b1 = 32,
      .b2 = 16,
      .bo = 16,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_361[] = {
    {
      .name = "COUNTER",
      .b1 = 52,
      .b2 = 4,
      .bo = 8,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "CHECKSUM",
      .b1 = 56,
      .b2 = 8,
      .bo = 0,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "DESIRED_ANGLE",
      .b1 = 0,
      .b2 = 18,
      .bo = 46,
      .is_signed = false,
      .factor = -0.01,
      .offset = 1310,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "SET_0x80_2",
      .b1 = 24,
      .b2 = 8,
      .bo = 32,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "MAX_TORQUE",
      .b1 = 32,
      .b2 = 8,
      .bo = 24,
      .is_signed = false,
      .factor = 0.01,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "SET_0x80",
      .b1 = 40,
      .b2 = 8,
      .bo = 16,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "LKA_ACTIVE",
      .b1 = 51,
      .b2 = 1,
      .bo = 12,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_389[] = {
    {
      .name = "COUNTER",
      .b1 = 52,
      .b2 = 4,
      .bo = 8,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "CHECKSUM",
      .b1 = 56,
      .b2 = 8,
      .bo = 0,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "STEER_TORQUE_DRIVER",
      .b1 = 0,
      .b2 = 12,
      .bo = 52,
      .is_signed = false,
      .factor = -0.01,
      .offset = 20.47,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "STEER_ANGLE",
      .b1 = 16,
      .b2 = 18,
      .bo = 30,
      .is_signed = false,
      .factor = -0.01,
      .offset = 1310,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "LKAS_ACTIVE",
      .b1 = 34,
      .b2 = 1,
      .bo = 29,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "STEER_TORQUE_LKAS",
      .b1 = 40,
      .b2 = 8,
      .bo = 16,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_460[] = {
    {
      .name = "BRAKE_PEDAL",
      .b1 = 0,
      .b2 = 8,
      .bo = 56,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_569[] = {
    {
      .name = "COUNTER",
      .b1 = 32,
      .b2 = 2,
      .bo = 30,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "GAS_PEDAL",
      .b1 = 0,
      .b2 = 8,
      .bo = 56,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "GAS_PEDAL_INVERTED",
      .b1 = 8,
      .b2 = 8,
      .bo = 48,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "unsure2",
      .b1 = 23,
      .b2 = 1,
      .bo = 40,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "CRUISE_AVAILABLE",
      .b1 = 22,
      .b2 = 1,
      .bo = 41,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "unsure1",
      .b1 = 16,
      .b2 = 6,
      .bo = 42,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "PROPILOT_BUTTON",
      .b1 = 31,
      .b2 = 1,
      .bo = 32,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "CANCEL_BUTTON",
      .b1 = 30,
      .b2 = 1,
      .bo = 33,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "FOLLOW_DISTANCE_BUTTON",
      .b1 = 29,
      .b2 = 1,
      .bo = 34,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "SET_BUTTON",
      .b1 = 28,
      .b2 = 1,
      .bo = 35,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "RES_BUTTON",
      .b1 = 27,
      .b2 = 1,
      .bo = 36,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "NO_BUTTON_PRESSED",
      .b1 = 26,
      .b2 = 1,
      .bo = 37,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "unsure3",
      .b1 = 24,
      .b2 = 2,
      .bo = 38,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "USER_BRAKE_PRESSED",
      .b1 = 34,
      .b2 = 1,
      .bo = 29,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "unsure5",
      .b1 = 40,
      .b2 = 8,
      .bo = 16,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "unsure6",
      .b1 = 48,
      .b2 = 8,
      .bo = 8,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "unsure7",
      .b1 = 56,
      .b2 = 8,
      .bo = 0,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_640[] = {
    {
      .name = "NEW_SIGNAL_2",
      .b1 = 7,
      .b2 = 1,
      .bo = 56,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "CANCEL_SEATBELT",
      .b1 = 6,
      .b2 = 1,
      .bo = 57,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "NEW_SIGNAL_1",
      .b1 = 0,
      .b2 = 6,
      .bo = 58,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "NEW_SIGNAL_3",
      .b1 = 8,
      .b2 = 56,
      .bo = 0,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_644[] = {
    {
      .name = "WHEEL_SPEED_FR",
      .b1 = 0,
      .b2 = 16,
      .bo = 48,
      .is_signed = false,
      .factor = 0.005,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "WHEEL_SPEED_FL",
      .b1 = 16,
      .b2 = 16,
      .bo = 32,
      .is_signed = false,
      .factor = 0.005,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_645[] = {
    {
      .name = "WHEEL_SPEED_RR",
      .b1 = 0,
      .b2 = 16,
      .bo = 48,
      .is_signed = false,
      .factor = 0.005,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "WHEEL_SPEED_RL",
      .b1 = 16,
      .b2 = 16,
      .bo = 32,
      .is_signed = false,
      .factor = 0.005,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_689[] = {
    {
      .name = "unknown02",
      .b1 = 6,
      .b2 = 2,
      .bo = 56,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "FOLLOW_DISTANCE",
      .b1 = 4,
      .b2 = 2,
      .bo = 58,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "unknown05",
      .b1 = 2,
      .b2 = 2,
      .bo = 60,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "SPEED_SET_ICON",
      .b1 = 0,
      .b2 = 2,
      .bo = 62,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "unknown08",
      .b1 = 15,
      .b2 = 7,
      .bo = 42,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "LARGE_WARNING_FLASHING",
      .b1 = 14,
      .b2 = 1,
      .bo = 49,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "SIDE_RADAR_ERROR_FLASHING1",
      .b1 = 13,
      .b2 = 1,
      .bo = 50,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "SIDE_RADAR_ERROR_FLASHING2",
      .b1 = 12,
      .b2 = 1,
      .bo = 51,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "RIGHT_LANE_YELLOW_FLASH",
      .b1 = 11,
      .b2 = 1,
      .bo = 52,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "LEFT_LANE_YELLOW_FLASH",
      .b1 = 10,
      .b2 = 1,
      .bo = 53,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "LEAD_CAR",
      .b1 = 9,
      .b2 = 1,
      .bo = 54,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "LEAD_CAR_ERROR",
      .b1 = 8,
      .b2 = 1,
      .bo = 55,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "FRONT_RADAR_ERROR",
      .b1 = 23,
      .b2 = 1,
      .bo = 40,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "FRONT_RADAR_ERROR_FLASHING",
      .b1 = 22,
      .b2 = 1,
      .bo = 41,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "RIGHT_LANE_GREEN",
      .b1 = 31,
      .b2 = 1,
      .bo = 32,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "LEFT_LANE_GREEN",
      .b1 = 30,
      .b2 = 1,
      .bo = 33,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "unknown26",
      .b1 = 29,
      .b2 = 1,
      .bo = 34,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "SIDE_RADAR_ERROR_FLASHING3",
      .b1 = 28,
      .b2 = 1,
      .bo = 35,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "unknown28",
      .b1 = 27,
      .b2 = 1,
      .bo = 36,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "LKAS_ERROR_FLASHING",
      .b1 = 26,
      .b2 = 1,
      .bo = 37,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "unknown31",
      .b1 = 24,
      .b2 = 2,
      .bo = 38,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "SET_SPEED",
      .b1 = 32,
      .b2 = 8,
      .bo = 24,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "SMALL_STEERING_WHEEL_ICON",
      .b1 = 45,
      .b2 = 3,
      .bo = 16,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "unknown43",
      .b1 = 44,
      .b2 = 1,
      .bo = 19,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "SAFETY_SHIELD_ACTIVE",
      .b1 = 43,
      .b2 = 1,
      .bo = 20,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "AUDIBLE_TONE",
      .b1 = 40,
      .b2 = 3,
      .bo = 21,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "unknown55",
      .b1 = 48,
      .b2 = 8,
      .bo = 8,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "unknown59",
      .b1 = 60,
      .b2 = 4,
      .bo = 0,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "LARGE_STEERING_WHEEL_ICON",
      .b1 = 58,
      .b2 = 2,
      .bo = 4,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "RIGHT_LANE_GREEN_FLASH",
      .b1 = 57,
      .b2 = 1,
      .bo = 6,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "LEFT_LANE_GREEN_FLASH",
      .b1 = 56,
      .b2 = 1,
      .bo = 7,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_783[] = {
    {
      .name = "CRUISE_ENABLED",
      .b1 = 4,
      .b2 = 1,
      .bo = 59,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_852[] = {
    {
      .name = "ESP_DISABLED",
      .b1 = 33,
      .b2 = 1,
      .bo = 30,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_853[] = {
    {
      .name = "SPEED_MPH",
      .b1 = 34,
      .b2 = 1,
      .bo = 29,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_856[] = {
    {
      .name = "LEFT_BLINKER",
      .b1 = 22,
      .b2 = 1,
      .bo = 41,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "RIGHT_BLINKER",
      .b1 = 21,
      .b2 = 1,
      .bo = 42,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_1057[] = {
    {
      .name = "GEAR_SHIFTER",
      .b1 = 2,
      .b2 = 3,
      .bo = 59,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_1227[] = {
    {
      .name = "LKAS_ENABLED",
      .b1 = 52,
      .b2 = 1,
      .bo = 11,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_1228[] = {
    {
      .name = "NA_HIGH_ACCEL_TEMP",
      .b1 = 7,
      .b2 = 1,
      .bo = 56,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "unknown07",
      .b1 = 0,
      .b2 = 7,
      .bo = 57,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "SIDE_RADAR_NA_HIGH_CABIN_TEMP",
      .b1 = 15,
      .b2 = 1,
      .bo = 48,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "unknown10",
      .b1 = 13,
      .b2 = 2,
      .bo = 49,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "SIDE_RADAR_MALFUNCTION",
      .b1 = 12,
      .b2 = 1,
      .bo = 51,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "LKAS_MALFUNCTION",
      .b1 = 11,
      .b2 = 1,
      .bo = 52,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "FRONT_RADAR_MALFUNCTION",
      .b1 = 10,
      .b2 = 1,
      .bo = 53,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "SIDE_RADAR_NA_CLEAN_REAR_CAMERA",
      .b1 = 9,
      .b2 = 1,
      .bo = 54,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "unknown15",
      .b1 = 8,
      .b2 = 1,
      .bo = 55,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "NA_POOR_ROAD_CONDITIONS",
      .b1 = 23,
      .b2 = 1,
      .bo = 40,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "CURRENTLY_UNAVAILABLE",
      .b1 = 22,
      .b2 = 1,
      .bo = 41,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "SAFETY_SHIELD_OFF",
      .b1 = 21,
      .b2 = 1,
      .bo = 42,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "unknown19",
      .b1 = 20,
      .b2 = 1,
      .bo = 43,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "FRONT_COLLISION_NA_FRONT_RADAR_OBSTRUCTION",
      .b1 = 19,
      .b2 = 1,
      .bo = 44,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "unknown23",
      .b1 = 16,
      .b2 = 3,
      .bo = 45,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "PEDAL_MISSAPPLICATION_SYSTEM_ACTIVATED",
      .b1 = 31,
      .b2 = 1,
      .bo = 32,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "SIDE_IMPACT_NA_RADAR_OBSTRUCTION",
      .b1 = 30,
      .b2 = 1,
      .bo = 33,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "unknown31",
      .b1 = 24,
      .b2 = 6,
      .bo = 34,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "unknown32",
      .b1 = 39,
      .b2 = 1,
      .bo = 24,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "WARNING_DO_NOT_ENTER",
      .b1 = 38,
      .b2 = 1,
      .bo = 25,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "SIDE_IMPACT_SYSTEM_OFF",
      .b1 = 37,
      .b2 = 1,
      .bo = 26,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "SIDE_IMPACT_MALFUNCTION",
      .b1 = 36,
      .b2 = 1,
      .bo = 27,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "FRONT_COLLISION_MALFUNCTION",
      .b1 = 35,
      .b2 = 1,
      .bo = 28,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "SIDE_RADAR_MALFUNCTION2",
      .b1 = 34,
      .b2 = 1,
      .bo = 29,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "LKAS_MALFUNCTION2",
      .b1 = 33,
      .b2 = 1,
      .bo = 30,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "FRONT_RADAR_MALFUNCTION2",
      .b1 = 32,
      .b2 = 1,
      .bo = 31,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "PROPILOT_NA_MSGS",
      .b1 = 45,
      .b2 = 3,
      .bo = 16,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "BOTTOM_MSG",
      .b1 = 42,
      .b2 = 3,
      .bo = 19,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "unknown46",
      .b1 = 41,
      .b2 = 1,
      .bo = 22,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "HANDS_ON_WHEEL_WARNING",
      .b1 = 40,
      .b2 = 1,
      .bo = 23,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "unknown50",
      .b1 = 53,
      .b2 = 3,
      .bo = 8,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "WARNING_STEP_ON_BRAKE_NOW",
      .b1 = 52,
      .b2 = 1,
      .bo = 11,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "PROPILOT_NA_FRONT_CAMERA_OBSTRUCTED",
      .b1 = 51,
      .b2 = 1,
      .bo = 12,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "PROPILOT_NA_HIGH_CABIN_TEMP",
      .b1 = 50,
      .b2 = 1,
      .bo = 13,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "WARNING_PROPILOT_MALFUNCTION",
      .b1 = 49,
      .b2 = 1,
      .bo = 14,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "unknown55",
      .b1 = 48,
      .b2 = 1,
      .bo = 15,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "unknown61",
      .b1 = 58,
      .b2 = 6,
      .bo = 0,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "ACC_UNAVAILABLE_HIGH_CABIN_TEMP",
      .b1 = 57,
      .b2 = 1,
      .bo = 6,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "ACC_NA_FRONT_CAMERA_IMPARED",
      .b1 = 56,
      .b2 = 1,
      .bo = 7,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_1549[] = {
    {
      .name = "DOOR_OPEN_FL",
      .b1 = 4,
      .b2 = 1,
      .bo = 59,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "DOOR_OPEN_FR",
      .b1 = 3,
      .b2 = 1,
      .bo = 60,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "DOOR_OPEN_RL",
      .b1 = 2,
      .b2 = 1,
      .bo = 61,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "DOOR_OPEN_RR",
      .b1 = 1,
      .b2 = 1,
      .bo = 62,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};

const Msg msgs[] = {
  {
    .name = "STEER_ANGLE_SENSOR",
    .address = 0x2,
    .size = 5,
    .num_sigs = ARRAYSIZE(sigs_2),
    .sigs = sigs_2,
  },
  {
    .name = "SEATBELT",
    .address = 0x2A,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_42),
    .sigs = sigs_42,
  },
  {
    .name = "LKAS",
    .address = 0x169,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_361),
    .sigs = sigs_361,
  },
  {
    .name = "STEER_TORQUE_SENSOR",
    .address = 0x185,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_389),
    .sigs = sigs_389,
  },
  {
    .name = "BRAKE_PEDAL",
    .address = 0x1CC,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_460),
    .sigs = sigs_460,
  },
  {
    .name = "CRUISE_THROTTLE",
    .address = 0x239,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_569),
    .sigs = sigs_569,
  },
  {
    .name = "CANCEL_MSG",
    .address = 0x280,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_640),
    .sigs = sigs_640,
  },
  {
    .name = "WHEEL_SPEEDS_FRONT",
    .address = 0x284,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_644),
    .sigs = sigs_644,
  },
  {
    .name = "WHEEL_SPEEDS_REAR",
    .address = 0x285,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_645),
    .sigs = sigs_645,
  },
  {
    .name = "PROPILOT_HUD",
    .address = 0x2B1,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_689),
    .sigs = sigs_689,
  },
  {
    .name = "CRUISE_STATE",
    .address = 0x30F,
    .size = 3,
    .num_sigs = ARRAYSIZE(sigs_783),
    .sigs = sigs_783,
  },
  {
    .name = "ESP",
    .address = 0x354,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_852),
    .sigs = sigs_852,
  },
  {
    .name = "HUD_SETTINGS",
    .address = 0x355,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_853),
    .sigs = sigs_853,
  },
  {
    .name = "LIGHTS",
    .address = 0x358,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_856),
    .sigs = sigs_856,
  },
  {
    .name = "GEARBOX",
    .address = 0x421,
    .size = 3,
    .num_sigs = ARRAYSIZE(sigs_1057),
    .sigs = sigs_1057,
  },
  {
    .name = "LKAS_SETTINGS",
    .address = 0x4CB,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_1227),
    .sigs = sigs_1227,
  },
  {
    .name = "PROPILOT_HUD_INFO_MSG",
    .address = 0x4CC,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_1228),
    .sigs = sigs_1228,
  },
  {
    .name = "DOORS_LIGHTS",
    .address = 0x60D,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_1549),
    .sigs = sigs_1549,
  },
};

const Val vals[] = {
    {
      .name = "AUDIBLE_TONE",
      .address = 0x2B1,
      .def_val = "0 NO_TONE 1 CONT 2 FAST_BEEP_CONT 3 TRIPLE_FAST_BEEP_CONT 4 SLOW_BEEP_CONT 5 QUAD_SLOW_BEEP_CONT 6 SINGLE_BEEP_ONCE 7 DOUBLE_BEEP_ONCE",
      .sigs = sigs_689,
    },
    {
      .name = "FOLLOW_DISTANCE",
      .address = 0x2B1,
      .def_val = "0 NO_FOLLOW_DISTANCE 1 FOLLOW_DISTANCE_1 2 FOLLOW_DISTANCE_2 3 FOLLOW_DISANCE_3",
      .sigs = sigs_689,
    },
    {
      .name = "LARGE_STEERING_WHEEL_ICON",
      .address = 0x2B1,
      .def_val = "0 NO_STEERINGWHEEL 1 GRAY_STEERINGWHEEL 2 GREEN_STEERINGWHEEL 3 GREEN_STEERINGWHEEL_FLASHING",
      .sigs = sigs_689,
    },
    {
      .name = "SMALL_STEERING_WHEEL_ICON",
      .address = 0x2B1,
      .def_val = "0 NO_ICON 1 GRAY_ICON 2 GRAY_ICON_FLASHING 3 GREEN_ICON 4 GREEN_ICON_FLASHING 5 RED_ICON 6 RED_ICON_FLASHING 7 YELLOW_ICON",
      .sigs = sigs_689,
    },
    {
      .name = "GEAR_SHIFTER",
      .address = 0x421,
      .def_val = "7 B 4 D 3 N 2 R 1 P",
      .sigs = sigs_1057,
    },
    {
      .name = "BOTTOM_MSG",
      .address = 0x4CC,
      .def_val = "0 OK_STEER_ASSIST_SETTINGS 1 NO_MSG 2 PRESS_SET_TO_SET_SPEED 3 PRESS_RES_SET_TO_CHANGE_SPEED 4 PRESS_RES_TO_RESTART 5 NO_MSG 6 CRUISE_NOT_AVAIL 7 NO_MSG",
      .sigs = sigs_1228,
    },
    {
      .name = "PROPILOT_NA_MSGS",
      .address = 0x4CC,
      .def_val = "0 NO_MSG 1 NA_FRONT_CAMERA_IMPARED 2 STEERING_ASSIST_ON_STANDBY 3 NA_PARKING_ASSIST_ENABLED 4 STEER_ASSIST_CURRENTLY_NA 5 NA_BAD_WEATHER 6 NA_PARK_BRAKE_ON 7 NA_SEATBELT_NOT_FASTENED",
      .sigs = sigs_1228,
    },
};

}

const DBC nissan_leaf_2018 = {
  .name = "nissan_leaf_2018",
  .num_msgs = ARRAYSIZE(msgs),
  .msgs = msgs,
  .vals = vals,
  .num_vals = ARRAYSIZE(vals),
};

dbc_init(nissan_leaf_2018)