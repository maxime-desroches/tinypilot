#include "common_dbc.h"

namespace {

const Signal sigs_2[] = {
    {
      .name = "Steering_Angle",
      .b1 = 0,
      .b2 = 16,
      .bo = 48,
      .is_signed = true,
      .factor = 0.1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Counter",
      .b1 = 28,
      .b2 = 3,
      .bo = 33,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Checksum",
      .b1 = 32,
      .b2 = 8,
      .bo = 24,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_208[] = {
    {
      .name = "Steering_Angle",
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
      .name = "Lateral",
      .b1 = 16,
      .b2 = 16,
      .bo = 32,
      .is_signed = true,
      .factor = -0.0035,
      .offset = 1,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Longitudinal",
      .b1 = 48,
      .b2 = 16,
      .bo = 0,
      .is_signed = true,
      .factor = -0.00035,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_209[] = {
    {
      .name = "Speed",
      .b1 = 0,
      .b2 = 16,
      .bo = 48,
      .is_signed = false,
      .factor = 0.05625,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Brake_Pedal",
      .b1 = 16,
      .b2 = 8,
      .bo = 40,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_210[] = {
    {
      .name = "Brake_Light",
      .b1 = 35,
      .b2 = 1,
      .bo = 28,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Brake_Related",
      .b1 = 36,
      .b2 = 1,
      .bo = 27,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Right_Brake",
      .b1 = 48,
      .b2 = 8,
      .bo = 8,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Left_Brake",
      .b1 = 56,
      .b2 = 8,
      .bo = 0,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_211[] = {
    {
      .name = "Brake_Light",
      .b1 = 21,
      .b2 = 1,
      .bo = 42,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Speed_Counter",
      .b1 = 32,
      .b2 = 8,
      .bo = 24,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Brake_Cruise_On",
      .b1 = 42,
      .b2 = 1,
      .bo = 21,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Brake_Pedal_On",
      .b1 = 46,
      .b2 = 1,
      .bo = 17,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Counter",
      .b1 = 48,
      .b2 = 8,
      .bo = 8,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_212[] = {
    {
      .name = "FL",
      .b1 = 0,
      .b2 = 16,
      .bo = 48,
      .is_signed = false,
      .factor = 0.0592,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "FR",
      .b1 = 16,
      .b2 = 16,
      .bo = 32,
      .is_signed = false,
      .factor = 0.0592,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "RL",
      .b1 = 32,
      .b2 = 16,
      .bo = 16,
      .is_signed = false,
      .factor = 0.0592,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "RR",
      .b1 = 48,
      .b2 = 16,
      .bo = 0,
      .is_signed = false,
      .factor = 0.0592,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_320[] = {
    {
      .name = "Throttle_Pedal",
      .b1 = 0,
      .b2 = 8,
      .bo = 56,
      .is_signed = false,
      .factor = 0.392157,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Counter",
      .b1 = 8,
      .b2 = 4,
      .bo = 52,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Not_Full_Throttle",
      .b1 = 14,
      .b2 = 1,
      .bo = 49,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Engine_RPM",
      .b1 = 16,
      .b2 = 14,
      .bo = 34,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Off_Throttle",
      .b1 = 30,
      .b2 = 1,
      .bo = 33,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Throttle_Cruise",
      .b1 = 32,
      .b2 = 8,
      .bo = 24,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Throttle_Combo",
      .b1 = 40,
      .b2 = 8,
      .bo = 16,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Throttle_Body",
      .b1 = 48,
      .b2 = 8,
      .bo = 8,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Off_Throttle_2",
      .b1 = 56,
      .b2 = 1,
      .bo = 7,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_321[] = {
    {
      .name = "Engine_Torque",
      .b1 = 0,
      .b2 = 15,
      .bo = 49,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Engine_Stop",
      .b1 = 15,
      .b2 = 1,
      .bo = 48,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Wheel_Torque",
      .b1 = 16,
      .b2 = 12,
      .bo = 36,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Engine_RPM",
      .b1 = 32,
      .b2 = 12,
      .bo = 20,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_324[] = {
    {
      .name = "OnOffButton",
      .b1 = 2,
      .b2 = 1,
      .bo = 61,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "SET_BUTTON",
      .b1 = 3,
      .b2 = 1,
      .bo = 60,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "RES_BUTTON",
      .b1 = 4,
      .b2 = 1,
      .bo = 59,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Button",
      .b1 = 13,
      .b2 = 1,
      .bo = 50,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Cruise_On",
      .b1 = 48,
      .b2 = 1,
      .bo = 15,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Cruise_Activated",
      .b1 = 49,
      .b2 = 1,
      .bo = 14,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Brake_Pedal_On",
      .b1 = 51,
      .b2 = 1,
      .bo = 12,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_328[] = {
    {
      .name = "Manual_Gear",
      .b1 = 4,
      .b2 = 4,
      .bo = 56,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Counter",
      .b1 = 8,
      .b2 = 4,
      .bo = 52,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Transmission_Engine",
      .b1 = 16,
      .b2 = 15,
      .bo = 33,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Gear",
      .b1 = 48,
      .b2 = 4,
      .bo = 12,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Gear_2",
      .b1 = 52,
      .b2 = 4,
      .bo = 8,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Paddle_Shift",
      .b1 = 60,
      .b2 = 2,
      .bo = 2,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_336[] = {
    {
      .name = "Brake_Pressure_Right",
      .b1 = 0,
      .b2 = 8,
      .bo = 56,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Brake_Pressure_Left",
      .b1 = 8,
      .b2 = 8,
      .bo = 48,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_338[] = {
    {
      .name = "Counter",
      .b1 = 12,
      .b2 = 4,
      .bo = 48,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Brake_Light",
      .b1 = 52,
      .b2 = 1,
      .bo = 11,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Runlights",
      .b1 = 58,
      .b2 = 1,
      .bo = 5,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Headlights",
      .b1 = 59,
      .b2 = 1,
      .bo = 4,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Highbeam",
      .b1 = 60,
      .b2 = 1,
      .bo = 3,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Wiper",
      .b1 = 62,
      .b2 = 1,
      .bo = 1,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_352[] = {
    {
      .name = "Brake_Pressure",
      .b1 = 0,
      .b2 = 16,
      .bo = 48,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Brake_Light",
      .b1 = 20,
      .b2 = 1,
      .bo = 43,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "ES_Error",
      .b1 = 21,
      .b2 = 1,
      .bo = 42,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Brake_On",
      .b1 = 22,
      .b2 = 1,
      .bo = 41,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Cruise_Activated",
      .b1 = 23,
      .b2 = 1,
      .bo = 40,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Counter",
      .b1 = 48,
      .b2 = 3,
      .bo = 13,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Checksum",
      .b1 = 56,
      .b2 = 8,
      .bo = 0,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_353[] = {
    {
      .name = "Throttle_Cruise",
      .b1 = 0,
      .b2 = 12,
      .bo = 52,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Signal1",
      .b1 = 12,
      .b2 = 4,
      .bo = 48,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Cruise_Activated",
      .b1 = 16,
      .b2 = 1,
      .bo = 47,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Signal2",
      .b1 = 17,
      .b2 = 3,
      .bo = 44,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Brake_On",
      .b1 = 20,
      .b2 = 1,
      .bo = 43,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "DistanceSwap",
      .b1 = 21,
      .b2 = 1,
      .bo = 42,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Standstill",
      .b1 = 22,
      .b2 = 1,
      .bo = 41,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Signal3",
      .b1 = 23,
      .b2 = 1,
      .bo = 40,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "CloseDistance",
      .b1 = 24,
      .b2 = 8,
      .bo = 32,
      .is_signed = false,
      .factor = 0.0196,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Signal4",
      .b1 = 32,
      .b2 = 9,
      .bo = 23,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Standstill_2",
      .b1 = 41,
      .b2 = 1,
      .bo = 22,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "ES_Error",
      .b1 = 42,
      .b2 = 1,
      .bo = 21,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Signal5",
      .b1 = 43,
      .b2 = 1,
      .bo = 20,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Counter",
      .b1 = 44,
      .b2 = 3,
      .bo = 17,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Signal6",
      .b1 = 47,
      .b2 = 1,
      .bo = 16,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Button",
      .b1 = 48,
      .b2 = 3,
      .bo = 13,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Signal7",
      .b1 = 51,
      .b2 = 5,
      .bo = 8,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Checksum",
      .b1 = 56,
      .b2 = 8,
      .bo = 0,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_354[] = {
    {
      .name = "Brake",
      .b1 = 8,
      .b2 = 1,
      .bo = 55,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Cruise_Activated",
      .b1 = 9,
      .b2 = 1,
      .bo = 54,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "RPM",
      .b1 = 16,
      .b2 = 16,
      .bo = 32,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Checksum",
      .b1 = 32,
      .b2 = 8,
      .bo = 24,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Counter",
      .b1 = 48,
      .b2 = 3,
      .bo = 13,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_356[] = {
    {
      .name = "Counter",
      .b1 = 0,
      .b2 = 3,
      .bo = 61,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "LKAS_Command",
      .b1 = 8,
      .b2 = 13,
      .bo = 43,
      .is_signed = true,
      .factor = -1.0,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "LKAS_Active",
      .b1 = 24,
      .b2 = 1,
      .bo = 39,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Checksum",
      .b1 = 56,
      .b2 = 8,
      .bo = 0,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_358[] = {
    {
      .name = "Not_Ready_Startup",
      .b1 = 0,
      .b2 = 3,
      .bo = 61,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Seatbelt_Disengage",
      .b1 = 12,
      .b2 = 2,
      .bo = 50,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Disengage_Alert",
      .b1 = 14,
      .b2 = 2,
      .bo = 48,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Cruise_On",
      .b1 = 16,
      .b2 = 1,
      .bo = 47,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Cruise_Activated",
      .b1 = 17,
      .b2 = 1,
      .bo = 46,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Signal1",
      .b1 = 18,
      .b2 = 1,
      .bo = 45,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "WHEELS_MOVING_2015",
      .b1 = 19,
      .b2 = 1,
      .bo = 44,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Driver_Input",
      .b1 = 20,
      .b2 = 1,
      .bo = 43,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Distance_Bars",
      .b1 = 21,
      .b2 = 3,
      .bo = 40,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Cruise_Set_Speed",
      .b1 = 24,
      .b2 = 8,
      .bo = 32,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "ES_Error",
      .b1 = 32,
      .b2 = 1,
      .bo = 31,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Cruise_On_2",
      .b1 = 34,
      .b2 = 1,
      .bo = 29,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Counter",
      .b1 = 37,
      .b2 = 3,
      .bo = 24,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Steep_Hill_Disengage",
      .b1 = 44,
      .b2 = 1,
      .bo = 19,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Lead_Car",
      .b1 = 46,
      .b2 = 1,
      .bo = 17,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Obstacle_Distance",
      .b1 = 48,
      .b2 = 4,
      .bo = 12,
      .is_signed = false,
      .factor = 5,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_359[] = {
    {
      .name = "All_depart_2015",
      .b1 = 0,
      .b2 = 1,
      .bo = 63,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Right_Line_2017",
      .b1 = 24,
      .b2 = 1,
      .bo = 39,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Left_Line_2017",
      .b1 = 25,
      .b2 = 1,
      .bo = 38,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Sig1All_Depart",
      .b1 = 28,
      .b2 = 1,
      .bo = 35,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Sig2All_Depart",
      .b1 = 31,
      .b2 = 1,
      .bo = 32,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "LKAS_Inactive_2017",
      .b1 = 36,
      .b2 = 1,
      .bo = 27,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "LKAS_Active",
      .b1 = 37,
      .b2 = 1,
      .bo = 26,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Sig1Right_Depart",
      .b1 = 48,
      .b2 = 1,
      .bo = 15,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Sig1Right_Depart_Front",
      .b1 = 49,
      .b2 = 1,
      .bo = 14,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Sig2Right_Depart",
      .b1 = 50,
      .b2 = 1,
      .bo = 13,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Left_Depart_Front",
      .b1 = 51,
      .b2 = 1,
      .bo = 12,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Sig3All_Depart",
      .b1 = 52,
      .b2 = 1,
      .bo = 11,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_604[] = {
    {
      .name = "Counter",
      .b1 = 0,
      .b2 = 3,
      .bo = 61,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "State",
      .b1 = 5,
      .b2 = 1,
      .bo = 58,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "R_ADJACENT",
      .b1 = 32,
      .b2 = 1,
      .bo = 31,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "L_ADJACENT",
      .b1 = 33,
      .b2 = 1,
      .bo = 30,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "R_APPROACHING",
      .b1 = 42,
      .b2 = 1,
      .bo = 21,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "L_APPROACHING",
      .b1 = 43,
      .b2 = 1,
      .bo = 20,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "R_RCTA",
      .b1 = 46,
      .b2 = 1,
      .bo = 17,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "L_RCTA",
      .b1 = 47,
      .b2 = 1,
      .bo = 16,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_642[] = {
    {
      .name = "Counter",
      .b1 = 12,
      .b2 = 4,
      .bo = 48,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "SEATBELT_FL",
      .b1 = 40,
      .b2 = 1,
      .bo = 23,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "LEFT_BLINKER",
      .b1 = 44,
      .b2 = 1,
      .bo = 19,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "RIGHT_BLINKER",
      .b1 = 45,
      .b2 = 1,
      .bo = 18,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_864[] = {
    {
      .name = "Oil_Temp",
      .b1 = 16,
      .b2 = 8,
      .bo = 40,
      .is_signed = false,
      .factor = 1,
      .offset = -40.0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Coolant_Temp",
      .b1 = 24,
      .b2 = 8,
      .bo = 32,
      .is_signed = false,
      .factor = 1,
      .offset = -40.0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Cruise_Activated",
      .b1 = 45,
      .b2 = 1,
      .bo = 18,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Saved_Speed",
      .b1 = 56,
      .b2 = 8,
      .bo = 0,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_880[] = {
    {
      .name = "Steering_Voltage_Flat",
      .b1 = 0,
      .b2 = 8,
      .bo = 56,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Steer_Torque_Sensor",
      .b1 = 29,
      .b2 = 11,
      .bo = 24,
      .is_signed = true,
      .factor = -1.0,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Counter",
      .b1 = 40,
      .b2 = 4,
      .bo = 20,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_881[] = {
    {
      .name = "Steering_Motor_Flat",
      .b1 = 0,
      .b2 = 10,
      .bo = 54,
      .is_signed = false,
      .factor = 32,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Steer_Torque_Output",
      .b1 = 16,
      .b2 = 11,
      .bo = 37,
      .is_signed = true,
      .factor = -32.0,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "LKA_Lockout",
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
      .name = "Steer_Torque_Sensor",
      .b1 = 29,
      .b2 = 11,
      .bo = 24,
      .is_signed = true,
      .factor = -1.0,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Steering_Angle",
      .b1 = 40,
      .b2 = 16,
      .bo = 8,
      .is_signed = true,
      .factor = -0.033,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_884[] = {
    {
      .name = "DOOR_OPEN_FR",
      .b1 = 24,
      .b2 = 1,
      .bo = 39,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "DOOR_OPEN_FL",
      .b1 = 25,
      .b2 = 1,
      .bo = 38,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "DOOR_OPEN_RL",
      .b1 = 26,
      .b2 = 1,
      .bo = 37,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "DOOR_OPEN_RR",
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
      .name = "DOOR_OPEN_Hatch",
      .b1 = 28,
      .b2 = 1,
      .bo = 35,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_1745[] = {
    {
      .name = "Units",
      .b1 = 15,
      .b2 = 1,
      .bo = 48,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = true,
      .type = SignalType::DEFAULT,
    },
};

const Msg msgs[] = {
  {
    .name = "Steering",
    .address = 0x2,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_2),
    .sigs = sigs_2,
  },
  {
    .name = "G_Sensor",
    .address = 0xD0,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_208),
    .sigs = sigs_208,
  },
  {
    .name = "Brake_Pedal",
    .address = 0xD1,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_209),
    .sigs = sigs_209,
  },
  {
    .name = "Brake_2",
    .address = 0xD2,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_210),
    .sigs = sigs_210,
  },
  {
    .name = "Brake_Type",
    .address = 0xD3,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_211),
    .sigs = sigs_211,
  },
  {
    .name = "Wheel_Speeds",
    .address = 0xD4,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_212),
    .sigs = sigs_212,
  },
  {
    .name = "Throttle",
    .address = 0x140,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_320),
    .sigs = sigs_320,
  },
  {
    .name = "Engine",
    .address = 0x141,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_321),
    .sigs = sigs_321,
  },
  {
    .name = "CruiseControl",
    .address = 0x144,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_324),
    .sigs = sigs_324,
  },
  {
    .name = "Transmission",
    .address = 0x148,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_328),
    .sigs = sigs_328,
  },
  {
    .name = "Brake_Pressure",
    .address = 0x150,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_336),
    .sigs = sigs_336,
  },
  {
    .name = "Stalk",
    .address = 0x152,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_338),
    .sigs = sigs_338,
  },
  {
    .name = "ES_Brake",
    .address = 0x160,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_352),
    .sigs = sigs_352,
  },
  {
    .name = "ES_CruiseThrottle",
    .address = 0x161,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_353),
    .sigs = sigs_353,
  },
  {
    .name = "ES_RPM",
    .address = 0x162,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_354),
    .sigs = sigs_354,
  },
  {
    .name = "ES_LKAS",
    .address = 0x164,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_356),
    .sigs = sigs_356,
  },
  {
    .name = "ES_DashStatus",
    .address = 0x166,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_358),
    .sigs = sigs_358,
  },
  {
    .name = "ES_LDW",
    .address = 0x167,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_359),
    .sigs = sigs_359,
  },
  {
    .name = "BSD_RCTA",
    .address = 0x25C,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_604),
    .sigs = sigs_604,
  },
  {
    .name = "Dashlights",
    .address = 0x282,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_642),
    .sigs = sigs_642,
  },
  {
    .name = "Engine_Temp",
    .address = 0x360,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_864),
    .sigs = sigs_864,
  },
  {
    .name = "Steering_Torque_2",
    .address = 0x370,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_880),
    .sigs = sigs_880,
  },
  {
    .name = "Steering_Torque",
    .address = 0x371,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_881),
    .sigs = sigs_881,
  },
  {
    .name = "BodyInfo",
    .address = 0x374,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_884),
    .sigs = sigs_884,
  },
  {
    .name = "Dash_State",
    .address = 0x6D1,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_1745),
    .sigs = sigs_1745,
  },
};

const Val vals[] = {
    {
      .name = "Gear",
      .address = 0x148,
      .def_val = "0 N 1 D 2 D 3 D 4 D 5 D 6 D 14 R 15 P",
      .sigs = sigs_328,
    },
    {
      .name = "Units",
      .address = 0x6D1,
      .def_val = "0 METRIC 1 IMPERIAL",
      .sigs = sigs_1745,
    },
};

}

const DBC subaru_outback_2019_generated = {
  .name = "subaru_outback_2019_generated",
  .num_msgs = ARRAYSIZE(msgs),
  .msgs = msgs,
  .vals = vals,
  .num_vals = ARRAYSIZE(vals),
};

dbc_init(subaru_outback_2019_generated)