#include "common_dbc.h"

namespace {

const Signal sigs_189[] = {
    {
      .name = "RegenPaddle",
      .b1 = 0,
      .b2 = 4,
      .bo = 60,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_190[] = {
    {
      .name = "BrakePedalPos",
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
      .name = "GasPedalAndAcc",
      .b1 = 16,
      .b2 = 8,
      .bo = 40,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_201[] = {
    {
      .name = "Standstill",
      .b1 = 5,
      .b2 = 1,
      .bo = 58,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "EngineRPM",
      .b1 = 10,
      .b2 = 14,
      .bo = 40,
      .is_signed = false,
      .factor = 0.25,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "CruiseMainOn",
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
      .name = "EngineTPS",
      .b1 = 32,
      .b2 = 8,
      .bo = 24,
      .is_signed = false,
      .factor = 0.392156863,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Brake_Pressed",
      .b1 = 47,
      .b2 = 1,
      .bo = 16,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_209[] = {
    {
      .name = "BrakePedalTorque",
      .b1 = 4,
      .b2 = 12,
      .bo = 48,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_241[] = {
    {
      .name = "BrakePedalPosition",
      .b1 = 8,
      .b2 = 8,
      .bo = 48,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_298[] = {
    {
      .name = "RearLeftDoor",
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
      .name = "FrontLeftDoor",
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
      .name = "FrontRightDoor",
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
      .name = "LeftSeatBelt",
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
      .name = "RearRightDoor",
      .b1 = 16,
      .b2 = 1,
      .bo = 47,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "RightSeatBelt",
      .b1 = 50,
      .b2 = 1,
      .bo = 13,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_309[] = {
    {
      .name = "PRNDL",
      .b1 = 5,
      .b2 = 3,
      .bo = 56,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "ESPButton",
      .b1 = 3,
      .b2 = 1,
      .bo = 60,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_320[] = {
    {
      .name = "HighBeamsTemporary",
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
      .name = "HighBeamsActive",
      .b1 = 0,
      .b2 = 1,
      .bo = 63,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Intellibeam",
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
      .name = "TurnSignals",
      .b1 = 20,
      .b2 = 2,
      .bo = 42,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_381[] = {
    {
      .name = "TractionControlOn",
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
      .name = "MSG17D_AccPower",
      .b1 = 36,
      .b2 = 12,
      .bo = 16,
      .is_signed = true,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_384[] = {
    {
      .name = "LKASteeringCmd",
      .b1 = 5,
      .b2 = 11,
      .bo = 48,
      .is_signed = true,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "LKASteeringCmdActive",
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
      .name = "RollingCounter",
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
      .name = "LKASteeringCmdChecksum",
      .b1 = 20,
      .b2 = 12,
      .bo = 32,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_388[] = {
    {
      .name = "LKATotalTorqueDelivered",
      .b1 = 5,
      .b2 = 11,
      .bo = 48,
      .is_signed = true,
      .factor = 0.01,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "LKATorqueDeliveredStatus",
      .b1 = 2,
      .b2 = 3,
      .bo = 59,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "LKATorqueDelivered",
      .b1 = 21,
      .b2 = 11,
      .bo = 32,
      .is_signed = true,
      .factor = 0.01,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "HandsOffSWDetectionMode",
      .b1 = 19,
      .b2 = 2,
      .bo = 43,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "HandsOffSWlDetectionStatus",
      .b1 = 18,
      .b2 = 1,
      .bo = 45,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "LKADriverAppldTrq",
      .b1 = 53,
      .b2 = 11,
      .bo = 0,
      .is_signed = true,
      .factor = 0.01,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_417[] = {
    {
      .name = "AcceleratorPedal",
      .b1 = 48,
      .b2 = 8,
      .bo = 8,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_451[] = {
    {
      .name = "GasPedalAndAcc2",
      .b1 = 48,
      .b2 = 8,
      .bo = 8,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_452[] = {
    {
      .name = "CruiseState",
      .b1 = 8,
      .b2 = 3,
      .bo = 53,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "AcceleratorPedal2",
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
const Signal sigs_481[] = {
    {
      .name = "DistanceButton",
      .b1 = 17,
      .b2 = 1,
      .bo = 46,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "LKAButton",
      .b1 = 16,
      .b2 = 1,
      .bo = 47,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "DriveModeButton",
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
      .name = "ACCButtons",
      .b1 = 41,
      .b2 = 3,
      .bo = 20,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_485[] = {
    {
      .name = "SteeringWheelAngle",
      .b1 = 8,
      .b2 = 16,
      .bo = 40,
      .is_signed = true,
      .factor = 0.0625,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "SteeringWheelRate",
      .b1 = 28,
      .b2 = 12,
      .bo = 24,
      .is_signed = true,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_489[] = {
    {
      .name = "LateralAcceleration",
      .b1 = 4,
      .b2 = 10,
      .bo = 50,
      .is_signed = true,
      .factor = 0.161,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "BrakePedalPressed",
      .b1 = 1,
      .b2 = 1,
      .bo = 62,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "YawRate",
      .b1 = 36,
      .b2 = 12,
      .bo = 16,
      .is_signed = true,
      .factor = 0.625,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "YawRate2",
      .b1 = 52,
      .b2 = 12,
      .bo = 0,
      .is_signed = true,
      .factor = 0.0625,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_560[] = {
    {
      .name = "EPBClosed",
      .b1 = 11,
      .b2 = 1,
      .bo = 52,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_711[] = {
    {
      .name = "HVBatteryCurrent",
      .b1 = 11,
      .b2 = 13,
      .bo = 40,
      .is_signed = true,
      .factor = 0.15,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "HVBatteryVoltage",
      .b1 = 24,
      .b2 = 12,
      .bo = 28,
      .is_signed = false,
      .factor = 0.125,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_715[] = {
    {
      .name = "GasRegenCmdActive",
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
      .name = "RollingCounter",
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
      .name = "GasRegenAlwaysOne2",
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
      .name = "GasRegenFullStopActive",
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
      .name = "GasRegenAlwaysOne",
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
      .name = "GasRegenCmd",
      .b1 = 17,
      .b2 = 12,
      .bo = 35,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "GasRegenAlwaysOne3",
      .b1 = 16,
      .b2 = 1,
      .bo = 47,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "GasRegenCmdActiveInv",
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
      .name = "GasRegenChecksum",
      .b1 = 40,
      .b2 = 24,
      .bo = 0,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_810[] = {
    {
      .name = "GPSLatitude",
      .b1 = 0,
      .b2 = 32,
      .bo = 32,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "GPSLongitude",
      .b1 = 32,
      .b2 = 32,
      .bo = 0,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_840[] = {
    {
      .name = "FLWheelSpd",
      .b1 = 0,
      .b2 = 16,
      .bo = 48,
      .is_signed = false,
      .factor = 0.0311,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "FRWheelSpd",
      .b1 = 16,
      .b2 = 16,
      .bo = 32,
      .is_signed = false,
      .factor = 0.0311,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_842[] = {
    {
      .name = "RLWheelSpd",
      .b1 = 0,
      .b2 = 16,
      .bo = 48,
      .is_signed = false,
      .factor = 0.0311,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "RRWheelSpd",
      .b1 = 16,
      .b2 = 16,
      .bo = 32,
      .is_signed = false,
      .factor = 0.0311,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_880[] = {
    {
      .name = "ACCAlwaysOne",
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
      .name = "ACCResumeButton",
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
      .name = "ACCSpeedSetpoint",
      .b1 = 20,
      .b2 = 12,
      .bo = 32,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "ACCGapLevel",
      .b1 = 18,
      .b2 = 2,
      .bo = 44,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "ACCCmdActive",
      .b1 = 16,
      .b2 = 1,
      .bo = 47,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "ACCAlwaysOne2",
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
      .name = "FCWAlert",
      .b1 = 46,
      .b2 = 2,
      .bo = 16,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "ACCLeadCar",
      .b1 = 43,
      .b2 = 1,
      .bo = 20,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_1001[] = {
    {
      .name = "VehicleSpeed",
      .b1 = 0,
      .b2 = 16,
      .bo = 48,
      .is_signed = false,
      .factor = 0.01,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_1033[] = {
    {
      .name = "ASCMKeepAliveAllZero",
      .b1 = 0,
      .b2 = 56,
      .bo = 8,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_1217[] = {
    {
      .name = "EngineCoolantTemp",
      .b1 = 16,
      .b2 = 8,
      .bo = 40,
      .is_signed = false,
      .factor = 1,
      .offset = -40.0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_1249[] = {
    {
      .name = "VINPart2",
      .b1 = 0,
      .b2 = 64,
      .bo = 0,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_1300[] = {
    {
      .name = "VINPart1",
      .b1 = 0,
      .b2 = 64,
      .bo = 0,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};

const Msg msgs[] = {
  {
    .name = "EBCMRegenPaddle",
    .address = 0xBD,
    .size = 7,
    .num_sigs = ARRAYSIZE(sigs_189),
    .sigs = sigs_189,
  },
  {
    .name = "ECMAcceleratorPos",
    .address = 0xBE,
    .size = 6,
    .num_sigs = ARRAYSIZE(sigs_190),
    .sigs = sigs_190,
  },
  {
    .name = "ECMEngineStatus",
    .address = 0xC9,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_201),
    .sigs = sigs_201,
  },
  {
    .name = "EBCMBrakePedalTorque",
    .address = 0xD1,
    .size = 7,
    .num_sigs = ARRAYSIZE(sigs_209),
    .sigs = sigs_209,
  },
  {
    .name = "EBCMBrakePedalPosition",
    .address = 0xF1,
    .size = 6,
    .num_sigs = ARRAYSIZE(sigs_241),
    .sigs = sigs_241,
  },
  {
    .name = "BCMDoorBeltStatus",
    .address = 0x12A,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_298),
    .sigs = sigs_298,
  },
  {
    .name = "ECMPRDNL",
    .address = 0x135,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_309),
    .sigs = sigs_309,
  },
  {
    .name = "BCMTurnSignals",
    .address = 0x140,
    .size = 3,
    .num_sigs = ARRAYSIZE(sigs_320),
    .sigs = sigs_320,
  },
  {
    .name = "ESPStatus",
    .address = 0x17D,
    .size = 6,
    .num_sigs = ARRAYSIZE(sigs_381),
    .sigs = sigs_381,
  },
  {
    .name = "ASCMLKASteeringCmd",
    .address = 0x180,
    .size = 4,
    .num_sigs = ARRAYSIZE(sigs_384),
    .sigs = sigs_384,
  },
  {
    .name = "PSCMStatus",
    .address = 0x184,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_388),
    .sigs = sigs_388,
  },
  {
    .name = "AcceleratorPedal",
    .address = 0x1A1,
    .size = 7,
    .num_sigs = ARRAYSIZE(sigs_417),
    .sigs = sigs_417,
  },
  {
    .name = "GasAndAcc",
    .address = 0x1C3,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_451),
    .sigs = sigs_451,
  },
  {
    .name = "AcceleratorPedal2",
    .address = 0x1C4,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_452),
    .sigs = sigs_452,
  },
  {
    .name = "ASCMSteeringButton",
    .address = 0x1E1,
    .size = 7,
    .num_sigs = ARRAYSIZE(sigs_481),
    .sigs = sigs_481,
  },
  {
    .name = "PSCMSteeringAngle",
    .address = 0x1E5,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_485),
    .sigs = sigs_485,
  },
  {
    .name = "EBCMVehicleDynamic",
    .address = 0x1E9,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_489),
    .sigs = sigs_489,
  },
  {
    .name = "EPBStatus",
    .address = 0x230,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_560),
    .sigs = sigs_560,
  },
  {
    .name = "BECMBatteryVoltageCurrent",
    .address = 0x2C7,
    .size = 6,
    .num_sigs = ARRAYSIZE(sigs_711),
    .sigs = sigs_711,
  },
  {
    .name = "ASCMGasRegenCmd",
    .address = 0x2CB,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_715),
    .sigs = sigs_715,
  },
  {
    .name = "TCICOnStarGPSPosition",
    .address = 0x32A,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_810),
    .sigs = sigs_810,
  },
  {
    .name = "EBCMWheelSpdFront",
    .address = 0x348,
    .size = 4,
    .num_sigs = ARRAYSIZE(sigs_840),
    .sigs = sigs_840,
  },
  {
    .name = "EBCMWheelSpdRear",
    .address = 0x34A,
    .size = 5,
    .num_sigs = ARRAYSIZE(sigs_842),
    .sigs = sigs_842,
  },
  {
    .name = "ASCMActiveCruiseControlStatus",
    .address = 0x370,
    .size = 6,
    .num_sigs = ARRAYSIZE(sigs_880),
    .sigs = sigs_880,
  },
  {
    .name = "ECMVehicleSpeed",
    .address = 0x3E9,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_1001),
    .sigs = sigs_1001,
  },
  {
    .name = "ASCMKeepAlive",
    .address = 0x409,
    .size = 7,
    .num_sigs = ARRAYSIZE(sigs_1033),
    .sigs = sigs_1033,
  },
  {
    .name = "ECMEngineCoolantTemp",
    .address = 0x4C1,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_1217),
    .sigs = sigs_1217,
  },
  {
    .name = "VIN_Part2",
    .address = 0x4E1,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_1249),
    .sigs = sigs_1249,
  },
  {
    .name = "VIN_Part1",
    .address = 0x514,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_1300),
    .sigs = sigs_1300,
  },
};

const Val vals[] = {
    {
      .name = "ESPButton",
      .address = 0x135,
      .def_val = "1 ACTIVE 0 INACTIVE",
      .sigs = sigs_309,
    },
    {
      .name = "PRNDL",
      .address = 0x135,
      .def_val = "3 R 2 D 1 N 0 P",
      .sigs = sigs_309,
    },
    {
      .name = "HighBeamsActive",
      .address = 0x140,
      .def_val = "1 ACTIVE 0 INACTIVE",
      .sigs = sigs_320,
    },
    {
      .name = "HighBeamsTemporary",
      .address = 0x140,
      .def_val = "1 ACTIVE 0 INACTIVE",
      .sigs = sigs_320,
    },
    {
      .name = "Intellibeam",
      .address = 0x140,
      .def_val = "1 ACTIVE 0 INACTIVE",
      .sigs = sigs_320,
    },
    {
      .name = "LKASteeringCmdActive",
      .address = 0x180,
      .def_val = "1 ACTIVE 0 INACTIVE",
      .sigs = sigs_384,
    },
    {
      .name = "HandsOffSWDetectionMode",
      .address = 0x184,
      .def_val = "2 FAILED 1 ENABLED 0 DISABLED",
      .sigs = sigs_388,
    },
    {
      .name = "HandsOffSWlDetectionStatus",
      .address = 0x184,
      .def_val = "1 HANDS_ON 0 HANDS_OFF",
      .sigs = sigs_388,
    },
    {
      .name = "LKATorqueDeliveredStatus",
      .address = 0x184,
      .def_val = "3 FAILED 2 TEMP._LIMITED 1 ACTIVE 0 INACTIVE",
      .sigs = sigs_388,
    },
    {
      .name = "CruiseState",
      .address = 0x1C4,
      .def_val = "4 STANDSTILL 3 FAULTED 1 ACTIVE 0 OFF",
      .sigs = sigs_452,
    },
    {
      .name = "ACCButtons",
      .address = 0x1E1,
      .def_val = "6 CANCEL 5 MAIN 3 SET 2 RESUME 1 NONE",
      .sigs = sigs_481,
    },
    {
      .name = "DistanceButton",
      .address = 0x1E1,
      .def_val = "1 ACTIVE 0 INACTIVE",
      .sigs = sigs_481,
    },
    {
      .name = "DriveModeButton",
      .address = 0x1E1,
      .def_val = "1 ACTIVE 0 INACTIVE",
      .sigs = sigs_481,
    },
    {
      .name = "LKAButton",
      .address = 0x1E1,
      .def_val = "1 ACTIVE 0 INACTIVE",
      .sigs = sigs_481,
    },
    {
      .name = "BrakePedalPressed",
      .address = 0x1E9,
      .def_val = "1 PRESSED 0 DEPRESSED",
      .sigs = sigs_489,
    },
    {
      .name = "GasRegenCmdActive",
      .address = 0x2CB,
      .def_val = "1 ACTIVE 0 INACTIVE",
      .sigs = sigs_715,
    },
    {
      .name = "GasRegenCmdActiveInv",
      .address = 0x2CB,
      .def_val = "1 INACTIVE 0 ACTIVE",
      .sigs = sigs_715,
    },
    {
      .name = "ACCCmdActive",
      .address = 0x370,
      .def_val = "1 ACTIVE 0 INACTIVE",
      .sigs = sigs_880,
    },
    {
      .name = "ACCGapLevel",
      .address = 0x370,
      .def_val = "3 FAR 2 MED 1 NEAR 0 INACTIVE",
      .sigs = sigs_880,
    },
    {
      .name = "ACCLeadCar",
      .address = 0x370,
      .def_val = "1 PRESENT 0 NOT_PRESENT",
      .sigs = sigs_880,
    },
    {
      .name = "ACCResumeButton",
      .address = 0x370,
      .def_val = "1 PRESSED 0 DEPRESSED",
      .sigs = sigs_880,
    },
};

}

const DBC gm_global_a_powertrain = {
  .name = "gm_global_a_powertrain",
  .num_msgs = ARRAYSIZE(msgs),
  .msgs = msgs,
  .vals = vals,
  .num_vals = ARRAYSIZE(vals),
};

dbc_init(gm_global_a_powertrain)