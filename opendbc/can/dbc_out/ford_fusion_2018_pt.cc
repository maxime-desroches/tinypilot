#include "common_dbc.h"

namespace {

const Signal sigs_118[] = {
    {
      .name = "SteWhlRelInit_An_Sns",
      .b1 = 1,
      .b2 = 15,
      .bo = 48,
      .is_signed = false,
      .factor = 0.1,
      .offset = -1600.0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "SteWhlRelCalib_An_Sns",
      .b1 = 16,
      .b2 = 15,
      .bo = 33,
      .is_signed = false,
      .factor = 0.1,
      .offset = -1600.0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "SteWhlAn_No_Cs",
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
      .name = "SteWhlAn_No_Cnt",
      .b1 = 40,
      .b2 = 4,
      .bo = 20,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "SteWhlRelInit2_An_Sns",
      .b1 = 48,
      .b2 = 16,
      .bo = 0,
      .is_signed = false,
      .factor = 0.1,
      .offset = -3200.0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_130[] = {
    {
      .name = "SteeringColumnTorque",
      .b1 = 0,
      .b2 = 8,
      .bo = 56,
      .is_signed = false,
      .factor = 0.0625,
      .offset = -8.0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "EPAS_FAILURE",
      .b1 = 14,
      .b2 = 2,
      .bo = 48,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "SAPPAngleControlStat2",
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
      .name = "SAPPAngleControlStat3",
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
      .name = "SAPPAngleControlStat4",
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
      .name = "SAPPAngleControlStat5",
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
      .name = "SAPPAngleControlStat6",
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
      .name = "SteMdule_I_Est",
      .b1 = 18,
      .b2 = 12,
      .bo = 34,
      .is_signed = false,
      .factor = 0.05,
      .offset = -64.0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "SAPPAngleControlStat1",
      .b1 = 16,
      .b2 = 2,
      .bo = 46,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "SteMdule_U_Meas",
      .b1 = 32,
      .b2 = 8,
      .bo = 24,
      .is_signed = false,
      .factor = 0.05,
      .offset = 6.0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_131[] = {
    {
      .name = "Left_Turn_Light",
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
      .name = "Right_Turn_Light",
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
      .name = "Cancel",
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
      .name = "Dist_Incr",
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
      .name = "Dist_Decr",
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
      .name = "Set",
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
      .name = "Resume",
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
      .name = "Main",
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
const Signal sigs_145[] = {
    {
      .name = "VehPtch_W_Actl",
      .b1 = 0,
      .b2 = 16,
      .bo = 48,
      .is_signed = false,
      .factor = 0.0002,
      .offset = -6.5,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "VehRol_W_Actl",
      .b1 = 16,
      .b2 = 16,
      .bo = 32,
      .is_signed = false,
      .factor = 0.0002,
      .offset = -6.5,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "VehYaw_W_Actl",
      .b1 = 32,
      .b2 = 16,
      .bo = 16,
      .is_signed = false,
      .factor = 0.0002,
      .offset = -6.5,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_146[] = {
    {
      .name = "VehLat_A_Actl",
      .b1 = 3,
      .b2 = 13,
      .bo = 48,
      .is_signed = false,
      .factor = 0.01,
      .offset = -40.0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "VehLatAActl_D_Qf",
      .b1 = 1,
      .b2 = 2,
      .bo = 61,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "VehLong_A_Actl",
      .b1 = 19,
      .b2 = 13,
      .bo = 32,
      .is_signed = false,
      .factor = 0.01,
      .offset = -40.0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "VehLongAActl_D_Qf",
      .b1 = 17,
      .b2 = 2,
      .bo = 45,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "VehVert_A_Actl",
      .b1 = 35,
      .b2 = 13,
      .bo = 16,
      .is_signed = false,
      .factor = 0.01,
      .offset = -40.0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "VehVertAActl_D_Qf",
      .b1 = 33,
      .b2 = 2,
      .bo = 29,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_357[] = {
    {
      .name = "Brake_Drv_Appl",
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
      .name = "Cruise_State",
      .b1 = 12,
      .b2 = 4,
      .bo = 48,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Set_Speed",
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
const Signal sigs_516[] = {
    {
      .name = "ApedPosScal_Pc_Actl",
      .b1 = 6,
      .b2 = 10,
      .bo = 48,
      .is_signed = false,
      .factor = 0.1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_534[] = {
    {
      .name = "WhlRotatFr_No_Cnt",
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
      .name = "WhlRotatFl_No_Cnt",
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
      .name = "WhlRotatRr_No_Cnt",
      .b1 = 16,
      .b2 = 8,
      .bo = 40,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "WhlRotatRl_No_Cnt",
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
      .name = "WhlDirRr_D_Actl",
      .b1 = 38,
      .b2 = 2,
      .bo = 24,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "WhlDirFl_D_Actl",
      .b1 = 36,
      .b2 = 2,
      .bo = 26,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "WhlDirFr_D_Actl",
      .b1 = 34,
      .b2 = 2,
      .bo = 28,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "WhlDirRl_D_Actl",
      .b1 = 32,
      .b2 = 2,
      .bo = 30,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "WHEEL_ROLLING_TIMESTAMP",
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
const Signal sigs_535[] = {
    {
      .name = "WhlFl_W_Meas",
      .b1 = 0,
      .b2 = 14,
      .bo = 50,
      .is_signed = false,
      .factor = 0.04,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "WhlFr_W_Meas",
      .b1 = 16,
      .b2 = 14,
      .bo = 34,
      .is_signed = false,
      .factor = 0.04,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "WhlRl_W_Meas",
      .b1 = 32,
      .b2 = 14,
      .bo = 18,
      .is_signed = false,
      .factor = 0.04,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "WhlRr_W_Meas",
      .b1 = 48,
      .b2 = 14,
      .bo = 2,
      .is_signed = false,
      .factor = 0.04,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_947[] = {
    {
      .name = "Door_RL_Open",
      .b1 = 55,
      .b2 = 1,
      .bo = 8,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Door_RR_Open",
      .b1 = 54,
      .b2 = 1,
      .bo = 9,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Door_FR_Open",
      .b1 = 59,
      .b2 = 1,
      .bo = 4,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Door_FL_Open",
      .b1 = 58,
      .b2 = 1,
      .bo = 5,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_963[] = {
    {
      .name = "Brake_Lights",
      .b1 = 15,
      .b2 = 1,
      .bo = 48,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_970[] = {
    {
      .name = "Lkas_Alert",
      .b1 = 4,
      .b2 = 4,
      .bo = 56,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Lkas_Action",
      .b1 = 0,
      .b2 = 4,
      .bo = 60,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Lane_Curvature",
      .b1 = 8,
      .b2 = 12,
      .bo = 44,
      .is_signed = false,
      .factor = 5e-06,
      .offset = -0.01,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Steer_Angle_Req",
      .b1 = 20,
      .b2 = 12,
      .bo = 32,
      .is_signed = false,
      .factor = 0.04297,
      .offset = -88.00445,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_972[] = {
    {
      .name = "LaActAvail_D_Actl",
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
      .name = "LaActDeny_B_Actl",
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
      .name = "LaHandsOff_B_Actl",
      .b1 = 0,
      .b2 = 1,
      .bo = 63,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
};
const Signal sigs_984[] = {
    {
      .name = "Set_Me_X80",
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
      .name = "Set_Me_X45",
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
      .name = "Hands_Warning",
      .b1 = 54,
      .b2 = 1,
      .bo = 9,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Hands_Warning_W_Chime",
      .b1 = 53,
      .b2 = 1,
      .bo = 10,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Lines_Hud",
      .b1 = 48,
      .b2 = 4,
      .bo = 12,
      .is_signed = false,
      .factor = 1,
      .offset = 0,
      .is_little_endian = false,
      .type = SignalType::DEFAULT,
    },
    {
      .name = "Set_Me_X30",
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

const Msg msgs[] = {
  {
    .name = "Steering_Wheel_Data_CG1",
    .address = 0x76,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_118),
    .sigs = sigs_118,
  },
  {
    .name = "EPAS_INFO",
    .address = 0x82,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_130),
    .sigs = sigs_130,
  },
  {
    .name = "Steering_Buttons",
    .address = 0x83,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_131),
    .sigs = sigs_131,
  },
  {
    .name = "Yaw_Data",
    .address = 0x91,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_145),
    .sigs = sigs_145,
  },
  {
    .name = "Accel_Data",
    .address = 0x92,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_146),
    .sigs = sigs_146,
  },
  {
    .name = "Cruise_Status",
    .address = 0x165,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_357),
    .sigs = sigs_357,
  },
  {
    .name = "EngineData_14",
    .address = 0x204,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_516),
    .sigs = sigs_516,
  },
  {
    .name = "WheelData",
    .address = 0x216,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_534),
    .sigs = sigs_534,
  },
  {
    .name = "WheelSpeed_CG1",
    .address = 0x217,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_535),
    .sigs = sigs_535,
  },
  {
    .name = "Doors",
    .address = 0x3B3,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_947),
    .sigs = sigs_947,
  },
  {
    .name = "BCM_to_HS_Body",
    .address = 0x3C3,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_963),
    .sigs = sigs_963,
  },
  {
    .name = "Lane_Keep_Assist_Control",
    .address = 0x3CA,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_970),
    .sigs = sigs_970,
  },
  {
    .name = "Lane_Keep_Assist_Status",
    .address = 0x3CC,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_972),
    .sigs = sigs_972,
  },
  {
    .name = "Lane_Keep_Assist_Ui",
    .address = 0x3D8,
    .size = 8,
    .num_sigs = ARRAYSIZE(sigs_984),
    .sigs = sigs_984,
  },
};

const Val vals[] = {
    {
      .name = "Cruise_State",
      .address = 0x165,
      .def_val = "4 ACTIVE 3 STANDBY 0 OFF",
      .sigs = sigs_357,
    },
    {
      .name = "Lkas_Action",
      .address = 0x3CA,
      .def_val = "15 OFF 9 ABRUPT 8 ABRUPT2 5 SMOOTH 4 SMOOTH2",
      .sigs = sigs_970,
    },
    {
      .name = "Lkas_Alert",
      .address = 0x3CA,
      .def_val = "15 NO_ALERT 3 HIGH_INTENSITY 2 MID_INTENSITY 1 LOW_INTENSITY",
      .sigs = sigs_970,
    },
    {
      .name = "LaActAvail_D_Actl",
      .address = 0x3CC,
      .def_val = "3 AVAILABLE 2 TBD 1 NOT_AVAILABLE 0 FAULT",
      .sigs = sigs_972,
    },
    {
      .name = "Lines_Hud",
      .address = 0x3D8,
      .def_val = "15 NONE 11 GREY_YELLOW 8 GREEN_RED 7 YELLOW_GREY 6 GREY_GREY 4 RED_GREEN 3 GREEN_GREEN",
      .sigs = sigs_984,
    },
};

}

const DBC ford_fusion_2018_pt = {
  .name = "ford_fusion_2018_pt",
  .num_msgs = ARRAYSIZE(msgs),
  .msgs = msgs,
  .vals = vals,
  .num_vals = ARRAYSIZE(vals),
};

dbc_init(ford_fusion_2018_pt)