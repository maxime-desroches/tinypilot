struct i2c_random_wr_payload start_reg_array[] = {{0x301a, 0x91c}};
//struct i2c_random_wr_payload stop_reg_array[] = {{0x301a, 0x10d8}};
struct i2c_random_wr_payload stop_reg_array[] = {{0x301a, 0x918}};;

struct i2c_random_wr_payload init_array_ar0231[] = {
  {0x301A, 0x0018}, // RESET_REGISTER
  //
  {0x3092, 0x0C24}, // ROW_NOISE_CONTROL
  {0x337A, 0x0C80}, // DBLC_SCALE0
  {0x3520, 0x1288}, // RESERVED_MFR_3520
  {0x3522, 0x880C}, // RESERVED_MFR_3522
  {0x3524, 0x0C12}, // RESERVED_MFR_3524
  {0x352C, 0x1212}, // RESERVED_MFR_352C
  {0x354A, 0x007F}, // RESERVED_MFR_354A
  {0x350C, 0x055C}, // RESERVED_MFR_350C
  {0x3506, 0x3333}, // RESERVED_MFR_3506
  {0x3508, 0x3333}, // RESERVED_MFR_3508
  {0x3100, 0x4000}, // DLO_CONTROL0
  {0x3280, 0x0CCC}, // RESERVED_MFR_3280
  {0x3282, 0x0CCC}, // RESERVED_MFR_3282
  {0x3284, 0x0CCC}, // RESERVED_MFR_3284
  {0x3286, 0x0CCC}, // RESERVED_MFR_3286
  {0x3288, 0x0FA0}, // RESERVED_MFR_3288
  {0x328A, 0x0FA0}, // RESERVED_MFR_328A
  {0x328C, 0x0FA0}, // RESERVED_MFR_328C
  {0x328E, 0x0FA0}, // RESERVED_MFR_328E
  {0x3290, 0x0FA0}, // RESERVED_MFR_3290
  {0x3292, 0x0FA0}, // RESERVED_MFR_3292
  {0x3294, 0x0FA0}, // RESERVED_MFR_3294
  {0x3296, 0x0FA0}, // RESERVED_MFR_3296
  {0x3298, 0x0FA0}, // RESERVED_MFR_3298
  {0x329A, 0x0FA0}, // RESERVED_MFR_329A
  {0x329C, 0x0FA0}, // RESERVED_MFR_329C
  {0x329E, 0x0FA0}, // RESERVED_MFR_329E
  {0x2512, 0x8000}, // SEQ_CTRL_PORT
  {0x2510, 0x0905}, // SEQ_DATA_PORT
  {0x2510, 0x3350}, // SEQ_DATA_PORT
  {0x2510, 0x2004}, // SEQ_DATA_PORT
  {0x2510, 0x1460}, // SEQ_DATA_PORT
  {0x2510, 0x1578}, // SEQ_DATA_PORT
  {0x2510, 0x0901}, // SEQ_DATA_PORT
  {0x2510, 0x7B24}, // SEQ_DATA_PORT
  {0x2510, 0xFF24}, // SEQ_DATA_PORT
  {0x2510, 0xFF24}, // SEQ_DATA_PORT
  {0x2510, 0xEA24}, // SEQ_DATA_PORT
  {0x2510, 0x1022}, // SEQ_DATA_PORT
  {0x2510, 0x2410}, // SEQ_DATA_PORT
  {0x2510, 0x155A}, // SEQ_DATA_PORT
  {0x2510, 0x0901}, // SEQ_DATA_PORT
  {0x2510, 0x1400}, // SEQ_DATA_PORT
  {0x2510, 0x24FF}, // SEQ_DATA_PORT
  {0x2510, 0x24FF}, // SEQ_DATA_PORT
  {0x2510, 0x24EA}, // SEQ_DATA_PORT
  {0x2510, 0x2324}, // SEQ_DATA_PORT
  {0x2510, 0x647A}, // SEQ_DATA_PORT
  {0x2510, 0x2404}, // SEQ_DATA_PORT
  {0x2510, 0x052C}, // SEQ_DATA_PORT
  {0x2510, 0x400A}, // SEQ_DATA_PORT
  {0x2510, 0xFF0A}, // SEQ_DATA_PORT
  {0x2510, 0xFF0A}, // SEQ_DATA_PORT
  {0x2510, 0x1008}, // SEQ_DATA_PORT
  {0x2510, 0x3851}, // SEQ_DATA_PORT
  {0x2510, 0x1440}, // SEQ_DATA_PORT
  {0x2510, 0x0004}, // SEQ_DATA_PORT
  {0x2510, 0x0801}, // SEQ_DATA_PORT
  {0x2510, 0x0408}, // SEQ_DATA_PORT
  {0x2510, 0x1180}, // SEQ_DATA_PORT
  {0x2510, 0x2652}, // SEQ_DATA_PORT
  {0x2510, 0x1518}, // SEQ_DATA_PORT
  {0x2510, 0x0906}, // SEQ_DATA_PORT
  {0x2510, 0x1348}, // SEQ_DATA_PORT
  {0x2510, 0x1002}, // SEQ_DATA_PORT
  {0x2510, 0x1016}, // SEQ_DATA_PORT
  {0x2510, 0x1181}, // SEQ_DATA_PORT
  {0x2510, 0x1189}, // SEQ_DATA_PORT
  {0x2510, 0x1056}, // SEQ_DATA_PORT
  {0x2510, 0x1210}, // SEQ_DATA_PORT
  {0x2510, 0x0901}, // SEQ_DATA_PORT
  {0x2510, 0x0D09}, // SEQ_DATA_PORT
  {0x2510, 0x1413}, // SEQ_DATA_PORT
  {0x2510, 0x8809}, // SEQ_DATA_PORT
  {0x2510, 0x2B15}, // SEQ_DATA_PORT
  {0x2510, 0x8809}, // SEQ_DATA_PORT
  {0x2510, 0x0311}, // SEQ_DATA_PORT
  {0x2510, 0xD909}, // SEQ_DATA_PORT
  {0x2510, 0x1214}, // SEQ_DATA_PORT
  {0x2510, 0x4109}, // SEQ_DATA_PORT
  {0x2510, 0x0312}, // SEQ_DATA_PORT
  {0x2510, 0x1409}, // SEQ_DATA_PORT
  {0x2510, 0x0110}, // SEQ_DATA_PORT
  {0x2510, 0xD612}, // SEQ_DATA_PORT
  {0x2510, 0x1012}, // SEQ_DATA_PORT
  {0x2510, 0x1212}, // SEQ_DATA_PORT
  {0x2510, 0x1011}, // SEQ_DATA_PORT
  {0x2510, 0xDD11}, // SEQ_DATA_PORT
  {0x2510, 0xD910}, // SEQ_DATA_PORT
  {0x2510, 0x5609}, // SEQ_DATA_PORT
  {0x2510, 0x1511}, // SEQ_DATA_PORT
  {0x2510, 0xDB09}, // SEQ_DATA_PORT
  {0x2510, 0x1511}, // SEQ_DATA_PORT
  {0x2510, 0x9B09}, // SEQ_DATA_PORT
  {0x2510, 0x0F11}, // SEQ_DATA_PORT
  {0x2510, 0xBB12}, // SEQ_DATA_PORT
  {0x2510, 0x1A12}, // SEQ_DATA_PORT
  {0x2510, 0x1014}, // SEQ_DATA_PORT
  {0x2510, 0x6012}, // SEQ_DATA_PORT
  {0x2510, 0x5010}, // SEQ_DATA_PORT
  {0x2510, 0x7610}, // SEQ_DATA_PORT
  {0x2510, 0xE609}, // SEQ_DATA_PORT
  {0x2510, 0x0812}, // SEQ_DATA_PORT
  {0x2510, 0x4012}, // SEQ_DATA_PORT
  {0x2510, 0x6009}, // SEQ_DATA_PORT
  {0x2510, 0x290B}, // SEQ_DATA_PORT
  {0x2510, 0x0904}, // SEQ_DATA_PORT
  {0x2510, 0x1440}, // SEQ_DATA_PORT
  {0x2510, 0x0923}, // SEQ_DATA_PORT
  {0x2510, 0x15C8}, // SEQ_DATA_PORT
  {0x2510, 0x13C8}, // SEQ_DATA_PORT
  {0x2510, 0x092C}, // SEQ_DATA_PORT
  {0x2510, 0x1588}, // SEQ_DATA_PORT
  {0x2510, 0x1388}, // SEQ_DATA_PORT
  {0x2510, 0x0C09}, // SEQ_DATA_PORT
  {0x2510, 0x0C14}, // SEQ_DATA_PORT
  {0x2510, 0x4109}, // SEQ_DATA_PORT
  {0x2510, 0x1112}, // SEQ_DATA_PORT
  {0x2510, 0x6212}, // SEQ_DATA_PORT
  {0x2510, 0x6011}, // SEQ_DATA_PORT
  {0x2510, 0xBF11}, // SEQ_DATA_PORT
  {0x2510, 0xBB10}, // SEQ_DATA_PORT
  {0x2510, 0x6611}, // SEQ_DATA_PORT
  {0x2510, 0xFB09}, // SEQ_DATA_PORT
  {0x2510, 0x3511}, // SEQ_DATA_PORT
  {0x2510, 0xBB12}, // SEQ_DATA_PORT
  {0x2510, 0x6312}, // SEQ_DATA_PORT
  {0x2510, 0x6014}, // SEQ_DATA_PORT
  {0x2510, 0x0015}, // SEQ_DATA_PORT
  {0x2510, 0x0011}, // SEQ_DATA_PORT
  {0x2510, 0xB812}, // SEQ_DATA_PORT
  {0x2510, 0xA012}, // SEQ_DATA_PORT
  {0x2510, 0x0010}, // SEQ_DATA_PORT
  {0x2510, 0x2610}, // SEQ_DATA_PORT
  {0x2510, 0x0013}, // SEQ_DATA_PORT
  {0x2510, 0x0011}, // SEQ_DATA_PORT
  {0x2510, 0x0008}, // SEQ_DATA_PORT
  {0x2510, 0x3053}, // SEQ_DATA_PORT
  {0x2510, 0x4215}, // SEQ_DATA_PORT
  {0x2510, 0x4013}, // SEQ_DATA_PORT
  {0x2510, 0x4010}, // SEQ_DATA_PORT
  {0x2510, 0x0210}, // SEQ_DATA_PORT
  {0x2510, 0x1611}, // SEQ_DATA_PORT
  {0x2510, 0x8111}, // SEQ_DATA_PORT
  {0x2510, 0x8910}, // SEQ_DATA_PORT
  {0x2510, 0x5612}, // SEQ_DATA_PORT
  {0x2510, 0x1009}, // SEQ_DATA_PORT
  {0x2510, 0x010D}, // SEQ_DATA_PORT
  {0x2510, 0x0815}, // SEQ_DATA_PORT
  {0x2510, 0xC015}, // SEQ_DATA_PORT
  {0x2510, 0xD013}, // SEQ_DATA_PORT
  {0x2510, 0x5009}, // SEQ_DATA_PORT
  {0x2510, 0x1313}, // SEQ_DATA_PORT
  {0x2510, 0xD009}, // SEQ_DATA_PORT
  {0x2510, 0x0215}, // SEQ_DATA_PORT
  {0x2510, 0xC015}, // SEQ_DATA_PORT
  {0x2510, 0xC813}, // SEQ_DATA_PORT
  {0x2510, 0xC009}, // SEQ_DATA_PORT
  {0x2510, 0x0515}, // SEQ_DATA_PORT
  {0x2510, 0x8813}, // SEQ_DATA_PORT
  {0x2510, 0x8009}, // SEQ_DATA_PORT
  {0x2510, 0x0213}, // SEQ_DATA_PORT
  {0x2510, 0x8809}, // SEQ_DATA_PORT
  {0x2510, 0x0411}, // SEQ_DATA_PORT
  {0x2510, 0xC909}, // SEQ_DATA_PORT
  {0x2510, 0x0814}, // SEQ_DATA_PORT
  {0x2510, 0x0109}, // SEQ_DATA_PORT
  {0x2510, 0x0B11}, // SEQ_DATA_PORT
  {0x2510, 0xD908}, // SEQ_DATA_PORT
  {0x2510, 0x1400}, // SEQ_DATA_PORT
  {0x2510, 0x091A}, // SEQ_DATA_PORT
  {0x2510, 0x1440}, // SEQ_DATA_PORT
  {0x2510, 0x0903}, // SEQ_DATA_PORT
  {0x2510, 0x1214}, // SEQ_DATA_PORT
  {0x2510, 0x0901}, // SEQ_DATA_PORT
  {0x2510, 0x10D6}, // SEQ_DATA_PORT
  {0x2510, 0x1210}, // SEQ_DATA_PORT
  {0x2510, 0x1212}, // SEQ_DATA_PORT
  {0x2510, 0x1210}, // SEQ_DATA_PORT
  {0x2510, 0x11DD}, // SEQ_DATA_PORT
  {0x2510, 0x11D9}, // SEQ_DATA_PORT
  {0x2510, 0x1056}, // SEQ_DATA_PORT
  {0x2510, 0x0917}, // SEQ_DATA_PORT
  {0x2510, 0x11DB}, // SEQ_DATA_PORT
  {0x2510, 0x0913}, // SEQ_DATA_PORT
  {0x2510, 0x11FB}, // SEQ_DATA_PORT
  {0x2510, 0x0905}, // SEQ_DATA_PORT
  {0x2510, 0x11BB}, // SEQ_DATA_PORT
  {0x2510, 0x121A}, // SEQ_DATA_PORT
  {0x2510, 0x1210}, // SEQ_DATA_PORT
  {0x2510, 0x1460}, // SEQ_DATA_PORT
  {0x2510, 0x1250}, // SEQ_DATA_PORT
  {0x2510, 0x1076}, // SEQ_DATA_PORT
  {0x2510, 0x10E6}, // SEQ_DATA_PORT
  {0x2510, 0x0901}, // SEQ_DATA_PORT
  {0x2510, 0x15A8}, // SEQ_DATA_PORT
  {0x2510, 0x0901}, // SEQ_DATA_PORT
  {0x2510, 0x13A8}, // SEQ_DATA_PORT
  {0x2510, 0x1240}, // SEQ_DATA_PORT
  {0x2510, 0x1260}, // SEQ_DATA_PORT
  {0x2510, 0x0925}, // SEQ_DATA_PORT
  {0x2510, 0x13AD}, // SEQ_DATA_PORT
  {0x2510, 0x0902}, // SEQ_DATA_PORT
  {0x2510, 0x0907}, // SEQ_DATA_PORT
  {0x2510, 0x1588}, // SEQ_DATA_PORT
  {0x2510, 0x0901}, // SEQ_DATA_PORT
  {0x2510, 0x138D}, // SEQ_DATA_PORT
  {0x2510, 0x0B09}, // SEQ_DATA_PORT
  {0x2510, 0x0914}, // SEQ_DATA_PORT
  {0x2510, 0x4009}, // SEQ_DATA_PORT
  {0x2510, 0x0B13}, // SEQ_DATA_PORT
  {0x2510, 0x8809}, // SEQ_DATA_PORT
  {0x2510, 0x1C0C}, // SEQ_DATA_PORT
  {0x2510, 0x0920}, // SEQ_DATA_PORT
  {0x2510, 0x1262}, // SEQ_DATA_PORT
  {0x2510, 0x1260}, // SEQ_DATA_PORT
  {0x2510, 0x11BF}, // SEQ_DATA_PORT
  {0x2510, 0x11BB}, // SEQ_DATA_PORT
  {0x2510, 0x1066}, // SEQ_DATA_PORT
  {0x2510, 0x090A}, // SEQ_DATA_PORT
  {0x2510, 0x11FB}, // SEQ_DATA_PORT
  {0x2510, 0x093B}, // SEQ_DATA_PORT
  {0x2510, 0x11BB}, // SEQ_DATA_PORT
  {0x2510, 0x1263}, // SEQ_DATA_PORT
  {0x2510, 0x1260}, // SEQ_DATA_PORT
  {0x2510, 0x1400}, // SEQ_DATA_PORT
  {0x2510, 0x1508}, // SEQ_DATA_PORT
  {0x2510, 0x11B8}, // SEQ_DATA_PORT
  {0x2510, 0x12A0}, // SEQ_DATA_PORT
  {0x2510, 0x1200}, // SEQ_DATA_PORT
  {0x2510, 0x1026}, // SEQ_DATA_PORT
  {0x2510, 0x1000}, // SEQ_DATA_PORT
  {0x2510, 0x1300}, // SEQ_DATA_PORT
  {0x2510, 0x1100}, // SEQ_DATA_PORT
  {0x2510, 0x437A}, // SEQ_DATA_PORT
  {0x2510, 0x0609}, // SEQ_DATA_PORT
  {0x2510, 0x0B05}, // SEQ_DATA_PORT
  {0x2510, 0x0708}, // SEQ_DATA_PORT
  {0x2510, 0x4137}, // SEQ_DATA_PORT
  {0x2510, 0x502C}, // SEQ_DATA_PORT
  {0x2510, 0x2CFE}, // SEQ_DATA_PORT
  {0x2510, 0x15FE}, // SEQ_DATA_PORT
  {0x2510, 0x0C2C}, // SEQ_DATA_PORT
  {0x32E6, 0x00E0}, // RESERVED_MFR_32E6
  {0x1008, 0x036F}, // RESERVED_PARAM_1008
  {0x100C, 0x058F}, // RESERVED_PARAM_100C
  {0x100E, 0x07AF}, // RESERVED_PARAM_100E
  {0x1010, 0x014F}, // RESERVED_PARAM_1010
  {0x3230, 0x0312}, // FINE_CORRECTION
  {0x3232, 0x0532}, // FINE_CORRECTION2
  {0x3234, 0x0752}, // FINE_CORRECTION3
  {0x3236, 0x00F2}, // FINE_CORRECTION4
  {0x3566, 0x3328}, // RESERVED_MFR_3566
  {0x32D0, 0x3A02}, // RESERVED_MFR_32D0
  {0x32D2, 0x3508}, // RESERVED_MFR_32D2
  {0x32D4, 0x3702}, // RESERVED_MFR_32D4
  {0x32D6, 0x3C04}, // RESERVED_MFR_32D6
  {0x32DC, 0x370A}, // RESERVED_MFR_32DC
  {0x30B0, 0x0800}, // DIGITAL_TEST
  {0x302A, 0x0006}, // VT_PIX_CLK_DIV
  {0x302C, 0x0001}, // VT_SYS_CLK_DIV
  {0x302E, 0x0002}, // PRE_PLL_CLK_DIV
  {0x3030, 0x002C}, // PLL_MULTIPLIER
  {0x3036, 0x000A}, // OP_WORD_CLK_DIV
  {0x3038, 0x0001}, // OP_SYS_CLK_DIV
  {0x30B0, 0x0800}, // DIGITAL_TEST
  {0x30A2, 0x0001}, // X_ODD_INC_
  {0x30A6, 0x0001}, // Y_ODD_INC_
  {0x3040, 0xC000}, // READ_MODE C000
  {0x30BA, 0x11F2}, // DIGITAL_CTRL
  {0x3044, 0x0400}, // DARK_CONTROL
  {0x3064, 0x1802}, // SMIA_TEST
  /*{0x3064, 0xCC2}, // STATS_EN
  {0x3270, 0x10}, //
  {0x3272, 0x30}, //
  {0x3274, 0x50}, //
  {0x3276, 0x10}, //
  {0x3278, 0x30}, //
  {0x327A, 0x50}, //

  {0x3144, 0x0}, //
  {0x3146, 0x0}, //
  {0x3244, 0x0}, //
  {0x3246, 0x0}, //
  {0x3268, 0x0}, //
  {0x326A, 0x0}, //
  */
  {0x33E0, 0x0C80}, // TEST_ASIL_ROWS
  {0x3180, 0x0080}, // RESERVED_MFR_3180
  {0x33E4, 0x0080}, // RESERVED_MFR_33E4
  {0x33E0, 0x0C80}, // TEST_ASIL_ROWS
  {0x33E0, 0x0C80}, // TEST_ASIL_ROWS
  {0x3004, 0x0000}, // X_ADDR_START_
  {0x3008, 0x0787}, // X_ADDR_END_ 787
  {0x3002, 0x0000}, // Y_ADDR_START_
  {0x3006, 0x04B7}, // Y_ADDR_END_ 4B7
  {0x3032, 0x0000}, // SCALING_MODE
  {0x3400, 0x0010}, // RESERVED_MFR_3400
  {0x3402, 0x0788}, // X_OUTPUT_CONTROL
  {0x3402, 0x0F10}, // X_OUTPUT_CONTROL
  {0x3404, 0x04B8}, // Y_OUTPUT_CONTROL
  {0x3404, 0x0970}, // Y_OUTPUT_CONTROL
  {0x30BA, 0x11F3}, // DIGITAL_CTRL

  // SLAV* MODE
  {0x30CE, 0x0120},
  {0x340A, 0xE6}, // E6 // 0000 1110 0110
  {0x340C, 0x802}, // 2 // 0000 0000 0010

  // FPS = 88e6 / 0x09C4 / 0x06E0 = 20
  {0x300C, 0x0672}, // LINE_LENGTH_PCK_ 9B4
  {0x300A, 0x0A6A}, // FRAME_LENGTH_LINES_ 6EB
  {0x3042, 0x0000}, // EXTRA_DELAY

  // Readout Settings
  {0x31AE, 0x0204}, // SERIAL_FORMAT, 4-lane MIPI
  {0x31AC, 0x0C0A}, // DATA_FORMAT_BITS, 12 -> 10
  {0x3342, 0x122B}, // MIPI_F1_PDT_EDT
  {0x3346, 0x122B}, // MIPI_F2_PDT_EDT
  {0x334A, 0x122B}, // MIPI_F3_PDT_EDT
  {0x334E, 0x122B}, // MIPI_F4_PDT_EDT
  {0x3344, 0x0011}, // MIPI_F1_VDT_VC
  {0x3348, 0x0111}, // MIPI_F2_VDT_VC
  {0x334C, 0x0211}, // MIPI_F3_VDT_VC
  {0x3350, 0x0311}, // MIPI_F4_VDT_VC
  {0x31B0, 0x0049}, // FRAME_PREAMBLE
  {0x31B2, 0x0033}, // LINE_PREAMBLE
  {0x31B4, 0x2185}, // RESERVED_MFR_31B4
  {0x31B6, 0x1146}, // RESERVED_MFR_31B6
  {0x31B8, 0x3047}, // RESERVED_MFR_31B8
  {0x31BA, 0x0186}, // RESERVED_MFR_31BA
  {0x31BC, 0x0805}, // RESERVED_MFR_31BC
  {0x301A, 0x01C}, // RESET_REGISTER

  // HDR Settings
  {0x3082, 0x0004}, // OPERATION_MODE_CTRL
  {0x3238, 0x0222}, // EXPOSURE_RATIO
  {0x3014, 0x098E}, // FINE_INTEGRATION_TIME_
  {0x321E, 0x098E}, // FINE_INTEGRATION_TIME2
  {0x3222, 0x098E}, // FINE_INTEGRATION_TIME3
  {0x3226, 0x098E}, // FINE_INTEGRATION_TIME4, 098E?
  {0x30B0, 0x0800}, // DIGITAL_TEST
  {0x32EA, 0x3C0E}, // RESERVED_MFR_32EA
  {0x32EC, 0x72A1}, // RESERVED_MFR_32EC
  {0x31D0, 0x0000}, // COMPANDING, no good in 10 bit?
  {0x33DA, 0x0000}, // COMPANDING
  {0x3362, 0x0000}, // DC GAIN
  {0x3370, 0x0231}, // DBLC
  {0x318E, 0x0200}, // PRE_HDR_GAIN_EN

   // Initial Gains
  {0x3022, 0x01}, // GROUPED_PARAMETER_HOLD_
  {0x3366, 0x5555}, // ANALOG_GAIN
  {0x3060, 0xBBBB}, // ANALOG_COLOR_GAIN
  {0x305A, 0x00D8}, // RED_GAIN
  {0x3058, 0x011B}, // BLUE_GAIN
  {0x3056, 0x009A}, // GREEN1_GAIN
  {0x305C, 0x009A}, // GREEN2_GAIN
  {0x3022, 0x00}, // GROUPED_PARAMETER_HOLD_

  // Initial Integration Time
  {0x3012, 0x256},

};

struct i2c_random_wr_payload poke_array_ov7750[] = {
  {0x3208, 0x0}, {0x380e, 0x1a}, {0x380f, 0xf0}, {0x3500, 0x0}, {0x3501, 0x0}, {0x3502, 0x10}, {0x350a, 0x0}, {0x350b, 0x10}, {0x3208, 0x10}, {0x3208, 0xa0},
  //{0x3208, 0x0}, {0x380e, 0x1a}, {0x380f, 0xf0}, {0x3500, 0x0}, {0x3501, 0x0}, {0x3502, 0x10}, {0x350a, 0x0}, {0x350b, 0x10}, {0x3208, 0x10}, {0x3208, 0xa0},
};

struct i2c_random_wr_payload preinit_array_ov7750[] = {
  {0x103, 0x1},
  {0x303b, 0x2},
  {0x302b, 0x80},
};

struct i2c_random_wr_payload init_array_ov7750[] = {
  // 2nd batch
  {0x3005, 0x0},
  {0x3012, 0xc0},
  {0x3013, 0xd2},
  {0x3014, 0x4},
  {0x3016, 0xf0},
  {0x3017, 0xf0},
  {0x3018, 0xf0},
  {0x301a, 0xf0},
  {0x301b, 0xf0},
  {0x301c, 0xf0},
  {0x3023, 0x5},
  {0x3037, 0xf0},
  {0x3098, 0x4},
  {0x3099, 0x28},
  {0x309a, 0x5},
  {0x309b, 0x4},
  {0x30b0, 0xa},
  {0x30b1, 0x1},
  {0x30b3, 0x64},
  {0x30b4, 0x3},
  {0x30b5, 0x5},
  {0x3106, 0xda},
  {0x3500, 0x0},
  {0x3501, 0x1f},
  {0x3502, 0x80},
  {0x3503, 0x7},
  {0x3509, 0x10},
  {0x350b, 0x10},
  {0x3600, 0x1c},
  {0x3602, 0x62},
  {0x3620, 0xb7},
  {0x3622, 0x4},
  {0x3626, 0x21},
  {0x3627, 0x30},
  {0x3630, 0x44},
  {0x3631, 0x35},
  {0x3634, 0x60},
  {0x3636, 0x0},
  {0x3662, 0x1},
  {0x3663, 0x70},
  {0x3664, 0xf0},
  {0x3666, 0xa},
  {0x3669, 0x1a},
  {0x366a, 0x0},
  {0x366b, 0x50},
  {0x3673, 0x1},
  {0x3674, 0xff},
  {0x3675, 0x3},
  {0x3705, 0xc1},
  {0x3709, 0x40},
  {0x373c, 0x8},
  {0x3742, 0x0},
  {0x3757, 0xb3},
  {0x3788, 0x0},
  {0x37a8, 0x1},
  {0x37a9, 0xc0},
  {0x3800, 0x0},
  {0x3801, 0x4},
  {0x3802, 0x0},
  {0x3803, 0x4},
  {0x3804, 0x2},
  {0x3805, 0x8b},
  {0x3806, 0x1},
  {0x3807, 0xeb},
  {0x3808, 0x2},
  {0x3809, 0x80},
  {0x380a, 0x1},
  {0x380b, 0xe0},
  {0x380c, 0x3},
  {0x380d, 0xa0},
  {0x380e, 0x6},
  {0x380f, 0xbc},
  {0x3810, 0x0},
  {0x3811, 0x4},
  {0x3812, 0x0},
  {0x3813, 0x5},
  {0x3814, 0x11},
  {0x3815, 0x11},
  {0x3820, 0x40},
  {0x3821, 0x0},
  {0x382f, 0xe},
  {0x3832, 0x0},
  {0x3833, 0x5},
  {0x3834, 0x0},
  {0x3835, 0xc},
  {0x3837, 0x0},
  {0x3b80, 0x0},
  {0x3b81, 0xa5},
  {0x3b82, 0x10},
  {0x3b83, 0x0},
  {0x3b84, 0x8},
  {0x3b85, 0x0},
  {0x3b86, 0x1},
  {0x3b87, 0x0},
  {0x3b88, 0x0},
  {0x3b89, 0x0},
  {0x3b8a, 0x0},
  {0x3b8b, 0x5},
  {0x3b8c, 0x0},
  {0x3b8d, 0x0},
  {0x3b8e, 0x0},
  {0x3b8f, 0x1a},
  {0x3b94, 0x5},
  {0x3b95, 0xf2},
  {0x3b96, 0x40},
  {0x3c00, 0x89},
  {0x3c01, 0x63},
  {0x3c02, 0x1},
  {0x3c03, 0x0},
  {0x3c04, 0x0},
  {0x3c05, 0x3},
  {0x3c06, 0x0},
  {0x3c07, 0x6},
  {0x3c0c, 0x1},
  {0x3c0d, 0xd0},
  {0x3c0e, 0x2},
  {0x3c0f, 0xa},
  {0x4001, 0x42},
  {0x4004, 0x4},
  {0x4005, 0x0},
  {0x404e, 0x1},
  {0x4300, 0xff},
  {0x4301, 0x0},
  {0x4315, 0x0},
  {0x4501, 0x48},
  {0x4600, 0x0},
  {0x4601, 0x4e},
  {0x4801, 0xf},
  {0x4806, 0xf},
  {0x4819, 0xaa},
  {0x4823, 0x3e},
  {0x4837, 0x19},
  {0x4a0d, 0x0},
  {0x4a47, 0x7f},
  {0x4a49, 0xf0},
  {0x4a4b, 0x30},
  {0x5000, 0x85},
  {0x5001, 0x80},
};

struct i2c_random_wr_payload init_array_ov8856[] = {
  // part 1 184
  {0x103, 0x1},
  {0x302, 0x3c},
  {0x303, 0x1},
  {0x31e, 0xc},
  {0x3000, 0x0},
  {0x300e, 0x0},
  {0x3010, 0x0},
  {0x3015, 0x84},
  {0x3018, 0x72},
  {0x3033, 0x24},
  {0x3500, 0x0},
  {0x3501, 0x4c},
  {0x3502, 0xe0},
  {0x3503, 0x8},
  {0x3505, 0x83},
  {0x3508, 0x1},
  {0x3509, 0x80},
  {0x350c, 0x0},
  {0x350d, 0x80},
  {0x350e, 0x4},
  {0x350f, 0x0},
  {0x3510, 0x0},
  {0x3511, 0x2},
  {0x3512, 0x0},
  {0x3600, 0x72},
  {0x3601, 0x40},
  {0x3602, 0x30},
  {0x3610, 0xc5},
  {0x3611, 0x58},
  {0x3612, 0x5c},
  {0x3613, 0x5a},
  {0x3614, 0x60},
  {0x3628, 0xff},
  {0x3629, 0xff},
  {0x362a, 0xff},
  {0x3633, 0x10},
  {0x3634, 0x10},
  {0x3635, 0x10},
  {0x3636, 0x10},
  {0x3663, 0x8},
  {0x3669, 0x34},
  {0x366e, 0x8},
  {0x3706, 0x86},
  {0x370b, 0x7e},
  {0x3714, 0x27},
  {0x3730, 0x12},
  {0x3733, 0x10},
  {0x3764, 0x0},
  {0x3765, 0x0},
  {0x3769, 0x62},
  {0x376a, 0x2a},
  {0x376b, 0x3b},
  {0x3780, 0x0},
  {0x3781, 0x24},
  {0x3782, 0x0},
  {0x3783, 0x23},
  {0x3798, 0x2f},
  {0x37a1, 0x60},
  {0x37a8, 0x6a},
  {0x37ab, 0x3f},
  {0x37c2, 0x14},
  {0x37c3, 0xf1},
  {0x37c9, 0x80},
  {0x37cb, 0x3},
  {0x37cc, 0xa},
  {0x37cd, 0x16},
  {0x37ce, 0x1f},
  {0x3800, 0x0},
  {0x3801, 0x0},
  {0x3802, 0x0},
  {0x3803, 0xc},
  {0x3804, 0xc},
  {0x3805, 0xdf},
  {0x3806, 0x9},
  {0x3807, 0xa3},
  {0x3808, 0x6},
  {0x3809, 0x60},
  {0x380a, 0x4},
  {0x380b, 0xc8},
  {0x380c, 0x7},
  {0x380d, 0x8c},
  {0x380e, 0x9},
  {0x380f, 0xb2},
  {0x3810, 0x0},
  {0x3811, 0x8},
  {0x3812, 0x0},
  {0x3813, 0x2},
  {0x3814, 0x3},
  {0x3815, 0x1},
  {0x3816, 0x0},
  {0x3817, 0x0},
  {0x3818, 0x0},
  {0x3819, 0x0},
  {0x3820, 0x90},
  {0x3821, 0x67},
  {0x382a, 0x3},
  {0x382b, 0x1},
  {0x3830, 0x6},
  {0x3836, 0x2},
  {0x3862, 0x4},
  {0x3863, 0x8},
  {0x3cc0, 0x33},
  {0x3d85, 0x17},
  {0x3d8c, 0x73},
  {0x3d8d, 0xde},
  {0x4001, 0xe0},
  {0x4003, 0x40},
  {0x4008, 0x0},
  {0x4009, 0x5},
  {0x400f, 0x80},
  {0x4010, 0xf0},
  {0x4011, 0xff},
  {0x4012, 0x2},
  {0x4013, 0x1},
  {0x4014, 0x1},
  {0x4015, 0x1},
  {0x4042, 0x0},
  {0x4043, 0x80},
  {0x4044, 0x0},
  {0x4045, 0x80},
  {0x4046, 0x0},
  {0x4047, 0x80},
  {0x4048, 0x0},
  {0x4049, 0x80},
  {0x4041, 0x3},
  {0x404c, 0x20},
  {0x404d, 0x0},
  {0x404e, 0x20},
  {0x4203, 0x80},
  {0x4307, 0x30},
  {0x4317, 0x0},
  {0x4503, 0x8},
  {0x4601, 0x80},
  {0x4816, 0x53},
  {0x481b, 0x58},
  {0x481f, 0x27},
  {0x4837, 0x16},
  {0x5000, 0x77},
  {0x5001, 0xe},
  {0x5004, 0x0},
  {0x502e, 0x0},
  {0x5030, 0x41},
  {0x5795, 0x0},
  {0x5796, 0x10},
  {0x5797, 0x10},
  {0x5798, 0x73},
  {0x5799, 0x73},
  {0x579a, 0x0},
  {0x579b, 0x28},
  {0x579c, 0x0},
  {0x579d, 0x16},
  {0x579e, 0x6},
  {0x579f, 0x20},
  {0x57a0, 0x4},
  {0x57a1, 0xa0},
  {0x5780, 0x14},
  {0x5781, 0xf},
  {0x5782, 0x44},
  {0x5783, 0x2},
  {0x5784, 0x1},
  {0x5785, 0x1},
  {0x5786, 0x0},
  {0x5787, 0x4},
  {0x5788, 0x2},
  {0x5789, 0xf},
  {0x578a, 0xfd},
  {0x578b, 0xf5},
  {0x578c, 0xf5},
  {0x578d, 0x3},
  {0x578e, 0x8},
  {0x578f, 0xc},
  {0x5790, 0x8},
  {0x5791, 0x4},
  {0x5792, 0x0},
  {0x5793, 0x52},
  {0x5794, 0xa3},
  {0x5a08, 0x2},
  {0x5b00, 0x2},
  {0x5b01, 0x10},
  {0x5b02, 0x3},
  {0x5b03, 0xcf},
  {0x5b05, 0x6c},
  {0x5e00, 0x0},

  // part 2 45
  {0x3501, 0x9a},
  {0x3502, 0x20},
  {0x366d, 0x0},
  {0x366e, 0x10},
  {0x3714, 0x23},
  {0x37c2, 0x4},
  {0x3800, 0x0},
  {0x3801, 0x0},
  {0x3802, 0x0},
  {0x3803, 0xc},
  {0x3804, 0xc},
  {0x3805, 0xdf},
  {0x3806, 0x9},
  {0x3807, 0xa3},
  {0x3808, 0xc},
  {0x3809, 0xc0},
  {0x380a, 0x9},
  {0x380b, 0x90},
  {0x380c, 0x7},
  {0x380d, 0x8c},
  {0x380e, 0x9},
  {0x380f, 0xb2},
  {0x3811, 0x10},
  {0x3813, 0x4},
  {0x3814, 0x1},
  {0x3820, 0xc6},
  {0x3821, 0x40},
  {0x382a, 0x1},
  {0x4009, 0xb},
  {0x4601, 0x80},
  {0x5003, 0xc8},
  {0x5006, 0x0},
  {0x5007, 0x0},
  {0x5795, 0x2},
  {0x5796, 0x20},
  {0x5797, 0x20},
  {0x5798, 0xd5},
  {0x5799, 0xd5},
  {0x579b, 0x50},
  {0x579d, 0x2c},
  {0x579e, 0xc},
  {0x579f, 0x40},
  {0x57a0, 0x9},
  {0x57a1, 0x40},
  {0x5e10, 0xfc},
};

