struct i2c_random_wr_payload start_reg_array_ar0231[] = {{0x301A, 0x91C}};
struct i2c_random_wr_payload stop_reg_array_ar0231[] = {{0x301A, 0x918}};
struct i2c_random_wr_payload start_reg_array_imx390[] = {{0x0, 0}};
struct i2c_random_wr_payload stop_reg_array_imx390[] = {{0x0, 1}};
struct i2c_random_wr_payload start_reg_array_os04c10[] = {{0x100, 1}};
struct i2c_random_wr_payload stop_reg_array_os04c10[] = {{0x100, 0}};

struct i2c_random_wr_payload init_array_os04c10[] = {
  {0x103, 1},

  //{0x5000, 1},

  //{0x5003, 0xc0},

  // switch to 2 lane mode
  //{0x3016, 0x32}
  //{0x100, 0},

  // disable reset

  // 24 mhz in

  // PLL1 (this is correct for 10-bit MIPI)
  // 24/1.5*0x5a = 1440 mhz PHY clock (MIPI)
  {0x303, 0x01},  // pre_div = /1.5
  {0x304, 0x00}, {0x305, 0x5a}, // pll1_divp = 0x5a
  {0x301, 0x84},  // pre_div0, pll1_divsys(/5)[7:6] = 2, pll1_predivp(/1)[5] = 0, pll1_divs(/4)[3:2] = 1
  // 1440/5/2 = 144 mhz SCLK
  {0x3106, 0x25}, // dig_sclk_div[3:2] = 1 (/2)
  // 1440/4/2 = 180 mhz PCLK (this is 2x too high)
  {0x3020, 0x17},     // pclk_sel[3:2] = 1 (/2)

  // PLL2
  // 24/2/3*0x2bc = 2800 mhz other clock (this valid?)
  {0x323, 0x04},                // pll2_prediv(/3)[2:0]
  {0x324, 0x02}, {0x325, 0xbc}, // pll2_divp
  // 2800/7 = 400 mhz SA1_CLK
  {0x326, 0x82},                // pre_div0(/2)[7], pll2_divsa1(/7)[1:0]
  // 2800/5 = 560 mhz SRAM_CLK
  {0x327, 0x04},                // pll2_digsram(/5)[2:0] 
  // 2800/2 = 1400 mhz DAC_CLK
  {0x329, 0x1},                 // pll2_divdac(/2)[1:0]

  //{0x302, 0x01},
  //{0x303, 0x02},
  //{0x303, 0x05}, // setting this to 5 gets output (/4)
  //{0x306, 0},                   // pll1_divpix

  //{0x323, 0x07},                // pll2_prediv
  //{0x324, 0x01}, {0x325, 0x5e}, // pll2_divp
  //{0x324, 0x00}, {0x325, 0xfc}, // pll2_divp

  // pll2_predivp = 1
  // pll2_divsa1 = 2
  //{0x321, 1},

  // exposure
  {0x3501, 0x3f},

  // https://github.com/hanwckf/linux-rk3328-box/blob/master/drivers/media/i2c/ov4689.c
  {0x481f, 0x40},
  {0x4829, 0x78},
  {0x4837, 0x10},

  // MIPI PCLK period (broken with 0xe stock, MIPI/PHY = 1440/180 = 8)
  // this makes output happen
  {0x4837, 0x8}

  // data type
  {0x4814, 0x2B},
  {0x4815, 0x2B},
  {0x4816, 0x2B},

  //{0x3508, 0xf},

  // MIPI (10-bit mode)
  //{0x3022, 0x01},

  // MIPI_CORE_37 = 0xe (MIPI PCLK period)

  // output size
  // 1928
  //{0x3808, 0x07}, {0x3809, 0x88},
  //{0x380A, 0x00}, {0x380B, 0xb8},
  //{0x380c, 0x2a}, {0x380d, 0x18},

  //{0x5000, 0xe8}, // isp disable
};

struct i2c_random_wr_payload init_array_imx390[] = {
  {0x2008, 0xd0}, {0x2009, 0x07}, {0x200a, 0x00}, // MODE_VMAX = time between frames
  {0x200C, 0xe4}, {0x200D, 0x0c},  // MODE_HMAX

  // crop
  {0x3410, 0x88}, {0x3411, 0x7},     // CROP_H_SIZE
  {0x3418, 0xb8}, {0x3419, 0x4},     // CROP_V_SIZE
  {0x0078, 1}, {0x03c0, 1},

  // external trigger (off)
  // while images still come in, they are blank with this
  {0x3650, 0},  // CU_MODE

  // exposure
  {0x000c, 0xc0}, {0x000d, 0x07},
  {0x0010, 0xc0}, {0x0011, 0x07},

  // WUXGA mode
  // not in datasheet, from https://github.com/bogsen/STLinux-Kernel/blob/master/drivers/media/platform/tegra/imx185.c
  {0x0086, 0xc4}, {0x0087, 0xff},   // WND_SHIFT_V = -60
  {0x03c6, 0xc4}, {0x03c7, 0xff},   // SM_WND_SHIFT_V_APL = -60

  {0x201c, 0xe1}, {0x201d, 0x12},   // image read amount
  {0x21ee, 0xc4}, {0x21ef, 0x04},   // image send amount (1220 is the end)
  {0x21f0, 0xc4}, {0x21f1, 0x04},   // image processing amount

  // disable a bunch of errors causing blanking
  {0x0390, 0x00}, {0x0391, 0x00}, {0x0392, 0x00},

  // flip bayer
  {0x2D64, 0x64 + 2},

  // color correction
  {0x0030, 0xf8}, {0x0031, 0x00},  // red gain
  {0x0032, 0x9a}, {0x0033, 0x00},  // gr gain
  {0x0034, 0x9a}, {0x0035, 0x00},  // gb gain
  {0x0036, 0x22}, {0x0037, 0x01},  // blue gain

  // hdr enable (noise with this on for now)
  {0x00f9, 0}
};

struct i2c_random_wr_payload init_array_ar0231[] = {
  {0x301A, 0x0018}, // RESET_REGISTER

  // CLOCK Settings
  // input clock is 19.2 / 2 * 0x37 = 528 MHz
  // pixclk is 528 / 6 = 88 MHz
  // full roll time is 1000/(PIXCLK/(LINE_LENGTH_PCK*FRAME_LENGTH_LINES)) = 39.99 ms
  // img  roll time is 1000/(PIXCLK/(LINE_LENGTH_PCK*Y_OUTPUT_CONTROL))   = 22.85 ms
  {0x302A, 0x0006}, // VT_PIX_CLK_DIV
  {0x302C, 0x0001}, // VT_SYS_CLK_DIV
  {0x302E, 0x0002}, // PRE_PLL_CLK_DIV
  {0x3030, 0x0037}, // PLL_MULTIPLIER
  {0x3036, 0x000C}, // OP_PIX_CLK_DIV
  {0x3038, 0x0001}, // OP_SYS_CLK_DIV

  // FORMAT
  {0x3040, 0xC000}, // READ_MODE
  {0x3004, 0x0000}, // X_ADDR_START_
  {0x3008, 0x0787}, // X_ADDR_END_
  {0x3002, 0x0000}, // Y_ADDR_START_
  {0x3006, 0x04B7}, // Y_ADDR_END_
  {0x3032, 0x0000}, // SCALING_MODE
  {0x30A2, 0x0001}, // X_ODD_INC_
  {0x30A6, 0x0001}, // Y_ODD_INC_
  {0x3402, 0x0788}, // X_OUTPUT_CONTROL
  {0x3404, 0x04B8}, // Y_OUTPUT_CONTROL
  {0x3064, 0x1982}, // SMIA_TEST
  {0x30BA, 0x11F2}, // DIGITAL_CTRL

  // Enable external trigger and disable GPIO outputs
  {0x30CE, 0x0120}, // SLAVE_SH_SYNC_MODE | FRAME_START_MODE
  {0x340A, 0xE0},   // GPIO3_INPUT_DISABLE | GPIO2_INPUT_DISABLE | GPIO1_INPUT_DISABLE
  {0x340C, 0x802},  // GPIO_HIDRV_EN | GPIO0_ISEL=2

  // Readout timing
  {0x300C, 0x0672}, // LINE_LENGTH_PCK (valid for 3-exposure HDR)
  {0x300A, 0x0855}, // FRAME_LENGTH_LINES
  {0x3042, 0x0000}, // EXTRA_DELAY

  // Readout Settings
  {0x31AE, 0x0204}, // SERIAL_FORMAT, 4-lane MIPI
  {0x31AC, 0x0C0C}, // DATA_FORMAT_BITS, 12 -> 12
  {0x3342, 0x1212}, // MIPI_F1_PDT_EDT
  {0x3346, 0x1212}, // MIPI_F2_PDT_EDT
  {0x334A, 0x1212}, // MIPI_F3_PDT_EDT
  {0x334E, 0x1212}, // MIPI_F4_PDT_EDT
  {0x3344, 0x0011}, // MIPI_F1_VDT_VC
  {0x3348, 0x0111}, // MIPI_F2_VDT_VC
  {0x334C, 0x0211}, // MIPI_F3_VDT_VC
  {0x3350, 0x0311}, // MIPI_F4_VDT_VC
  {0x31B0, 0x0053}, // FRAME_PREAMBLE
  {0x31B2, 0x003B}, // LINE_PREAMBLE
  {0x301A, 0x001C}, // RESET_REGISTER

  // Noise Corrections
  {0x3092, 0x0C24}, // ROW_NOISE_CONTROL
  {0x337A, 0x0C80}, // DBLC_SCALE0
  {0x3370, 0x03B1}, // DBLC
  {0x3044, 0x0400}, // DARK_CONTROL

  // Enable temperature sensor
  {0x30B4, 0x0007}, // TEMPSENS0_CTRL_REG
  {0x30B8, 0x0007}, // TEMPSENS1_CTRL_REG

  // Enable dead pixel correction using
  // the 1D line correction scheme
  {0x31E0, 0x0003},

  // HDR Settings
  {0x3082, 0x0004}, // OPERATION_MODE_CTRL
  {0x3238, 0x0444}, // EXPOSURE_RATIO

  {0x1008, 0x0361}, // FINE_INTEGRATION_TIME_MIN
  {0x100C, 0x0589}, // FINE_INTEGRATION_TIME2_MIN
  {0x100E, 0x07B1}, // FINE_INTEGRATION_TIME3_MIN
  {0x1010, 0x0139}, // FINE_INTEGRATION_TIME4_MIN

  // TODO: do these have to be lower than LINE_LENGTH_PCK?
  {0x3014, 0x08CB}, // FINE_INTEGRATION_TIME_
  {0x321E, 0x0894}, // FINE_INTEGRATION_TIME2

  {0x31D0, 0x0000}, // COMPANDING, no good in 10 bit?
  {0x33DA, 0x0000}, // COMPANDING
  {0x318E, 0x0200}, // PRE_HDR_GAIN_EN

  // DLO Settings
  {0x3100, 0x4000}, // DLO_CONTROL0
  {0x3280, 0x0CCC}, // T1 G1
  {0x3282, 0x0CCC}, // T1 R
  {0x3284, 0x0CCC}, // T1 B
  {0x3286, 0x0CCC}, // T1 G2
  {0x3288, 0x0FA0}, // T2 G1
  {0x328A, 0x0FA0}, // T2 R
  {0x328C, 0x0FA0}, // T2 B
  {0x328E, 0x0FA0}, // T2 G2

   // Initial Gains
  {0x3022, 0x0001}, // GROUPED_PARAMETER_HOLD_
  {0x3366, 0xFF77}, // ANALOG_GAIN (1x)

  {0x3060, 0x3333}, // ANALOG_COLOR_GAIN

  {0x3362, 0x0000}, // DC GAIN

  {0x305A, 0x00F8}, // red gain
  {0x3058, 0x0122}, // blue gain
  {0x3056, 0x009A}, // g1 gain
  {0x305C, 0x009A}, // g2 gain

  {0x3022, 0x0000}, // GROUPED_PARAMETER_HOLD_

  // Initial Integration Time
  {0x3012, 0x0005},
};
