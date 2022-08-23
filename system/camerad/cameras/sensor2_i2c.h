struct i2c_random_wr_payload start_reg_array_ar0231[] = {{0x301A, 0x91C}};
struct i2c_random_wr_payload stop_reg_array_ar0231[] = {{0x301A, 0x918}};
struct i2c_random_wr_payload start_reg_array_imx390[] = {{0x0, 0}};
struct i2c_random_wr_payload stop_reg_array_imx390[] = {{0x0, 1}};
struct i2c_random_wr_payload start_reg_array_os04c10[] = {{0x100, 1}};
struct i2c_random_wr_payload stop_reg_array_os04c10[] = {{0x100, 0}};

struct i2c_random_wr_payload init_array_os04c10[] = {
  {0x107, 1},

// X3C_1920x1280_60fps_HDR4_LFR_PWL12_mipi1200
{0x4d5a, 0x1a},
{0x4d09, 0xff},
{0x4d09, 0xdf},
{0x3208, 0x04},
{0x4620, 0x04},
{0x3208, 0x14},
{0x3208, 0x05},
{0x4620, 0x04},
{0x3208, 0x15},
{0x3208, 0x02},
{0x3507, 0x00},
{0x3208, 0x12},
{0x3208, 0xa2},

// PLL setup
{0x0301, 0xc8},
{0x0303, 0x01},
{0x0304, 0x01},
{0x0305, 0x2c},
{0x0306, 0x04},
{0x0307, 0x01},
{0x0316, 0x00},
{0x0317, 0x00},
{0x0318, 0x00},
{0x0323, 0x05},
{0x0324, 0x01},
{0x0325, 0x2c},

// SCLK/PCLK
{0x0400, 0xe0},
{0x0401, 0x80},
{0x0403, 0xde},
{0x0404, 0x34},
{0x0405, 0x3b},
{0x0406, 0xde},
{0x0407, 0x08},
{0x0408, 0xe0},
{0x0409, 0x7f},
{0x040a, 0xde},
{0x040b, 0x34},
{0x040c, 0x47},
{0x040d, 0xd8},
{0x040e, 0x08},

{0x2803, 0xfe},
{0x280b, 0x00},
{0x280c, 0x79},
{0x3001, 0x03},
{0x3002, 0xf8},
{0x3005, 0x80},
{0x3007, 0x01},
{0x3008, 0x80},
{0x3012, 0x41},
{0x3020, 0x05},
{0x3700, 0x28},
{0x3701, 0x15},
{0x3702, 0x19},
{0x3703, 0x23},
{0x3704, 0x0a},
{0x3705, 0x00},
{0x3706, 0x3e},
{0x3707, 0x0d},
{0x3708, 0x50},
{0x3709, 0x5a},
{0x370a, 0x00},
{0x370b, 0x96},
{0x3711, 0x11},
{0x3712, 0x13},
{0x3717, 0x02},
{0x3718, 0x73},
{0x372c, 0x40},
{0x3733, 0x01},
{0x3738, 0x36},
{0x3739, 0x36},
{0x373a, 0x25},
{0x373b, 0x25},
{0x373f, 0x21},
{0x3740, 0x21},
{0x3741, 0x21},
{0x3742, 0x21},
{0x3747, 0x28},
{0x3748, 0x28},
{0x3749, 0x19},
{0x3755, 0x1a},
{0x3756, 0x0a},
{0x3757, 0x1c},
{0x3765, 0x19},
{0x3766, 0x05},
{0x3767, 0x05},
{0x3768, 0x13},
{0x376c, 0x07},
{0x3778, 0x20},
{0x377c, 0xc8},
{0x3781, 0x02},
{0x3783, 0x02},
{0x379c, 0x58},
{0x379e, 0x00},
{0x379f, 0x00},
{0x37a0, 0x00},
{0x37bc, 0x22},
{0x37c0, 0x01},
{0x37c4, 0x3e},
{0x37c5, 0x3e},
{0x37c6, 0x2a},
{0x37c7, 0x28},
{0x37c8, 0x02},
{0x37c9, 0x12},
{0x37cb, 0x29},
{0x37cd, 0x29},
{0x37d2, 0x00},
{0x37d3, 0x73},
{0x37d6, 0x00},
{0x37d7, 0x6b},
{0x37dc, 0x00},
{0x37df, 0x54},
{0x37e2, 0x00},
{0x37e3, 0x00},
{0x37f8, 0x00},
{0x37f9, 0x01},
{0x37fa, 0x00},
{0x37fb, 0x19},
{0x3c03, 0x01},
{0x3c04, 0x01},
{0x3c06, 0x21},
{0x3c08, 0x01},
{0x3c09, 0x01},
{0x3c0a, 0x01},
{0x3c0b, 0x21},
{0x3c13, 0x21},
{0x3c14, 0x82},
{0x3c16, 0x13},
{0x3c21, 0x00},
{0x3c22, 0xf3},
{0x3c37, 0x12},
{0x3c38, 0x31},
{0x3c3c, 0x00},
{0x3c3d, 0x03},
{0x3c44, 0x16},
{0x3c5c, 0x8a},
{0x3c5f, 0x03},
{0x3c61, 0x80},
{0x3c6f, 0x2b},
{0x3c70, 0x5f},
{0x3c71, 0x2c},
{0x3c72, 0x2c},
{0x3c73, 0x2c},
{0x3c76, 0x12},
{0x3182, 0x12},
{0x320e, 0x00},
{0x320f, 0x00},
{0x3211, 0x61},
{0x3215, 0xcd},
{0x3219, 0x08},
{0x3506, 0x30},
{0x350a, 0x01},
{0x350b, 0x00},
{0x350c, 0x00},
{0x3586, 0x60},
{0x358a, 0x01},
{0x358b, 0x00},
{0x358c, 0x00},
{0x3541, 0x00},
{0x3542, 0x04},
{0x3548, 0x04},
{0x3549, 0x40},
{0x354a, 0x01},
{0x354b, 0x00},
{0x354c, 0x00},
{0x35c1, 0x00},
{0x35c2, 0x02},
{0x35c6, 0xa0},
{0x3600, 0x8f},
{0x3605, 0x16},
{0x3609, 0xf0},
{0x360a, 0x01},
{0x360e, 0x1d},
{0x360f, 0x10},
{0x3610, 0x70},
{0x3611, 0x3a},
{0x3612, 0x28},
{0x361a, 0x29},
{0x361b, 0x6c},
{0x361c, 0x0b},
{0x361d, 0x00},
{0x361e, 0xfc},
{0x362a, 0x00},
{0x364d, 0x0f},
{0x364e, 0x18},
{0x364f, 0x12},
{0x3653, 0x1c},
{0x3654, 0x00},
{0x3655, 0x1f},
{0x3656, 0x1f},
{0x3657, 0x0c},
{0x3658, 0x0a},
{0x3659, 0x14},
{0x365a, 0x18},
{0x365b, 0x14},
{0x365c, 0x10},
{0x365e, 0x12},
{0x3674, 0x08},
{0x3677, 0x3a},
{0x3678, 0x3a},
{0x3679, 0x19},

// Y_ADDR_START = 4
{0x3802, 0x00},
{0x3803, 0x04},
// Y_ADDR_END = 0x50b
{0x3806, 0x05},
{0x3807, 0x0b},

// X_OUTPUT_SIZE = 0x780 = 1920 (changed to 1928)
{0x3808, 0x07},
{0x3809, 0x88},

// Y_OUTPUT_SIZE = 0x500 = 1280 (changed to 0x4b0)
{0x380a, 0x04},
{0x380b, 0xb0},

// horizontal timing 0x447
{0x380c, 0x04},
{0x380d, 0x47},

// rows per frame (was 0x2ae)
{0x380e, 0x08},
{0x380f, 0xae},

{0x3810, 0x00},
{0x3811, 0x08},
{0x3812, 0x00},
{0x3813, 0x04},
{0x3816, 0x01},
{0x3817, 0x01},
{0x381c, 0x18},
{0x381e, 0x01},
{0x381f, 0x01},

// mirror and flip
{0x3820, 0x24},

{0x3821, 0x19},
{0x3832, 0x00},
{0x3834, 0x00},
{0x384c, 0x02},
{0x384d, 0x0d},
{0x3850, 0x00},
{0x3851, 0x42},
{0x3852, 0x00},
{0x3853, 0x40},
{0x3858, 0x04},
{0x388c, 0x02},
{0x388d, 0x2b},
{0x3b40, 0x05},
{0x3b41, 0x40},
{0x3b42, 0x00},
{0x3b43, 0x90},
{0x3b44, 0x00},
{0x3b45, 0x20},
{0x3b46, 0x00},
{0x3b47, 0x20},
{0x3b48, 0x19},
{0x3b49, 0x12},
{0x3b4a, 0x16},
{0x3b4b, 0x2e},
{0x3b4c, 0x00},
{0x3b4d, 0x00},
{0x3b86, 0x00},
{0x3b87, 0x34},
{0x3b88, 0x00},
{0x3b89, 0x08},
{0x3b8a, 0x05},
{0x3b8b, 0x00},
{0x3b8c, 0x07},
{0x3b8d, 0x80},
{0x3b8e, 0x00},
{0x3b8f, 0x00},
{0x3b92, 0x05},
{0x3b93, 0x00},
{0x3b94, 0x07},
{0x3b95, 0x80},
{0x3b9e, 0x09},
{0x3d82, 0x73},
{0x3d85, 0x05},
{0x3d8a, 0x03},
{0x3d8b, 0xff},
{0x3d99, 0x00},
{0x3d9a, 0x9f},
{0x3d9b, 0x00},
{0x3d9c, 0xa0},
{0x3da4, 0x00},
{0x3da7, 0x50},
{0x420e, 0x6b},
{0x420f, 0x6e},
{0x4210, 0x06},
{0x4211, 0xc1},
{0x421e, 0x02},
{0x421f, 0x45},
{0x4220, 0xe1},
{0x4221, 0x01},
{0x4301, 0xff},
{0x4307, 0x03},
{0x4308, 0x13},
{0x430a, 0x13},
{0x430d, 0x93},
{0x430f, 0x57},
{0x4310, 0x95},
{0x4311, 0x16},
{0x4316, 0x00},

{0x4317, 0x38}, // both embedded rows are enabled

{0x4319, 0x03},
{0x431a, 0x00}, // 8 bit mipi
{0x431b, 0x00},
{0x431d, 0x2a},
{0x431e, 0x11},

{0x431f, 0x20}, // enable PWL, 12 bits

{0x4320, 0x19},
{0x4323, 0x80},
{0x4324, 0x00},
{0x4503, 0x4e},
{0x4505, 0x00},
{0x4509, 0x00},
{0x450a, 0x00},
{0x4580, 0xf8},
{0x4583, 0x07},
{0x4584, 0x6a},
{0x4585, 0x08},
{0x4586, 0x05},
{0x4587, 0x04},
{0x4588, 0x73},
{0x4589, 0x05},
{0x458a, 0x1f},
{0x458b, 0x02},
{0x458c, 0xdc},
{0x458d, 0x03},
{0x458e, 0x02},
{0x4597, 0x07},
{0x4598, 0x40},
{0x4599, 0x0e},
{0x459a, 0x0e},
{0x459b, 0xfb},
{0x459c, 0xf3},
{0x4602, 0x00},
{0x4603, 0x13},
{0x4604, 0x00},
{0x4609, 0x0a},
{0x460a, 0x30},
{0x4610, 0x00},
{0x4611, 0x70},
{0x4612, 0x01},
{0x4613, 0x00},
{0x4614, 0x00},
{0x4615, 0x70},
{0x4616, 0x01},
{0x4617, 0x00},
{0x4800, 0x04},
{0x480a, 0x22},
{0x4813, 0xe4},

// mipi
{0x4814, 0x2a},
{0x4837, 0x0d},
{0x484b, 0x47},
{0x484f, 0x00},
{0x4887, 0x51},
{0x4d00, 0x4a},
{0x4d01, 0x18},
{0x4d05, 0xff},
{0x4d06, 0x88},
{0x4d08, 0x63},
{0x4d09, 0xdf},
{0x4d15, 0x7d},
{0x4d1a, 0x20},
{0x4d30, 0x0a},
{0x4d31, 0x00},
{0x4d34, 0x7d},
{0x4d3c, 0x7d},
{0x4f00, 0x00},
{0x4f01, 0x00},
{0x4f02, 0x00},
{0x4f03, 0x20},
{0x4f04, 0xe0},
{0x6a00, 0x00},
{0x6a01, 0x20},
{0x6a02, 0x00},
{0x6a03, 0x20},
{0x6a04, 0x02},
{0x6a05, 0x80},
{0x6a06, 0x01},
{0x6a07, 0xe0},
{0x6a08, 0xcf},
{0x6a09, 0x01},
{0x6a0a, 0x40},
{0x6a20, 0x00},
{0x6a21, 0x02},
{0x6a22, 0x00},
{0x6a23, 0x00},
{0x6a24, 0x00},
{0x6a25, 0x00},
{0x6a26, 0x00},
{0x6a27, 0x00},
{0x6a28, 0x00},
{0x5000, 0x8f},
{0x5001, 0x75},
{0x5002, 0x7f},
{0x5003, 0x7a},
{0x5004, 0x3e},
{0x5005, 0x1e},
{0x5006, 0x1e},
{0x5007, 0x1e},
{0x5008, 0x00},
{0x500c, 0x00},
{0x502c, 0x00},
{0x502e, 0x00},
{0x502f, 0x00},
{0x504b, 0x00},
{0x5053, 0x00},
{0x505b, 0x00},
{0x5063, 0x00},
{0x5070, 0x00},
{0x5074, 0x04},
{0x507a, 0x04},
{0x507b, 0x09},
{0x5500, 0x02},
{0x5700, 0x02},
{0x5900, 0x02},
{0x6007, 0x04},
{0x6008, 0x05},
{0x6009, 0x02},
{0x600b, 0x08},
{0x600c, 0x07},
{0x600d, 0x88},
{0x6016, 0x00},
{0x6027, 0x04},
{0x6028, 0x05},
{0x6029, 0x02},
{0x602b, 0x08},
{0x602c, 0x07},
{0x602d, 0x88},
{0x6047, 0x04},
{0x6048, 0x05},
{0x6049, 0x02},
{0x604b, 0x08},
{0x604c, 0x07},
{0x604d, 0x88},
{0x6067, 0x04},
{0x6068, 0x05},
{0x6069, 0x02},
{0x606b, 0x08},
{0x606c, 0x07},
{0x606d, 0x88},
{0x6087, 0x04},
{0x6088, 0x05},
{0x6089, 0x02},
{0x608b, 0x08},
{0x608c, 0x07},
{0x608d, 0x88},
{0x5e00, 0x00},
{0x5e01, 0x09},
{0x5e02, 0x09},
{0x5e03, 0x0a},
{0x5e04, 0x0a},
{0x5e05, 0x0a},
{0x5e06, 0x0b},
{0x5e07, 0x0b},
{0x5e08, 0x0c},
{0x5e09, 0x0c},
{0x5e0a, 0x0d},
{0x5e0b, 0x0d},
{0x5e0c, 0x0e},
{0x5e0d, 0x0e},
{0x5e0e, 0x0f},
{0x5e0f, 0x0f},
{0x5e10, 0x10},
{0x5e11, 0x10},
{0x5e12, 0x11},
{0x5e13, 0x11},
{0x5e14, 0x12},
{0x5e15, 0x12},
{0x5e16, 0x12},
{0x5e17, 0x12},
{0x5e18, 0x13},
{0x5e19, 0x13},
{0x5e1a, 0x13},
{0x5e1b, 0x14},
{0x5e1c, 0x14},
{0x5e1d, 0x14},
{0x5e1e, 0x15},
{0x5e1f, 0x15},
{0x5e20, 0x15},
{0x5e21, 0x16},
{0x5e22, 0x00},
{0x5e23, 0x02},
{0x5e26, 0x00},
{0x5e27, 0xff},
{0x5e29, 0x01},
{0x5e2a, 0x00},
{0x5e2c, 0x01},
{0x5e2d, 0x00},
{0x5e2f, 0x01},
{0x5e30, 0x00},
{0x5e32, 0x00},
{0x5e33, 0x80},
{0x5e34, 0x00},
{0x5e35, 0x00},
{0x5e36, 0x80},
{0x5e37, 0x00},
{0x5e38, 0x00},
{0x5e39, 0x80},
{0x5e3a, 0x00},
{0x5e3b, 0x00},
{0x5e3c, 0x80},
{0x5e3d, 0x00},
{0x5e3e, 0x00},
{0x5e3f, 0x80},
{0x5e40, 0x00},
{0x5e41, 0x00},
{0x5e42, 0x80},
{0x5e43, 0x00},
{0x5e44, 0x00},
{0x5e45, 0x80},
{0x5e46, 0x00},
{0x5e47, 0x00},
{0x5e48, 0x80},
{0x5e49, 0x00},
{0x5e4a, 0x00},
{0x5e4b, 0x80},
{0x5e4c, 0x00},
{0x5e4d, 0x00},
{0x5e4e, 0x80},
{0x5e50, 0x00},
{0x5e51, 0x80},
{0x5e53, 0x00},
{0x5e54, 0x80},
{0x5e56, 0x00},
{0x5e57, 0x40},
{0x5e59, 0x00},
{0x5e5a, 0x40},
{0x5e5c, 0x00},
{0x5e5d, 0x40},
{0x5e5f, 0x00},
{0x5e60, 0x40},
{0x5e62, 0x00},
{0x5e63, 0x40},
{0x5e65, 0x00},
{0x5e66, 0x40},
{0x5e68, 0x00},
{0x5e69, 0x40},
{0x5e6b, 0x00},
{0x5e6c, 0x40},
{0x5e6e, 0x00},
{0x5e6f, 0x40},
{0x5e71, 0x00},
{0x5e72, 0x40},
{0x5e74, 0x00},
{0x5e75, 0x40},
{0x5e77, 0x00},
{0x5e78, 0x40},
{0x5e7a, 0x00},
{0x5e7b, 0x40},
{0x5e7d, 0x00},
{0x5e7e, 0x40},
{0x5e80, 0x00},
{0x5e81, 0x40},
{0x5e83, 0x00},
{0x5e84, 0x40},
{0x5f00, 0x02},
{0x5f01, 0x08},
{0x5f02, 0x09},
{0x5f03, 0x0a},
{0x5f04, 0x0b},
{0x5f05, 0x0c},
{0x5f06, 0x0c},
{0x5f07, 0x0c},
{0x5f08, 0x0c},
{0x5f09, 0x0c},
{0x5f0a, 0x0d},
{0x5f0b, 0x0d},
{0x5f0c, 0x0d},
{0x5f0d, 0x0d},
{0x5f0e, 0x0d},
{0x5f0f, 0x0e},
{0x5f10, 0x0e},
{0x5f11, 0x0e},
{0x5f12, 0x0e},
{0x5f13, 0x0f},
{0x5f14, 0x0f},
{0x5f15, 0x10},
{0x5f16, 0x11},
{0x5f17, 0x11},
{0x5f18, 0x12},
{0x5f19, 0x12},
{0x5f1a, 0x13},
{0x5f1b, 0x13},
{0x5f1c, 0x14},
{0x5f1d, 0x14},
{0x5f1e, 0x16},
{0x5f1f, 0x16},
{0x5f20, 0x16},
{0x5f21, 0x08},
{0x5f22, 0x00},
{0x5f23, 0x01},
{0x5f26, 0x02},
{0x5f27, 0x00},
{0x5f29, 0x02},
{0x5f2a, 0x00},
{0x5f2c, 0x02},
{0x5f2d, 0x00},
{0x5f2f, 0x02},
{0x5f30, 0x00},
{0x5f32, 0x02},
{0x5f33, 0x00},
{0x5f34, 0x00},
{0x5f35, 0x02},
{0x5f36, 0x00},
{0x5f37, 0x00},
{0x5f38, 0x02},
{0x5f39, 0x00},
{0x5f3a, 0x00},
{0x5f3b, 0x02},
{0x5f3c, 0x00},
{0x5f3d, 0x00},
{0x5f3e, 0x02},
{0x5f3f, 0x00},
{0x5f40, 0x00},
{0x5f41, 0x02},
{0x5f42, 0x00},
{0x5f43, 0x00},
{0x5f44, 0x02},
{0x5f45, 0x00},
{0x5f46, 0x00},
{0x5f47, 0x04},
{0x5f48, 0x00},
{0x5f49, 0x00},
{0x5f4a, 0x04},
{0x5f4b, 0x00},
{0x5f4c, 0x00},
{0x5f4d, 0x04},
{0x5f4e, 0x00},
{0x5f50, 0x04},
{0x5f51, 0x00},
{0x5f53, 0x04},
{0x5f54, 0x00},
{0x5f56, 0x04},
{0x5f57, 0x00},
{0x5f59, 0x04},
{0x5f5a, 0x00},
{0x5f5c, 0x04},
{0x5f5d, 0x00},
{0x5f5f, 0x08},
{0x5f60, 0x00},
{0x5f62, 0x08},
{0x5f63, 0x00},
{0x5f65, 0x08},
{0x5f66, 0x00},
{0x5f68, 0x08},
{0x5f69, 0x00},
{0x5f6b, 0x08},
{0x5f6c, 0x00},
{0x5f6e, 0x10},
{0x5f6f, 0x00},
{0x5f71, 0x10},
{0x5f72, 0x00},
{0x5f74, 0x10},
{0x5f75, 0x00},
{0x5f77, 0x10},
{0x5f78, 0x00},
{0x5f7a, 0x20},
{0x5f7b, 0x00},
{0x5f7d, 0x20},
{0x5f7e, 0x00},
{0x5f80, 0x20},
{0x5f81, 0x00},
{0x5f83, 0x00},
{0x5f84, 0xff},
{0x5240, 0x0f},
{0x5243, 0x00},
{0x5244, 0x00},
{0x5245, 0x00},
{0x5246, 0x00},
{0x5247, 0x00},
{0x5248, 0x00},
{0x5249, 0x00},
{0x5440, 0x0f},
{0x5443, 0x00},
{0x5445, 0x00},
{0x5447, 0x00},
{0x5448, 0x00},
{0x5449, 0x00},
{0x5640, 0x0f},
{0x5642, 0x00},
{0x5643, 0x00},
{0x5644, 0x00},
{0x5645, 0x00},
{0x5646, 0x00},
{0x5647, 0x00},
{0x5649, 0x00},
{0x5840, 0x0f},
{0x5842, 0x00},
{0x5843, 0x00},
{0x5845, 0x00},
{0x5846, 0x00},
{0x5847, 0x00},
{0x5848, 0x00},
{0x5849, 0x00},
{0x4001, 0x2b},
{0x4008, 0x02},
{0x4009, 0x03},
{0x4018, 0x12},
{0x4022, 0x40},
{0x4023, 0x20},
{0x4026, 0x00},
{0x4027, 0x40},
{0x4028, 0x00},
{0x4029, 0x40},
{0x402a, 0x00},
{0x402b, 0x40},
{0x402c, 0x00},
{0x402d, 0x40},
{0x405e, 0x00},
{0x405f, 0x00},
{0x4060, 0x00},
{0x4061, 0x00},
{0x4062, 0x00},
{0x4063, 0x00},
{0x4064, 0x00},
{0x4065, 0x00},
{0x4066, 0x00},
{0x4067, 0x00},
{0x4068, 0x00},
{0x4069, 0x00},
{0x406a, 0x00},
{0x406b, 0x00},
{0x406c, 0x00},
{0x406d, 0x00},
{0x406e, 0x00},
{0x406f, 0x00},
{0x4070, 0x00},
{0x4071, 0x00},
{0x4072, 0x00},
{0x4073, 0x00},
{0x4074, 0x00},
{0x4075, 0x00},
{0x4076, 0x00},
{0x4077, 0x00},
{0x4078, 0x00},
{0x4079, 0x00},
{0x407a, 0x00},
{0x407b, 0x00},
{0x407c, 0x00},
{0x407d, 0x00},
{0x407e, 0xcc},
{0x407f, 0x18},
{0x4080, 0xff},
{0x4081, 0xff},
{0x4082, 0x01},
{0x4083, 0x53},
{0x4084, 0x01},
{0x4085, 0x2b},
{0x4086, 0x00},
{0x4087, 0xb3},
{0x4640, 0x40},
{0x4641, 0x11},
{0x4642, 0x0e},
{0x4643, 0xee},
{0x4646, 0x0f},
{0x4648, 0x00},
{0x4649, 0x03},
{0x4f00, 0x00},
{0x4f01, 0x00},
{0x4f02, 0x80},
{0x4f03, 0x2c},
{0x4f04, 0xf8},
{0x4d09, 0xff},
{0x4d09, 0xdf},
{0x5003, 0x7a},
{0x5b80, 0x08},
{0x5c00, 0x08},
{0x5c80, 0x00},
{0x5bbe, 0x12},
{0x5c3e, 0x12},
{0x5cbe, 0x12},
{0x5b8a, 0x80},
{0x5b8b, 0x80},
{0x5b8c, 0x80},
{0x5b8d, 0x80},
{0x5b8e, 0x60},
{0x5b8f, 0x80},
{0x5b90, 0x80},
{0x5b91, 0x80},
{0x5b92, 0x80},
{0x5b93, 0x20},
{0x5b94, 0x80},
{0x5b95, 0x80},
{0x5b96, 0x80},
{0x5b97, 0x20},
{0x5b98, 0x00},
{0x5b99, 0x80},
{0x5b9a, 0x40},
{0x5b9b, 0x20},
{0x5b9c, 0x00},
{0x5b9d, 0x00},
{0x5b9e, 0x80},
{0x5b9f, 0x00},
{0x5ba0, 0x00},
{0x5ba1, 0x00},
{0x5ba2, 0x00},
{0x5ba3, 0x00},
{0x5ba4, 0x00},
{0x5ba5, 0x00},
{0x5ba6, 0x00},
{0x5ba7, 0x00},
{0x5ba8, 0x02},
{0x5ba9, 0x00},
{0x5baa, 0x02},
{0x5bab, 0x76},
{0x5bac, 0x03},
{0x5bad, 0x08},
{0x5bae, 0x00},
{0x5baf, 0x80},
{0x5bb0, 0x00},
{0x5bb1, 0xc0},
{0x5bb2, 0x01},
{0x5bb3, 0x00},
{0x5c0a, 0x80},
{0x5c0b, 0x80},
{0x5c0c, 0x80},
{0x5c0d, 0x80},
{0x5c0e, 0x60},
{0x5c0f, 0x80},
{0x5c10, 0x80},
{0x5c11, 0x80},
{0x5c12, 0x60},
{0x5c13, 0x20},
{0x5c14, 0x80},
{0x5c15, 0x80},
{0x5c16, 0x80},
{0x5c17, 0x20},
{0x5c18, 0x00},
{0x5c19, 0x80},
{0x5c1a, 0x40},
{0x5c1b, 0x20},
{0x5c1c, 0x00},
{0x5c1d, 0x00},
{0x5c1e, 0x80},
{0x5c1f, 0x00},
{0x5c20, 0x00},
{0x5c21, 0x00},
{0x5c22, 0x00},
{0x5c23, 0x00},
{0x5c24, 0x00},
{0x5c25, 0x00},
{0x5c26, 0x00},
{0x5c27, 0x00},
{0x5c28, 0x02},
{0x5c29, 0x00},
{0x5c2a, 0x02},
{0x5c2b, 0x76},
{0x5c2c, 0x03},
{0x5c2d, 0x08},
{0x5c2e, 0x00},
{0x5c2f, 0x80},
{0x5c30, 0x00},
{0x5c31, 0xc0},
{0x5c32, 0x01},
{0x5c33, 0x00},
{0x5c8a, 0x80},
{0x5c8b, 0x80},
{0x5c8c, 0x80},
{0x5c8d, 0x80},
{0x5c8e, 0x80},
{0x5c8f, 0x80},
{0x5c90, 0x80},
{0x5c91, 0x80},
{0x5c92, 0x80},
{0x5c93, 0x60},
{0x5c94, 0x80},
{0x5c95, 0x80},
{0x5c96, 0x80},
{0x5c97, 0x60},
{0x5c98, 0x40},
{0x5c99, 0x80},
{0x5c9a, 0x80},
{0x5c9b, 0x80},
{0x5c9c, 0x40},
{0x5c9d, 0x00},
{0x5c9e, 0x80},
{0x5c9f, 0x80},
{0x5ca0, 0x80},
{0x5ca1, 0x20},
{0x5ca2, 0x00},
{0x5ca3, 0x80},
{0x5ca4, 0x80},
{0x5ca5, 0x00},
{0x5ca6, 0x00},
{0x5ca7, 0x00},
{0x5ca8, 0x01},
{0x5ca9, 0x00},
{0x5caa, 0x02},
{0x5cab, 0x00},
{0x5cac, 0x03},
{0x5cad, 0x08},
{0x5cae, 0x01},
{0x5caf, 0x00},
{0x5cb0, 0x02},
{0x5cb1, 0x00},
{0x5cb2, 0x03},
{0x5cb3, 0x08},
{0x5be7, 0x80},
{0x5bc9, 0x80},
{0x5bca, 0x80},
{0x5bcb, 0x80},
{0x5bcc, 0x80},
{0x5bcd, 0x80},
{0x5bce, 0x80},
{0x5bcf, 0x80},
{0x5bd0, 0x80},
{0x5bd1, 0x80},
{0x5bd2, 0x20},
{0x5bd3, 0x80},
{0x5bd4, 0x40},
{0x5bd5, 0x20},
{0x5bd6, 0x00},
{0x5bd7, 0x00},
{0x5bd8, 0x00},
{0x5bd9, 0x00},
{0x5bda, 0x00},
{0x5bdb, 0x00},
{0x5bdc, 0x00},
{0x5bdd, 0x00},
{0x5bde, 0x00},
{0x5bdf, 0x00},
{0x5be0, 0x00},
{0x5be1, 0x00},
{0x5be2, 0x00},
{0x5be3, 0x00},
{0x5be4, 0x00},
{0x5be5, 0x00},
{0x5be6, 0x00},
{0x5c49, 0x80},
{0x5c4a, 0x80},
{0x5c4b, 0x80},
{0x5c4c, 0x80},
{0x5c4d, 0x40},
{0x5c4e, 0x80},
{0x5c4f, 0x80},
{0x5c50, 0x80},
{0x5c51, 0x60},
{0x5c52, 0x20},
{0x5c53, 0x80},
{0x5c54, 0x80},
{0x5c55, 0x80},
{0x5c56, 0x20},
{0x5c57, 0x00},
{0x5c58, 0x80},
{0x5c59, 0x40},
{0x5c5a, 0x20},
{0x5c5b, 0x00},
{0x5c5c, 0x00},
{0x5c5d, 0x80},
{0x5c5e, 0x00},
{0x5c5f, 0x00},
{0x5c60, 0x00},
{0x5c61, 0x00},
{0x5c62, 0x00},
{0x5c63, 0x00},
{0x5c64, 0x00},
{0x5c65, 0x00},
{0x5c66, 0x00},
{0x5cc9, 0x80},
{0x5cca, 0x80},
{0x5ccb, 0x80},
{0x5ccc, 0x80},
{0x5ccd, 0x80},
{0x5cce, 0x80},
{0x5ccf, 0x80},
{0x5cd0, 0x80},
{0x5cd1, 0x80},
{0x5cd2, 0x60},
{0x5cd3, 0x80},
{0x5cd4, 0x80},
{0x5cd5, 0x80},
{0x5cd6, 0x60},
{0x5cd7, 0x40},
{0x5cd8, 0x80},
{0x5cd9, 0x80},
{0x5cda, 0x80},
{0x5cdb, 0x40},
{0x5cdc, 0x20},
{0x5cdd, 0x80},
{0x5cde, 0x80},
{0x5cdf, 0x80},
{0x5ce0, 0x20},
{0x5ce1, 0x00},
{0x5ce2, 0x80},
{0x5ce3, 0x80},
{0x5ce4, 0x80},
{0x5ce5, 0x00},
{0x5ce6, 0x00},
{0x5d74, 0x01},
{0x5d75, 0x00},
{0x5d1f, 0x81},
{0x5d11, 0x00},
{0x5d12, 0x10},
{0x5d13, 0x10},
{0x5d15, 0x05},
{0x5d16, 0x05},
{0x5d17, 0x05},
{0x5d08, 0x03},
{0x5d09, 0xb6},
{0x5d0a, 0x03},
{0x5d0b, 0xb6},
{0x5d18, 0x03},
{0x5d19, 0xb6},
{0x5d62, 0x01},
{0x5d40, 0x02},
{0x5d41, 0x01},
{0x5d63, 0x1f},
{0x5d64, 0x00},
{0x5d65, 0x80},
{0x5d56, 0x00},
{0x5d57, 0x20},
{0x5d58, 0x00},
{0x5d59, 0x20},
{0x5d5a, 0x00},
{0x5d5b, 0x0c},
{0x5d5c, 0x02},
{0x5d5d, 0x40},
{0x5d5e, 0x02},
{0x5d5f, 0x40},
{0x5d60, 0x03},
{0x5d61, 0x40},
{0x5d4a, 0x02},
{0x5d4b, 0x40},
{0x5d4c, 0x02},
{0x5d4d, 0x40},
{0x5d4e, 0x02},
{0x5d4f, 0x40},
{0x5d50, 0x18},
{0x5d51, 0x80},
{0x5d52, 0x18},
{0x5d53, 0x80},
{0x5d54, 0x18},
{0x5d55, 0x80},
{0x5d46, 0x20},
{0x5d47, 0x00},
{0x5d48, 0x22},
{0x5d49, 0x00},
{0x5d42, 0x20},
{0x5d43, 0x00},
{0x5d44, 0x22},
{0x5d45, 0x00},
{0x5004, 0x1e},
{0x4221, 0x03},

// ugh, in here twice
//{0x380e, 0x02},
//{0x380f, 0xae},

{0x380c, 0x04},
{0x380d, 0x47},
{0x384c, 0x02},
{0x384d, 0x0d},
{0x388c, 0x02},
{0x388d, 0x2b},
{0x3501, 0x01},
{0x3502, 0xc8},
{0x3541, 0x01},
{0x3542, 0xc8},
{0x35c1, 0x00},
{0x35c2, 0x01},
{0x420e, 0x66},
{0x420f, 0x5d},
{0x4210, 0xa8},
{0x4211, 0x55},
{0x507a, 0x5f},
{0x507b, 0x46},
{0x4f00, 0x00},
{0x4f01, 0x01},
{0x4f02, 0x80},
{0x4f04, 0x2c},

  /*{0x103, 1},

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
  {0x329, 0x1},                 // pll2_divdac(/2)[1:0]*/

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
  /*{0x3501, 0x1f},

  // https://github.com/hanwckf/linux-rk3328-box/blob/master/drivers/media/i2c/ov4689.c
  {0x481f, 0x40},
  {0x4829, 0x78},
  {0x4837, 0x10},

  // https://patchwork.kernel.org/project/linux-media/patch/20210813081845.26619-1-arec.kao@intel.com/#24384123
  {0x484b, 0x01},

  // MIPI PCLK period (broken with 0xe stock, MIPI/PHY = 1440/180 = 8)
  // this makes some output happen
  //{0x4837, 0x8},
  //{0x4837, 0xe},

  // data type
  {0x4814, 0x2B},
  {0x4815, 0x2B},
  {0x4816, 0x2B},*/

  // disable ISP
  //{0x5000, 0x0},

  //{0x484b, 0x3f},

  //{0x3508, 0xf},

  // MIPI (10-bit mode)
  //{0x3022, 0x01},

  // MIPI_CORE_37 = 0xe (MIPI PCLK period)

  // output size
  // 1928
  //{0x3808, 0x07}, {0x3809, 0x88},
  //{0x380A, 0x00}, {0x380B, 0xb8},
  //{0x380c, 0x2a}, {0x380d, 0x18},

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
