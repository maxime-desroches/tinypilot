#pragma OPENCL EXTENSION cl_khr_fp16 : enable

const half black_level = 42.0;

const __constant half3 color_correction[3] = {
  // post wb CCM
  (half3)(1.44602146, -0.24727126, -0.0403062),
  (half3)(-0.37658179, 1.26329038, -0.45978396),
  (half3)(-0.06943967, -0.01601912, 1.50009016),
};

half3 color_correct(half3 rgb) {
  half3 ret = (0,0,0);
  rgb = 2*rgb / (1.0f + 2*rgb); // reinhard
  ret += rgb.x * color_correction[0];
  ret += rgb.y * color_correction[1];
  ret += rgb.z * color_correction[2];
  ret = max(0.0h, min(1.0h, ret));
  return ret;
}

half val_from_10(const uchar * source, int gx, int gy) {
  // parse 10bit
  int start = gy * FRAME_STRIDE + (5 * (gx / 4));
  int offset = gx % 4;
  uint major = (uint)source[start + offset] << 2;
  uint minor = (source[start + 4] >> (2 * offset)) & 3;
  half pv = (half)(major + minor);

  // normalize
  pv = max(0.0h, pv - black_level);
  pv /= (1024.0f - black_level);

  // correct vignetting
  if (CAM_NUM == 1) { // fcamera
    gx = (gx - RGB_WIDTH/2);
    gy = (gy - RGB_HEIGHT/2);
    half r = pow(gx*gx + gy*gy, 0.825);
    half s = 1 / (1-0.00000733*r);
    pv = s * pv;
  }

  return pv;
}

half fabs_diff(half x, half y) {
  return fabs(x-y);
}

half phi(half x) {
  // detection funtion
  if (x > 1) {
    return 1 / x;
  } else {
    return 2 - x;
  }
}

__kernel void debayer10(const __global uchar * in,
                        __global uchar * out,
                        __local half * cached
                       )
{
  const int x_global = get_global_id(0);
  const int y_global = get_global_id(1);

  const int localRowLen = 2 + get_local_size(0); // 2 padding
  const int x_local = get_local_id(0); // 0-15
  const int y_local = get_local_id(1); // 0-15
  const int localOffset = (y_local + 1) * localRowLen + x_local + 1; // max 18x18-1

  int out_idx = 3 * x_global + 3 * y_global * RGB_WIDTH;

  half pv = val_from_10(in, x_global, y_global);
  cached[localOffset] = pv;

  // don't care
  if (x_global < 1 || x_global >= RGB_WIDTH - 1 || y_global < 1 || y_global >= RGB_HEIGHT - 1) {
    barrier(CLK_LOCAL_MEM_FENCE);
    return;
  }

  // cache padding
  int localColOffset = -1;
  int globalColOffset = -1;

  // cache padding
  if (x_local < 1) {
    localColOffset = x_local;
    globalColOffset = -1;
    cached[(y_local + 1) * localRowLen + x_local] = val_from_10(in, x_global-1, y_global);
  } else if (x_local >= get_local_size(0) - 1) {
    localColOffset = x_local + 2;
    globalColOffset = 1;
    cached[localOffset + 1] = val_from_10(in, x_global+1, y_global);
  }

  if (y_local < 1) {
    cached[y_local * localRowLen + x_local + 1] = val_from_10(in, x_global, y_global-1);
    if (localColOffset != -1) {
      cached[y_local * localRowLen + localColOffset] = val_from_10(in, x_global+globalColOffset, y_global-1);
    }
  } else if (y_local >= get_local_size(1) - 1) {
    cached[(y_local + 2) * localRowLen + x_local + 1] = val_from_10(in, x_global, y_global+1);
    if (localColOffset != -1) {
      cached[(y_local + 2) * localRowLen + localColOffset] = val_from_10(in, x_global+globalColOffset, y_global+1);
    }
  }

  // sync
  barrier(CLK_LOCAL_MEM_FENCE);

  half d1 = cached[localOffset - localRowLen - 1];
  half d2 = cached[localOffset - localRowLen + 1];
  half d3 = cached[localOffset + localRowLen - 1];
  half d4 = cached[localOffset + localRowLen + 1];
  half n1 = cached[localOffset - localRowLen];
  half n2 = cached[localOffset + 1];
  half n3 = cached[localOffset + localRowLen];
  half n4 = cached[localOffset - 1];

  half3 rgb;

  // main
  if (x_global % 2 == 0) {
    if (y_global % 2 == 0) {
      rgb.y = pv; // G1(R)
      half k1 = phi(fabs_diff(d1, pv) + fabs_diff(d2, pv));
      half k2 = phi(fabs_diff(d2, pv) + fabs_diff(d4, pv));
      half k3 = phi(fabs_diff(d3, pv) + fabs_diff(d4, pv));
      half k4 = phi(fabs_diff(d1, pv) + fabs_diff(d3, pv));
      // R_G1
      rgb.x = (k2*n2+k4*n4)/(k2+k4);
      // B_G1
      rgb.z = (k1*n1+k3*n3)/(k1+k3);
    } else {
      rgb.z = pv; // B
      half k1 = phi(fabs_diff(d1, d3) + fabs_diff(d2, d4));
      half k2 = phi(fabs_diff(n1, n4) + fabs_diff(n2, n3));
      half k3 = phi(fabs_diff(d1, d2) + fabs_diff(d3, d4));
      half k4 = phi(fabs_diff(n1, n2) + fabs_diff(n3, n4));
      // G_B
      rgb.y = (k1*(n1+n3)/2+k3*(n2+n4)/2)/(k1+k3);
      // R_B
      rgb.x = (k2*(d2+d3)/2+k4*(d1+d4)/2)/(k2+k4);
    }
  } else {
    if (y_global % 2 == 0) {
      rgb.x = pv; // R
      half k1 = phi(fabs_diff(d1, d3) + fabs_diff(d2, d4));
      half k2 = phi(fabs_diff(n1, n4) + fabs_diff(n2, n3));
      half k3 = phi(fabs_diff(d1, d2) + fabs_diff(d3, d4));
      half k4 = phi(fabs_diff(n1, n2) + fabs_diff(n3, n4));
      // G_R
      rgb.y = (k1*(n1+n3)/2+k3*(n2+n4)/2)/(k1+k3);
      // B_R
      rgb.z = (k2*(d2+d3)/2+k4*(d1+d4)/2)/(k2+k4);
    } else {
      rgb.y = pv; // G2(B)
      half k1 = phi(fabs_diff(d1, pv) + fabs_diff(d2, pv));
      half k2 = phi(fabs_diff(d2, pv) + fabs_diff(d4, pv));
      half k3 = phi(fabs_diff(d3, pv) + fabs_diff(d4, pv));
      half k4 = phi(fabs_diff(d1, pv) + fabs_diff(d3, pv));
      // R_G2
      rgb.x = (k1*n1+k3*n3)/(k1+k3);
      // B_G2
      rgb.z = (k2*n2+k4*n4)/(k2+k4);
    }
  }

  rgb = color_correct(rgb);

  out[out_idx + 0] = (uchar)(255.0f * rgb.z);
  out[out_idx + 1] = (uchar)(255.0f * rgb.y);
  out[out_idx + 2] = (uchar)(255.0f * rgb.x);

}
