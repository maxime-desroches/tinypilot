/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) lead_cost_y_0_fun_jac_ut_xt_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_sq CASADI_PREFIX(sq)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

casadi_real casadi_sq(casadi_real x) { return x*x;}

static const casadi_int casadi_s0[7] = {3, 1, 0, 3, 0, 1, 2};
static const casadi_int casadi_s1[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s2[6] = {2, 1, 0, 2, 0, 1};
static const casadi_int casadi_s3[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s4[15] = {4, 4, 0, 2, 4, 6, 8, 1, 2, 1, 2, 2, 3, 0, 2};

/* lead_cost_y_0_fun_jac_ut_xt:(i0[3],i1,i2[2])->(o0[4],o1[4x4,8nz]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a2, a3, a4, a5, a6, a7, a8, a9;
  a0=2.9999999999999999e-01;
  a1=1.8000000000000000e+00;
  a2=arg[0]? arg[0][1] : 0;
  a3=(a1*a2);
  a4=arg[2]? arg[2][1] : 0;
  a5=(a4-a2);
  a1=(a1*a5);
  a3=(a3-a1);
  a1=casadi_sq(a2);
  a5=1.9620000000000001e+01;
  a1=(a1/a5);
  a3=(a3+a1);
  a4=casadi_sq(a4);
  a4=(a4/a5);
  a3=(a3-a4);
  a4=4.;
  a5=(a3+a4);
  a1=arg[2]? arg[2][0] : 0;
  a6=arg[0]? arg[0][0] : 0;
  a7=(a1-a6);
  a5=(a5-a7);
  a7=5.0000000000000000e-01;
  a8=(a2+a7);
  a8=sqrt(a8);
  a9=1.0000000000000001e-01;
  a10=(a8+a9);
  a5=(a5/a10);
  a11=(a0*a5);
  a11=exp(a11);
  a12=1.;
  a13=(a11-a12);
  if (res[0]!=0) res[0][0]=a13;
  a1=(a1-a6);
  a3=(a3+a4);
  a1=(a1-a3);
  a3=5.0000000000000003e-02;
  a4=(a3*a2);
  a4=(a4+a7);
  a1=(a1/a4);
  if (res[0]!=0) res[0][1]=a1;
  a7=arg[0]? arg[0][2] : 0;
  a6=(a9*a2);
  a6=(a6+a12);
  a13=(a7*a6);
  if (res[0]!=0) res[0][2]=a13;
  a13=arg[1]? arg[1][0] : 0;
  a14=(a9*a2);
  a14=(a14+a12);
  a12=(a13*a14);
  if (res[0]!=0) res[0][3]=a12;
  a12=(a0/a10);
  a12=(a11*a12);
  if (res[1]!=0) res[1][0]=a12;
  a12=3.6000000000000001e+00;
  a15=5.0968399592252800e-02;
  a2=(a2+a2);
  a15=(a15*a2);
  a12=(a12+a15);
  a15=(a12/a10);
  a5=(a5/a10);
  a8=(a8+a8);
  a5=(a5/a8);
  a15=(a15-a5);
  a0=(a0*a15);
  a11=(a11*a0);
  if (res[1]!=0) res[1][1]=a11;
  a11=(1./a4);
  a11=(-a11);
  if (res[1]!=0) res[1][2]=a11;
  a12=(a12/a4);
  a1=(a1/a4);
  a3=(a3*a1);
  a12=(a12+a3);
  a12=(-a12);
  if (res[1]!=0) res[1][3]=a12;
  a7=(a9*a7);
  if (res[1]!=0) res[1][4]=a7;
  if (res[1]!=0) res[1][5]=a6;
  if (res[1]!=0) res[1][6]=a14;
  a9=(a9*a13);
  if (res[1]!=0) res[1][7]=a9;
  return 0;
}

CASADI_SYMBOL_EXPORT int lead_cost_y_0_fun_jac_ut_xt(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int lead_cost_y_0_fun_jac_ut_xt_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int lead_cost_y_0_fun_jac_ut_xt_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void lead_cost_y_0_fun_jac_ut_xt_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int lead_cost_y_0_fun_jac_ut_xt_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void lead_cost_y_0_fun_jac_ut_xt_release(int mem) {
}

CASADI_SYMBOL_EXPORT void lead_cost_y_0_fun_jac_ut_xt_incref(void) {
}

CASADI_SYMBOL_EXPORT void lead_cost_y_0_fun_jac_ut_xt_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int lead_cost_y_0_fun_jac_ut_xt_n_in(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_int lead_cost_y_0_fun_jac_ut_xt_n_out(void) { return 2;}

CASADI_SYMBOL_EXPORT casadi_real lead_cost_y_0_fun_jac_ut_xt_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* lead_cost_y_0_fun_jac_ut_xt_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* lead_cost_y_0_fun_jac_ut_xt_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* lead_cost_y_0_fun_jac_ut_xt_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* lead_cost_y_0_fun_jac_ut_xt_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s3;
    case 1: return casadi_s4;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int lead_cost_y_0_fun_jac_ut_xt_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 3;
  if (sz_res) *sz_res = 2;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
