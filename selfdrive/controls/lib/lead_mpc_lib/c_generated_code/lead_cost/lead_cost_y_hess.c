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
  #define CASADI_PREFIX(ID) lead_cost_y_hess_ ## ID
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
static const casadi_int casadi_s2[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s3[6] = {2, 1, 0, 2, 0, 1};
static const casadi_int casadi_s4[15] = {4, 4, 0, 1, 3, 7, 8, 2, 1, 2, 0, 1, 2, 3, 2};

/* lead_cost_y_hess:(i0[3],i1,i2[4],i3[2])->(o0[4x4,8nz]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a3, a4, a5, a6, a7, a8, a9;
  a0=1.0000000000000001e-01;
  a1=arg[2]? arg[2][3] : 0;
  a2=(a0*a1);
  if (res[0]!=0) res[0][0]=a2;
  a2=2.9999999999999999e-01;
  a3=arg[2]? arg[2][0] : 0;
  a4=1.8000000000000000e+00;
  a5=arg[0]? arg[0][1] : 0;
  a6=(a4*a5);
  a7=arg[3]? arg[3][1] : 0;
  a8=(a7-a5);
  a8=(a4*a8);
  a6=(a6-a8);
  a8=casadi_sq(a5);
  a9=1.9620000000000001e+01;
  a8=(a8/a9);
  a6=(a6+a8);
  a7=casadi_sq(a7);
  a7=(a7/a9);
  a6=(a6-a7);
  a7=4.;
  a9=(a6+a7);
  a8=arg[3]? arg[3][0] : 0;
  a10=arg[0]? arg[0][0] : 0;
  a11=(a8-a10);
  a9=(a9-a11);
  a11=5.0000000000000000e-01;
  a12=(a5+a11);
  a12=sqrt(a12);
  a13=(a12+a0);
  a9=(a9/a13);
  a14=(a2*a9);
  a14=exp(a14);
  a15=(a2/a13);
  a15=(a14*a15);
  a15=(a3*a15);
  a15=(a2*a15);
  a16=(a15/a13);
  if (res[0]!=0) res[0][1]=a16;
  a17=5.0000000000000003e-02;
  a18=arg[2]? arg[2][1] : 0;
  a19=(a17*a5);
  a19=(a19+a11);
  a11=(1./a19);
  a11=(a11/a19);
  a11=(a18*a11);
  a11=(a17*a11);
  a20=(a14*a3);
  a20=(a2*a20);
  a21=(1./a13);
  a21=casadi_sq(a21);
  a21=(a20*a21);
  a22=(a9/a13);
  a15=(a22*a15);
  a21=(a21+a15);
  a15=(a12+a12);
  a21=(a21/a15);
  a11=(a11-a21);
  a21=(a5+a5);
  a23=5.0968399592252800e-02;
  a24=(a23*a16);
  a24=(a21*a24);
  a11=(a11+a24);
  a24=(a4*a16);
  a11=(a11+a24);
  a16=(a4*a16);
  a11=(a11+a16);
  if (res[0]!=0) res[0][2]=a11;
  a1=(a0*a1);
  if (res[0]!=0) res[0][3]=a1;
  a1=3.6000000000000001e+00;
  a5=(a5+a5);
  a5=(a23*a5);
  a1=(a1+a5);
  a5=(a1/a13);
  a9=(a9/a13);
  a12=(a12+a12);
  a9=(a9/a12);
  a5=(a5-a9);
  a9=(a2*a5);
  a14=(a14*a9);
  a3=(a3*a14);
  a2=(a2*a3);
  a3=(a2/a13);
  a14=(a20/a13);
  a9=(a14/a13);
  a9=(a9/a12);
  a3=(a3-a9);
  a9=(a18/a19);
  a11=(a9/a19);
  a11=(a17*a11);
  a16=(a3+a11);
  if (res[0]!=0) res[0][4]=a16;
  a1=(a1/a19);
  a8=(a8-a10);
  a6=(a6+a7);
  a8=(a8-a6);
  a8=(a8/a19);
  a6=(a8/a19);
  a6=(a17*a6);
  a1=(a1+a6);
  a1=(a1/a19);
  a8=(a8/a19);
  a8=(a8/a19);
  a8=(a17*a8);
  a1=(a1+a8);
  a18=(a18*a1);
  a17=(a17*a18);
  a5=(a5/a13);
  a13=(a22/a13);
  a13=(a13/a12);
  a5=(a5-a13);
  a5=(a20*a5);
  a2=(a22*a2);
  a5=(a5+a2);
  a5=(a5/a15);
  a22=(a22*a20);
  a22=(a22/a15);
  a22=(a22/a15);
  a12=(1./a12);
  a12=(a12+a12);
  a22=(a22*a12);
  a5=(a5-a22);
  a17=(a17-a5);
  a5=2.;
  a14=(a14-a9);
  a14=(a23*a14);
  a5=(a5*a14);
  a3=(a3+a11);
  a23=(a23*a3);
  a21=(a21*a23);
  a5=(a5+a21);
  a17=(a17+a5);
  a5=(a4*a3);
  a17=(a17+a5);
  a4=(a4*a3);
  a17=(a17+a4);
  if (res[0]!=0) res[0][5]=a17;
  a17=arg[2]? arg[2][2] : 0;
  a4=(a0*a17);
  if (res[0]!=0) res[0][6]=a4;
  a0=(a0*a17);
  if (res[0]!=0) res[0][7]=a0;
  return 0;
}

CASADI_SYMBOL_EXPORT int lead_cost_y_hess(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int lead_cost_y_hess_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int lead_cost_y_hess_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void lead_cost_y_hess_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int lead_cost_y_hess_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void lead_cost_y_hess_release(int mem) {
}

CASADI_SYMBOL_EXPORT void lead_cost_y_hess_incref(void) {
}

CASADI_SYMBOL_EXPORT void lead_cost_y_hess_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int lead_cost_y_hess_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int lead_cost_y_hess_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real lead_cost_y_hess_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* lead_cost_y_hess_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* lead_cost_y_hess_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* lead_cost_y_hess_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    case 3: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* lead_cost_y_hess_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s4;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int lead_cost_y_hess_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 4;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
