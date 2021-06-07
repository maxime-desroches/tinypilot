import os

from cffi import FFI
from common.ffi_wrapper import suffix

mpc_dir = os.path.dirname(os.path.abspath(__file__))
libmpc_fn = os.path.join(mpc_dir, "libmpc"+suffix())

ffi = FFI()
ffi.cdef("""
const int MPC_N = 16;

typedef struct {
    double x, y, psi, curvature, curvature_rate;
} state_t;

typedef struct {
    double x[MPC_N+1];
    double y[MPC_N+1];
    double psi[MPC_N+1];
    double curvature[MPC_N+1];
    double curvature_rate[MPC_N];
    double cost;
} log_t;

void init();
void set_weights(double pathCost, double headingCost, double steerRateCost);
int run_mpc(state_t * x0, log_t * solution,
             double v_ego, double rotation_radius,
             double target_y[MPC_N+1], double target_psi[MPC_N+1]);
""")

libmpc = ffi.dlopen(libmpc_fn)
