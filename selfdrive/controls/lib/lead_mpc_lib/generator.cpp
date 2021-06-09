#include <acado_code_generation.hpp>
#include "selfdrive/common/modeldata.h"

using namespace std;

#define G 9.81
#define TR 1.8

#define RW(v_ego, v_l) (v_ego * TR - (v_l - v_ego) * TR + v_ego*v_ego/(2*G) - v_l*v_l / (2*G))
#define NORM_RW_ERROR(v_ego, v_l, p) ((RW(v_ego, v_l) + 4.0 - p)/(sqrt(v_ego + 0.5) + 0.1))

int main( )
{
  USING_NAMESPACE_ACADO


  DifferentialEquation f;

  DifferentialState x_ego, v_ego, a_ego;
  OnlineData x_l, v_l;

  Control j_ego;

  auto desired = 4.0 + RW(v_ego, v_l);
  auto d_l = x_l - x_ego;

  // Equations of motion
  f << dot(x_ego) == v_ego;
  f << dot(v_ego) == a_ego;
  f << dot(a_ego) == j_ego;

  // Running cost
  Function h;
  h << exp(0.3 * NORM_RW_ERROR(v_ego, v_l, d_l)) - 1;
  h << (d_l - desired) / (0.05 * v_ego + 0.5);
  h << a_ego * (0.1 * v_ego + 1.0);
  h << j_ego * (0.1 * v_ego + 1.0);

  // Weights are defined in mpc.
  BMatrix Q(4,4); Q.setAll(true);

  // Terminal cost
  Function hN;
  hN << exp(0.3 * NORM_RW_ERROR(v_ego, v_l, d_l)) - 1;
  hN << (d_l - desired) / (0.05 * v_ego + 0.5);
  hN << a_ego;

  // Weights are defined in mpc.
  BMatrix QN(3,3); QN.setAll(true);

  double T_IDXS_ARR[LON_MPC_N + 1];
  memcpy(T_IDXS_ARR, T_IDXS, (LON_MPC_N + 1) * sizeof(double));
  Grid times(LON_MPC_N + 1, T_IDXS_ARR);
  OCP ocp(times);
  ocp.subjectTo(f);

  ocp.minimizeLSQ(Q, h);
  ocp.minimizeLSQEndTerm(QN, hN);

  ocp.subjectTo( 1e-3 <= v_ego);
  ocp.setNOD(2);

  OCPexport mpc(ocp);
  mpc.set( HESSIAN_APPROXIMATION, GAUSS_NEWTON );
  mpc.set( DISCRETIZATION_TYPE, MULTIPLE_SHOOTING );
  mpc.set( INTEGRATOR_TYPE, INT_RK4 );
  mpc.set( NUM_INTEGRATOR_STEPS, 2500);
  mpc.set( MAX_NUM_QP_ITERATIONS, 500);
  mpc.set( CG_USE_VARIABLE_WEIGHTING_MATRIX, YES);

  mpc.set( SPARSE_QP_SOLUTION, CONDENSING );
  mpc.set( QP_SOLVER, QP_QPOASES );
  mpc.set( HOTSTART_QP, YES );
  mpc.set( GENERATE_TEST_FILE, NO);
  mpc.set( GENERATE_MAKE_FILE, NO );
  mpc.set( GENERATE_MATLAB_INTERFACE, NO );
  mpc.set( GENERATE_SIMULINK_INTERFACE, NO );

  if (mpc.exportCode( "lib_mpc_export" ) != SUCCESSFUL_RETURN)
    exit( EXIT_FAILURE );

  mpc.printDimensionsQP( );

  return EXIT_SUCCESS;
}
