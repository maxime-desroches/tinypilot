#include "acado_common.h"
#include "acado_auxiliary_functions.h"

#include <stdio.h>

#define NX          ACADO_NX  /* Number of differential state variables.  */
#define NXA         ACADO_NXA /* Number of algebraic variables. */
#define NU          ACADO_NU  /* Number of control inputs. */
#define NOD         ACADO_NOD  /* Number of online data values. */

#define NY          ACADO_NY  /* Number of measurements/references on nodes 0..N - 1. */
#define NYN         ACADO_NYN /* Number of measurements/references on node N. */

#define N           ACADO_N   /* Number of intervals in the horizon. */

ACADOvariables acadoVariables;
ACADOworkspace acadoWorkspace;

typedef struct {
  double x, y, psi, dpsi, ddpsi;
} state_t;


typedef struct {
  double x[N+1];
  double y[N+1];
  double psi[N+1];
  double dpsi[N+1];
  double ddpsi[N];
  double cost;
} log_t;

void init_weights(double pathCost, double yawRateCost, double steerRateCost){
  int    i;
  for (i = 0; i < N; i++) {
    // Setup diagonal entries
    acadoVariables.W[NY*NY*i + (NY+1)*0] = pathCost;
    acadoVariables.W[NY*NY*i + (NY+1)*1] = yawRateCost;
    acadoVariables.W[NY*NY*i + (NY+1)*2] = steerRateCost;
  }
  acadoVariables.WN[(NYN+1)*0] = pathCost;
}

void init(double pathCost, double yawRateCost, double steerRateCost){
  acado_initializeSolver();
  int    i;

  /* Initialize the states and controls. */
  for (i = 0; i < NX * (N + 1); ++i)  acadoVariables.x[ i ] = 0.0;
  for (i = 0; i < NU * N; ++i)  acadoVariables.u[ i ] = 0.1 ;

  /* Initialize the measurements/reference. */
  for (i = 0; i < NY * N; ++i)  acadoVariables.y[ i ] = 0.0;
  for (i = 0; i < NYN; ++i)  acadoVariables.yN[ i ] = 0.0;

  /* MPC: initialize the current state feedback. */
  for (i = 0; i < NX; ++i) acadoVariables.x0[ i ] = 0.0;

  init_weights(pathCost, yawRateCost, steerRateCost);
}

int run_mpc(state_t * x0, log_t * solution, double v_poly[4],
             double target_y[N+1], double target_dpsi[N+1],
             double target_ddpsi[N+1]){

  int    i;

  for (i = 0; i <= NOD * N; i+= NOD){
    acadoVariables.od[i+0] = v_poly[0];
    acadoVariables.od[i+1] = v_poly[1];
    acadoVariables.od[i+2] = v_poly[2];
    acadoVariables.od[i+3] = v_poly[3];
  }
  for (i = 0; i < N; i+= 1){
    acadoVariables.y[NY*i + 0] = target_y[i];
    acadoVariables.y[NY*i + 1] = target_dpsi[i];
    acadoVariables.y[NY*i + 2] = target_ddpsi[i];
  }
  acadoVariables.yN[0] = target_y[N];

  acadoVariables.x0[0] = x0->x;
  acadoVariables.x0[1] = x0->y;
  acadoVariables.x0[2] = x0->psi;
  acadoVariables.x0[3] = x0->dpsi;


  acado_preparationStep();
  acado_feedbackStep();

  /* printf("lat its: %d\n", acado_getNWSR());  // n iterations
  printf("Objective: %.6f\n", acado_getObjective());  // solution cost */

  for (i = 0; i <= N; i++){
    solution->x[i] = acadoVariables.x[i*NX];
    solution->y[i] = acadoVariables.x[i*NX+1];
    solution->psi[i] = acadoVariables.x[i*NX+2];
    solution->dpsi[i] = acadoVariables.x[i*NX+3];
    if (i < N){
      solution->ddpsi[i] = acadoVariables.u[i];
    }
  }
  solution->cost = acado_getObjective();

  // Dont shift states here. Current solution is closer to next timestep than if
  // we use the old solution as a starting point
  //acado_shiftStates(2, 0, 0);
  //acado_shiftControls( 0 );

  return acado_getNWSR();
}
