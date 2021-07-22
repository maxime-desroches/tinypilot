#!/usr/bin/env python3
import numpy as np

from selfdrive.swaglog import cloudlog
from common.realtime import sec_since_boot
from selfdrive.controls.lib.lateral_mpc.lat_mpc import generate_code, get_default_simulink_options
from selfdrive.controls.lib.drive_helpers import LON_MPC_N as N
from selfdrive.modeld.constants import T_IDXS
import os
import sys
from common.basedir import BASEDIR

from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import SX, vertcat

# TODO: clean this up
acados_path = os.path.join(BASEDIR, "phonelibs/acados/x86_64")
os.environ["TERA_PATH"] = os.path.join(acados_path, "t_renderer")
sys.path.append(os.path.join(BASEDIR, "pyextra"))



LON_MPC_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_FILE = "acados_ocp_long.json"


def gen_long_model():
  model = AcadosModel()
  model.name = 'long'

  # set up states & controls
  x_ego = SX.sym('x_ego')
  v_ego = SX.sym('v_ego')
  a_ego = SX.sym('a_ego')
  model.x = vertcat(x_ego, v_ego, a_ego)

  # controls
  j_ego = SX.sym('j_ego')
  model.u = vertcat(j_ego)

  # xdot
  x_ego_dot = SX.sym('x_ego_dot')
  v_ego_dot = SX.sym('v_ego_dot')
  a_ego_dot = SX.sym('a_ego_dot')
  model.xdot = vertcat(x_ego_dot, v_ego_dot, a_ego_dot)

  # dynamics model
  f_expl = vertcat(v_ego, a_ego, j_ego)
  model.f_impl_expr = model.xdot - f_expl
  model.f_expl_expr = f_expl
  return model


def gen_long_mpc_solver():
  ocp = AcadosOcp()
  ocp.model = gen_long_model()

  Tf = np.array(T_IDXS)[N]

  # set dimensions
  ocp.dims.N = N

  # set cost module
  ocp.cost.cost_type = 'NONLINEAR_LS'
  ocp.cost.cost_type_e = 'NONLINEAR_LS'

  QR = np.diag([0.0, 0.0, 0.0, 0.0])
  Q = np.diag([0.0, 0.0, 0.0])

  ocp.cost.W = QR
  ocp.cost.W_e = Q

  x_ego, v_ego, a_ego = ocp.model.x[0], ocp.model.x[1], ocp.model.x[2]
  j_ego = ocp.model.u[0]

  ocp.cost.yref = np.zeros((4, ))
  ocp.cost.yref_e = np.zeros((3, ))
  # TODO hacky weights to keep behavior the same
  ocp.model.cost_y_expr = vertcat(x_ego, v_ego, a_ego, j_ego)
  ocp.model.cost_y_expr_e = vertcat(x_ego, v_ego, a_ego)

  # set constraints
  ocp.constraints.constr_type = 'BGH'
  ocp.constraints.idxbx = np.array([0, 1,2])
  ocp.constraints.lbx = np.array([0., 0, -1.2])
  ocp.constraints.ubx = np.array([10000, 100., 1.2])
  ocp.constraints.Jsbx = np.eye(3)
  x0 = np.array([0.0, 0.0, 0.0])
  ocp.constraints.x0 = x0

  l2_penalty = 10000.0
  l1_penalty = 0.0
  weights = np.array([0.0, 1e4, 1e4])
  ocp.cost.Zl = l2_penalty * weights
  ocp.cost.Zu = l2_penalty * weights
  ocp.cost.zl = l1_penalty * weights
  ocp.cost.zu = l1_penalty * weights

  ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
  ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
  ocp.solver_options.integrator_type = 'ERK'
  ocp.solver_options.nlp_solver_type = 'SQP_RTI'
  ocp.solver_options.qp_solver_iter_max = 100
  ocp.solver_options.qp_solver_cond_N = N

  # set prediction horizon
  ocp.solver_options.tf = Tf
  ocp.solver_options.shooting_nodes = np.array(T_IDXS)[:N+1]

  ocp.code_export_directory = os.path.join(LON_MPC_DIR, "c_generated_code")
  return ocp


class LongitudinalMpc():
  def __init__(self):
    ocp = gen_long_mpc_solver()
    self.solver = AcadosOcpSolver(ocp, json_file=JSON_FILE, build=False,
                                  simulink_opts=get_default_simulink_options())
    self.x_sol = np.zeros((N+1, 3))
    self.u_sol = np.zeros((N))
    self.set_weights()

    self.v_solution = [0.0 for i in range(len(T_IDXS))]
    self.a_solution = [0.0 for i in range(len(T_IDXS))]
    self.j_solution = [0.0 for i in range(len(T_IDXS)-1)]
    self.last_cloudlog_t = 0
    self.status = True

  def set_weights(self):
    W = np.diag([0.0, 1.0, 0.0, 50.0])
    Ws = np.tile(W[None], reps=(N,1,1))
    self.solver.cost_set_slice(0, N, 'W', Ws, api='old')
    #TODO hacky weights to keep behavior the same
    self.solver.cost_set(N, 'W', (3/20.)*W[:3,:3])

  def set_accel_limits(self, min_a, max_a):
    self.min_a = min_a
    self.max_a = max_a

  def set_cur_state(self, v, a):
    self.x0 = np.array([0, v, a])
    self.solver.constraints_set(0, "lbx", self.x0)
    self.solver.constraints_set(0, "ubx", self.x0)

  def update(self, carstate, model, v_cruise):
    v_cruise_clipped = np.clip(v_cruise, self.x0[1] - 10., self.x0[1] + 10.0)
    poss = v_cruise_clipped * np.array(T_IDXS[:N+1])
    speeds = v_cruise_clipped * np.ones(N+1)
    accels = np.zeros(N+1)
    yref = np.column_stack([poss, speeds, accels, np.zeros(N+1)])
    #p = np.array([self.min_a, self.max_a])
    #for i in range(N):
    #  self.solver.set(i, "p", p)
    for i in range(1,N):
      self.solver.constraints_set(i, "lbx", np.array([0.0, 0.0,self.min_a]))
      self.solver.constraints_set(i, "ubx", np.array([000000.0, 100.0,self.max_a]))
    self.solver.cost_set_slice(0, N, "yref", yref[:N])
    self.solver.set(N, "yref", yref[N][:3])

    self.solver.solve()
    self.x_sol = self.solver.get_slice(0, N+1, 'x')
    self.u_sol = self.solver.get_slice(0, N, 'u')
    self.cost = self.solver.get_cost()

    self.v_solution = list(self.x_sol[:,1])
    self.a_solution = list(self.x_sol[:,2])
    self.j_solution = list(self.u_sol[:,0])

    # Reset if NaN or goes through lead car
    nans = np.any(np.isnan(self.x_sol))

    t = sec_since_boot()
    if nans:
      if t > self.last_cloudlog_t + 5.0:
        self.last_cloudlog_t = t
        cloudlog.warning("Longitudinal model mpc reset - nans")
      #TODO
      #self.reset_mpc()


if __name__ == "__main__":
  ocp = gen_long_mpc_solver()
  generate_code(ocp, json_file=JSON_FILE)
