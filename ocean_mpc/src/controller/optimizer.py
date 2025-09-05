import os
import numpy as np
import sys

import casadi as cs
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
from ocean_mpc.src.utils.utils import q_to_rot_mat, skew_symmetric
from copy import copy
import time
import tf

class MPC_Optimizer:

    def __init__(self, robot, n_nodes, q_cost, r_cost, dt):
        
        self.robot = robot
        self.max_u = np.array([0.6, 0.6, 0.6, 0.6, 0.6, 0.6])
        self.min_u = np.array([-0.6, -0.6, -0.6, -0.6, -0.6, -0.6])
        
        self.robot_max_force = robot.max_force
        self.robot_max_moment = robot.max_moment
        
        self.N = n_nodes
        self.T = self.N * dt
        self.dt = dt
        
        self.p = cs.MX.sym("p", 3)
        self.q = cs.MX.sym("q", 4)
        self.v = cs.MX.sym("v", 3)
        self.r = cs.MX.sym("r", 3)
        
        self.x = cs.vertcat(self.p, self.q, self.v, self.r)
        
        uu = cs.MX.sym("ux")
        uv = cs.MX.sym("uy")
        uw = cs.MX.sym("uz")
        up = cs.MX.sym("up")
        uq = cs.MX.sym("uq")
        ur = cs.MX.sym("ur")
        
        self.u = cs.vertcat(uu, uv, uw, up, uq, ur)
        self.acados_ocp_solver = {}
        self.acados_models_dir = ("/home/dlmux/WorkSpace/koopman_mpc/src/ocean_mpc/acados_models")
        
        ocp = AcadosOcp()
        ocp.dims.N = self.N
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"
        ocp.cost.W = np.diag(np.concatenate((q_cost, r_cost)))
        ocp.cost.W_e = np.diag(q_cost)

        self.standard_dynamics = self.robot_dynamics()
        stand_acados_model = self.acados_setup_model(
            self.standard_dynamics(x=self.x, u=self.u)["x_next"],
        )

        ocp.model = stand_acados_model
        nx = stand_acados_model.x.size()[0]
        nu = stand_acados_model.u.size()[0]
        ny = nx + nu

        x_ref = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        y_ref = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        y_ref_e = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        ocp.cost.yref = y_ref
        ocp.cost.yref_e = y_ref_e
        
        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[-nu:, -nu:] = np.eye(nu)
        ocp.cost.Vx_e = np.eye(nx)

        ocp.constraints.x0 = x_ref
        ocp.constraints.lbu = np.array(self.min_u)
        ocp.constraints.ubu = np.array(self.max_u)
        ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4, 5])

        ocp.solver_options.tf = self.T
        ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "DISCRETE"
        ocp.solver_options.print_level = 0
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.nlp_solver_max_iter = 50
        ocp.solver_options.qp_solver_iter_max = 2000

        json_file = os.path.join(
            self.acados_models_dir, stand_acados_model.name + "_acados_ocp.json"
        )
        self.acados_ocp_solver = AcadosOcpSolver(ocp, json_file=json_file)

        self.max_opt_time = 0
        self.sum_opt_time = 0
        self.num_opt = 0

    def robot_dynamics(self):
        
        x_next = cs.vertcat(self.p_dynamics(), self.q_dynamics(), self.v_dynamics(), self.r_dynamics())
        return cs.Function("x_next", [self.x, self.u], [x_next], ["x", "u"], ["x_next"])
    
    def acados_setup_model(self, dyn):
    
        def fill_in_acados_model(x, u, p, dynamics):

            model = AcadosModel()
            model.disc_dyn_expr = dynamics 
            model.x = x
            model.u = u
            model.p = p
            model.name = "koopman"
            return model

        acados_models = {}
        dynamics_ = dyn
        acados_models = fill_in_acados_model(
            x=self.x, u=self.u, p=[], dynamics=dynamics_
        )

        return acados_models

    def p_dynamics(self):
    
        res = cs.mtimes(q_to_rot_mat(self.q), self.v)
        self.p[0] = self.p[0] + res[0] * self.dt
        self.p[1] = self.p[1] + res[1] * self.dt
        self.p[2] = self.p[2] + res[2] * self.dt
        return self.p
        
    def q_dynamics(self):
    
        self.q[0] = self.q[0] - 0.5 * (self.q[1] * self.r[0] + self.q[2] * self.r[1] + self.q[3] * self.r[2]) * self.dt
        self.q[1] = self.q[1] + 0.5 * (self.q[0] * self.r[0] + self.q[2] * self.r[2] - self.q[3] * self.r[1]) * self.dt
        self.q[2] = self.q[2] + 0.5 * (self.q[0] * self.r[1] - self.q[1] * self.r[2] + self.q[3] * self.r[0]) * self.dt
        self.q[3] = self.q[3] + 0.5 * (self.q[0] * self.r[2] + self.q[1] * self.r[1] - self.q[2] * self.r[0]) * self.dt
        return self.q
    
    def v_dynamics(self):

        self.v[0] = self.v[0] + ((self.robot_max_force[0] * self.u[0] - (self.robot.linear_damp_v[0] * self.v[0])) / (self.robot.mass + self.robot.add_mass_v[0])) * self.dt
        self.v[1] = self.v[1] + ((self.robot_max_force[1] * self.u[1] - (self.robot.linear_damp_v[1] * self.v[1])) / (self.robot.mass + self.robot.add_mass_v[1])) * self.dt
        self.v[2] = self.v[2] + ((self.robot_max_force[2] * self.u[2] - (self.robot.linear_damp_v[2] * self.v[2])) / (self.robot.mass + self.robot.add_mass_v[2])) * self.dt

        return self.v

    def r_dynamics(self):
        
        qw = self.q[0]
        qx = self.q[1]
        qy = self.q[2]
        qz = self.q[3]
        
        R = cs.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
        P = cs.asin(2 * (qw * qy - qz * qx))
        Y = cs.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))

        # koopman model
        self.r[0] = 1.00209152 * self.r[0] + 0.00219668 * self.r[0] ** 2 -0.00826154 * self.r[0] * self.r[1] -0.00812254 * self.r[0] * self.r[2] -0.02041151 * self.r[0] ** 3 +4.55032887e-01 * self.u[3] + 1.27776529e-03 * cs.cos(P) * cs.cos(R) -3.27872569e-01 * cs.cos(P) * cs.sin(R) -3.49511153e-02 * cs.sin(P)
        
        self.r[1] = self.r[1] + (((self.robot_max_moment[1] * self.u[4] - (self.robot.linear_damp_r[1] * self.r[1])) / (self.robot.inertia[1] + self.robot.add_mass_r[1])) * self.dt)
        self.r[2] = self.r[2] + (((self.robot_max_moment[2] * self.u[5] - (self.robot.linear_damp_r[2] * self.r[2])) / (self.robot.inertia[2] + self.robot.add_mass_r[2])) * self.dt)
        
        return self.r 

    
    def set_reference_state(self, x_target):
        
        target = np.array((x_target))
        ref_u = np.array([0, 0, 0, 0, 0, 0])
        ref = np.concatenate((target, ref_u), axis=0)

        for j in range(self.N):
            self.acados_ocp_solver.cost_set(j, "yref", ref)
        
        self.acados_ocp_solver.cost_set(self.N, "yref", target)

    def set_reference_trajectory(self, x_target):
        while x_target.shape[0] < self.N + 1:
            x_target = np.concatenate(
                (x_target, np.expand_dims(x_target[-1, :], 0)), axis=0
            )
        ref_u = np.array([0, 0, 0, 0, 0, 0])

        for j in range(self.N):
            target = x_target[j, :]
            ref = np.concatenate((target, ref_u), axis=0)
            self.acados_ocp_solver.cost_set(j, "yref", ref)

        self.acados_ocp_solver.cost_set(self.N, "yref", x_target[self.N, :])

    def run_optimize(self, initial_state):

        x_init = initial_state
        x_init = np.stack(x_init)
        
        self.acados_ocp_solver.set(0, "lbx", x_init)
        self.acados_ocp_solver.set(0, "ubx", x_init)

        t = time.time()
        status = self.acados_ocp_solver.solve()
        opt_time = time.time() - t

        cost = self.acados_ocp_solver.get_cost()
        if opt_time > self.max_opt_time:
            self.max_opt_time = opt_time

        self.sum_opt_time += opt_time
        self.num_opt += 1
        u_opt_acados = np.ndarray((self.N, 6))

        for i in range(self.N):
            u_opt_acados[i, :] = self.acados_ocp_solver.get(i, "u")
        u = np.array(u_opt_acados[0, :])
        return u
