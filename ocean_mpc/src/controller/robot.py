#
import numpy as np
from ocean_mpc.src.utils.utils import q_to_rot_mat, skew_symmetric, unit_quat
import math
import tf


class ROBOT:

    def __init__(self, mass, inertia, add_mass_v, add_mass_r, linear_damp_v, linear_damp_r, max_force, max_moment,):

        self.mass = mass
        self.inertia = inertia
        self.add_mass_v = add_mass_v
        self.add_mass_r = add_mass_r
        self.linear_damp_v = linear_damp_v
        self.linear_damp_r = linear_damp_r

        self.max_force = max_force
        self.max_moment = max_moment

        self.real_pos = np.zeros((3,))
        self.real_vel = np.zeros((3,))
        self.real_att = np.array([1.0, 0.0, 0.0, 0.0])  # Quaternion format: qw, qx, qy, qz
        self.real_rate = np.zeros((3,))

        self.sim_pos = np.zeros((3,))
        self.sim_vel = np.zeros((3,))
        self.sim_att = np.array([1.0, 0.0, 0.0, 0.0])  # Quaternion format: qw, qx, qy, qz
        self.sim_rate = np.zeros((3,))

    def set_pos(self, pos):
        self.real_pos = pos

    def set_vel(self, vel):
        self.real_vel = vel

    def set_att(self, att):
        self.real_att = att

    def set_rate(self, rate):
        self.real_rate = rate

    def get_real_state(self):
        x_real = np.concatenate((np.concatenate((self.real_pos, self.real_att)),
                                 np.concatenate((self.real_vel, self.real_rate)),))
        return x_real

    def get_sim_state(self):
        x_sim = np.concatenate((np.concatenate((self.sim_pos, self.sim_att)),
                                np.concatenate((self.sim_vel, self.sim_rate)),))
        return x_sim

    def p_dynamics(self, x_sim):
        return np.dot(q_to_rot_mat(x_sim[3:7]), x_sim[7:10])

    def q_dynamics(self, x_sim):
        return 1 / 2 * np.dot(skew_symmetric(x_sim[10:13]), x_sim[3:7])

    def v_dynamics(self, x_sim, u_force):

        force = np.array([u_force[0] * self.max_force[0], u_force[1] * self.max_force[1], u_force[2] * self.max_force[2]])
        
        dot_u = (force[0] - (self.linear_damp_v[0] * x_sim[7])) / (self.mass + self.add_mass_v[0])
        dot_v = (force[1] - (self.linear_damp_v[1] * x_sim[8])) / (self.mass + self.add_mass_v[1])
        dot_w = (force[2] - (self.linear_damp_v[2] * x_sim[9])) / (self.mass + self.add_mass_v[2])
        
        return np.array([dot_u, dot_v, dot_w])

    def r_dynamics(self, x_sim, u_moment):

        moment = np.array([u_moment[0] * self.max_moment[0], u_moment[1] * self.max_moment[1], u_moment[2] * self.max_moment[2]])

        dot_p = (moment[0] - (self.linear_damp_r[0] * x_sim[10])) / (self.inertia[0] + self.add_mass_r[0])
        dot_q = (moment[1] - (self.linear_damp_r[1] * x_sim[11])) / (self.inertia[1] + self.add_mass_r[1])
        dot_r = (moment[2] - (self.linear_damp_r[2] * x_sim[12])) / (self.inertia[2] + self.add_mass_r[2])
        
        return np.array([dot_p, dot_q, dot_r])

    def update(self, dt, opt_u):
        
        x_sim = self.get_sim_state()
        
        k1 = np.concatenate((self.p_dynamics(x_sim), self.q_dynamics(x_sim),
                             np.concatenate((self.v_dynamics(x_sim, opt_u[0:3]), self.r_dynamics(x_sim, opt_u[3:6])))))

        x_aux = [x_sim[i] + dt / 2 * k1[i] for i in range(13)]

        k2 = np.concatenate((self.p_dynamics(x_aux), self.q_dynamics(x_aux), 
                             np.concatenate((self.v_dynamics(x_aux, opt_u[0:3]), self.r_dynamics(x_aux, opt_u[3:6])))))
        
        x_aux = [x_sim[i] + dt / 2 * k2[i] for i in range(13)]

        k3 = np.concatenate((self.p_dynamics(x_aux), self.q_dynamics(x_aux),
                             np.concatenate((self.v_dynamics(x_aux, opt_u[0:3]), self.r_dynamics(x_aux, opt_u[3:6])))))
        
        x_aux = [x_sim[i] + dt / 2 * k3[i] for i in range(13)]

        k4 = np.concatenate((self.p_dynamics(x_aux), self.q_dynamics(x_aux),
                             np.concatenate((self.v_dynamics(x_aux, opt_u[0:3]), self.r_dynamics(x_aux, opt_u[3:6])))))

        x_sim = [x_sim[i] + dt * (1.0 / 6.0 * k1[i] + 2.0 / 6.0 * k2[i] + 2.0 / 6.0 * k3[i] + 1.0 / 6.0 * k4[i]) for i in range(13)]

        # Ensure unit quaternion
        # x[3:7] = unit_quat(x[3:7])
        
        self.sim_pos = x_sim[0:3]
        self.sim_att = x_sim[3:7]
        self.sim_vel = x_sim[7:10]
        self.sim_rate = x_sim[10:13]
        

