
import numpy as np
from ocean_mpc.src.controller.mpc import MPC
from ocean_mpc.src.controller.robot import ROBOT

class ROS_MPC:
    
    def __init__(self, mass, inertia, add_mass_v, add_mass_r, linear_damp_v, linear_damp_r, 
                 max_force, max_moment, n_nodes, q_cost, r_cost, exp_type, dt):
       
        self.robot = ROBOT(mass, inertia, add_mass_v, add_mass_r, linear_damp_v, linear_damp_r, 
                           max_force, max_moment)
        self.mpc = MPC(self.robot, n_nodes, q_cost, r_cost, exp_type, dt)
        
    def set_pos(self, pos):
        self.robot.set_pos(pos)

    def set_att(self, att):
        self.robot.set_att(att)

    def set_vel(self, vel):
        self.robot.set_vel(vel)

    def set_rate(self, rate):
        self.robot.set_rate(rate)  
        
    def set_reference(self, x_ref):
        return self.mpc.set_reference(x_ref)
    
    def optimize(self):
        u = self.mpc.optimize()    
        return u
    
    def simulate(self, dt, opt_u):
        self.mpc.simulate(dt, opt_u)

    def get_current_real_state(self):
        return self.mpc.get_real_state()

    def get_current_sim_state(self):
        return self.mpc.get_sim_state()
