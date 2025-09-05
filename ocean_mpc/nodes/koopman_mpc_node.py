#!/usr/bin/env python

import rospy
import numpy as np
from mavros_msgs.msg import ActuatorControl
from mavros_msgs.msg import WheelOdomStamped
from std_msgs.msg import Header
from geometry_msgs.msg import Vector3Stamped
from geometry_msgs.msg import QuaternionStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import sys
sys.path.append("/home/dlmux/WorkSpace/koopman_mpc/src")
from ocean_mpc.src.controller.ros_mpc import ROS_MPC
from std_srvs.srv import SetBool


class OCEAN_MPCWrapper:
    def __init__(self, exp_type, controller_state=True):
        
        self.controller_state = controller_state
        self.exp_type = exp_type
    
        self.q_cost = np.array([0.5, 0.5, 0.5, 1, 0.5, 0.5, 0.5, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        self.r_cost = np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.5])
       
        mass = rospy.get_param("~ocean_mass")
        inertia = rospy.get_param("~ocean_inertia")
        add_mass_v = rospy.get_param("~ocean_add_mass_v")
        add_mass_r = rospy.get_param("~ocean_add_mass_r")
        linear_damp_v = rospy.get_param("~ocean_linear_damp_v")
        linear_damp_r = rospy.get_param("~ocean_linear_damp_r")
        max_force = rospy.get_param("~ocean_max_force")
        max_moment = rospy.get_param("~ocean_max_moment")
        motor_max = rospy.get_param("~motor_max")
        
        self.dim = 13
        self.freq = rospy.get_param("~control_freq_factor", default=10)
        self.n_mpc_nodes = rospy.get_param("~n_nodes", default=10)
        self.dt = 1 / self.freq
        
        self.ros_mpc = ROS_MPC(mass, inertia, add_mass_v, add_mass_r, linear_damp_v, linear_damp_r, max_force, max_moment, 
                               self.n_mpc_nodes, self.q_cost, self.r_cost, self.exp_type, self.dt)

        self.pos = np.zeros((3,))
        self.vel = np.zeros((3,))
        self.att = np.array([1.0, 0.0, 0.0, 0.0])  # Quaternion format: qw, qx, qy, qz
        self.rate = np.zeros((3,))

        self.ref_pos = np.zeros((3,))
        self.ref_vel = np.zeros((3,))
        self.ref_att = np.array([1.0, 0.0, 0.0, 0.0])
        self.ref_rate = np.zeros((3,))
        
        self.motor_max = motor_max * 9.8
        self.max_forceMoment = np.concatenate((max_force, max_moment))

        self.motor_matrix = np.array([[0.57736, -0.57736, 0.57736, -0.57736, 0.57736, -0.57736, 0.57736, -0.57736,],  # x
                                      [-0.57736, -0.57736, 0.57736, 0.57736, 0.57736, 0.57736, -0.57736, -0.57736,],  # y
                                      [0.57736, -0.57736, -0.57736, 0.57736, -0.57736, 0.57736, 0.57736, -0.57736,],  # z
                                      [0.199105, 0.199105, 0.199105, 0.199105, -0.199105, -0.199105, -0.199105, -0.199105,],  # roll
                                      [-0.217233, 0.217233, 0.217233, -0.217233, -0.217233, 0.217233, 0.217233, -0.217233,],
                                      [-0.210211, -0.210211, 0.210211, 0.210211, -0.210211, -0.210211, 0.210211, 0.210211,],]).reshape((6, 8))  # yaw

        rospy.Subscriber("/imu/angular_velocity", Vector3Stamped, self.rate_callback_imu, queue_size=1)
        rospy.Subscriber("/filter/quaternion", QuaternionStamped, self.updateAtt_callback, queue_size=1)
        rospy.Subscriber("/test_reference", Odometry, self.ref_callback, queue_size=1)
        self.motor_pub = rospy.Publisher("/mavros/actuator_control", ActuatorControl, queue_size=10)
        self.control_pub = rospy.Publisher("/control_u", WheelOdomStamped, queue_size=10)
        self.wp_sim_pub = rospy.Publisher("/sim_wp", PoseStamped, queue_size = 10)
        rospy.Service("/cutoff_signal", SetBool, self.controller_state_callback)
        
        mpc_rate = rospy.Rate(self.freq)
        while True:
            self.runMPC()
            mpc_rate.sleep()

    def ref_callback(self, msg):
        self.ref_pos = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z,]
        self.ref_att = [msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z,]
        self.ref_vel = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
        self.ref_rate = [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]
        self.ref = np.concatenate((self.ref_pos, np.concatenate((self.ref_att, np.concatenate((self.ref_vel, self.ref_rate))))))
        self.ros_mpc.set_reference(self.ref)

    def rate_callback_imu(self, msg):
        self.rate = [msg.vector.x, msg.vector.y, msg.vector.z]
        self.ros_mpc.set_rate(self.rate)

    def updateAtt_callback(self, msg):
        self.att = [msg.quaternion.w, msg.quaternion.x, msg.quaternion.y, msg.quaternion.z,]
        self.ros_mpc.set_att(self.att)

    def runMPC(self):
        
        if self.controller_state == True:
            u = self.ros_mpc.optimize()
            
            if self.exp_type == "sim":
                self.ros_mpc.simulate(self.dt, u)
                sim_cur_state = self.ros_mpc.get_current_sim_state()
                sim_cur_p = PoseStamped()
                sim_cur_p.pose.position.x, sim_cur_p.pose.position.y, sim_cur_p.pose.position.z = sim_cur_state[0: 3]
                sim_cur_p.pose.orientation.w, sim_cur_p.pose.orientation.x, sim_cur_p.pose.orientation.y, sim_cur_p.pose.orientation.z = sim_cur_state[3:7]
                self.wp_sim_pub.publish(sim_cur_p)
                
            control = WheelOdomStamped()
            control.header = Header()
            control.header.stamp = rospy.Time.now()
            control.data = u[:6]

            self.control_pub.publish(control)
            max_motor_ = np.diag(np.array([self.motor_max, self.motor_max, self.motor_max, self.motor_max, 
                                           self.motor_max, self.motor_max, self.motor_max, self.motor_max,]))

            # motor_percent = f_max(8*8)^-1 * A^-1 * tau_max(6*6) * u
            motor_ = np.dot(np.linalg.inv(max_motor_),
                            np.dot(np.linalg.pinv(self.motor_matrix), 
                                   np.dot(np.diag(self.max_forceMoment), u)))
            
            motor = ActuatorControl()
            motor.controls = motor_[:8]
            self.motor_pub.publish(motor)

        else:
            rospy.loginfo("Motor is stopping...")
            motor = ActuatorControl()
            motor.controls = [0, 0, 0, 0, 0, 0, 0, 0]
            self.motor_pub.publish(motor)
            
    def controller_state_callback(self, msg):
        if msg.data == True:
            self.controller_state = False
            return True
        else:
            self.controller_state = True
            return False


def main():
    rospy.init_node("koopman_mpc_node")
    exp_type = rospy.get_param("~exp_type", default= "sim")
    controller_state = True
    OCEAN_MPCWrapper(exp_type, controller_state)

if __name__ == "__main__":
    main()
