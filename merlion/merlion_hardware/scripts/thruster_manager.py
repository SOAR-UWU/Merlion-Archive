#!/usr/bin/env python2

import numpy as np
import time
import rospy
from geometry_msgs.msg import Twist 
from merlion_hardware.msg import Motor 

class ThrusterManagerNode:
    def __init__(self):
        self.node_name = 'merlion_thruster'
        rospy.init_node(self.node_name)

        self.tranform_matrix = np.array([
            # x  y   z roll pitch yaw
            [0,  0,  0, -1,  -1,  0],  # left front motor
            [0,  0,  0,  1,   1,  0],  # right front motor
            [-1, 0,  0,  0,   0, -1],  # left rear motor
            [1,  0,  0,  0,   0, -1],  # right rear motor
            [0,  0, -1,  0,   1,  0]   # rear motor
        ])

        self.normalizing_matrix = np.array([
            [1/3, 0 ,  0 ,  0 ,  0 ], 
            [ 0, 1/3,  0 ,  0 ,  0 ], 
            [ 0,  0 , 1/2,  0 ,  0 ], 
            [ 0,  0,   0,  1/2,  0 ], 
            [ 0,  0,   0,   0,  1/2], 
        ])
        self.scale = 100
        self.num_thrusters = len(self.tranform_matrix)

        rospy.Subscriber(self.node_name + '/input', Twist, self.input_callback)
        self.thruster_pub = rospy.Publisher('merlion_hardware/thruster_values', Motor, queue_size=1)
        self.relay = rospy.Publisher('merlion_hardware/relay_thrust',Motor,queue_size=1)
        self.counter = 1
        rospy.loginfo("ThrusterManagerNode started")

    def input_callback(self, msg):
        def motor_limit(motorval, offset, thres=80):
            if int(motorval) + offset == 0:
                return 0

            value = int(motorval) + offset
            return ( min(abs(value), thres) * (value/abs(value)) )

        control_vector = np.array([
            [msg.linear.x], [msg.linear.y], [msg.linear.z],
            [msg.angular.x], [msg.angular.y], [msg.angular.z]
        ])
        output = self.scale * self.tranform_matrix.dot(control_vector)
        # self.update_thrusters(output)
        msg = Motor()
        msg.m1 = motor_limit(output[0], 45) 
        msg.m2 = motor_limit(output[1], -45)
        msg.m3 = motor_limit(output[2], 0)
        msg.m4 = motor_limit(output[3], 0)
        msg.m5 = motor_limit(output[4], 60) #was -50
        #time.sleep(0.02)
        self.thruster_pub.publish(msg)
        if self.counter >=3:
            self.relay.publish(msg)
            self.counter = 0

        else:
            self.counter +=1
            #self.relay.publish(msg)

if __name__ == '__main__':
    ThrusterManagerNode()
    rospy.spin()
