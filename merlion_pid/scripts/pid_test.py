#!/usr/bin/env python2
import rospy
from std_msgs.msg import Float64, String
import numpy as np

class PID_Test(): 
    def __init__(self): 
        rospy.init_node("pid_test", anonymous=True)
        self.state = "inactive"
        self.yaw_setpoint_pub = rospy.Publisher('/merlion_pid/yaw/setpoint', Float64, queue_size=10)
        self.pitch_setpoint_pub = rospy.Publisher('/merlion_pid/pitch/setpoint', Float64, queue_size=10)
        self.roll_setpoint_pub = rospy.Publisher('/merlion_pid/roll/setpoint', Float64, queue_size=10)
        self.speed_setpoint_pub = rospy.Publisher('/merlion_pid/speed/setpoint', Float64, queue_size=10)
        self.depth_setpoint_pub = rospy.Publisher('/merlion_pid/depth/setpoint', Float64, queue_size=10)
        self._setpoint_vec = np.array([0, 0, 0, 0, 0]) #[yaw, pitch, roll, speed, depth]
        self.calibration_vec = np.array([0, 0, 0, 0, 0]) #[in case initial state has values that are referenced not at 0]
        self.state_sub = rospy.Subscriber("/merlion_pid/pid_test", String, lambda msg: self.set_state(msg.data))

    @property 
    def setpoint_vec(self): 
        return self._setpoint_vec
    
    @setpoint_vec.setter
    def setpoint_vec(self, value): 
        self._setpoint_vec = value    

    def set_state(self, state): 
        print("setting state to ", state )
        self.state = state

    def publish_setpoints(self): 
        yaw, pitch, roll, speed, depth = self.setpoint_vec + self.calibration_vec
        self.yaw_setpoint_pub.publish(Float64(yaw))
        self.pitch_setpoint_pub.publish(Float64(pitch))
        self.roll_setpoint_pub.publish(Float64(roll))
        self.speed_setpoint_pub.publish(Float64(speed))
        self.depth_setpoint_pub.publish(Float64(depth))

    def forward(self):
        self.setpoint_vec = np.array(0, 0, 0.5, 0.5, 0)

    def hover(self): 
        self.setpoint_vec = np.array([0, 0, 0, 0, 0])

    def left(self): 
        self.setpoint_vec = np.array([-np.pi/2, 0, 0, 0, 0])

    def right(self): 
        self.setpoint_vec = np.array([np.pi/2, 0, 0, 0, 0])

    def back(self): 
        self.setpoint_vec = np.array([np.pi, 0, 0, 0, 0])

    def pitch_forwards(self): 
        self.setpoint_vec = np.array([0, -np.pi/4, 0, 0, 0])
    
    def pitch_backwards(self): 
        self.setpoint_vec = np.array([0, np.pi/4, 0, 0, 0])

    def roll_left(self): 
        self.setpoint_vec = np.array([0, 0, -np.pi/4, 0, 0])

    def roll_right(self): 
        self.setpoint_vec = np.array([0, 0, np.pi/4, 0, 0])

    def submerge(self): 
        self.setpoint_vec = np.array([0, 0, 0, 0, 1])

    def pub_setpoints(self,event): 
        if self.state == "f": 
        # f for forwards
            self.forward()
        elif self.state == "l": 
            # l for left 
            self.left() 
        elif self.state == "r":
            # r for right  
            self.right()
        elif self.state == "b": 
            # b for back 
            self.back()
        elif self.state == "pf": 
            # pf for pitch forwards 
            self.pitch_forwards()
        elif self.state == "pb":
            # pb for pitch backwards 
            self.pitch_backwards() 
        elif self.state == "rl": 
            # rl for roll left
            self.roll_left() 
        elif self.state == "rr":
            # rr for roll right 
            self.roll_right() 
        elif self.state == "hover": 
            # h for hover
            self.hover()
        elif self.state == "submerge": 
            self.submerge()

        self.publish_setpoints()

if __name__ == "__main__": 
    pid_test = PID_Test() 
    time = 0
    print(pid_test.state)
    timer = rospy.Timer(rospy.Duration(0.01), pid_test.pub_setpoints)
    pid_test.publish_setpoints()
    rospy.spin()
    timer.shutdown()
