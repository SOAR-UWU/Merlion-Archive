#!/usr/bin/env python2

import rospy
from std_msgs.msg import Float64
from std_msgs.msg import Float32
from sensor_msgs.msg import Imu
from sensor_msgs.msg import FluidPressure
from tf.transformations import euler_from_quaternion


class StatePublisher:
    def __init__(self):
        rospy.init_node('merlion_state_publisher')

        self.yaw_pub = rospy.Publisher('yaw', Float64, queue_size=10)
        self.roll_pub = rospy.Publisher('roll', Float64, queue_size=10)
        self.pitch_pub = rospy.Publisher('pitch', Float64, queue_size=10)
        self.depth_pub = rospy.Publisher('depth', Float64, queue_size=10)

        rospy.Subscriber('input/imu', Imu, self.imu_callback) #.vectornav/IMU 
        # rospy.Subscriber('input/pressure', FluidPressure, self.pressure_callback)
        rospy.Subscriber('/merlion_hardware/sensor_depth', Float32, self.pressure_callback)

    def imu_callback(self, msg):
        quaternion = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        roll, pitch, yaw = euler_from_quaternion(quaternion)
        self.yaw_pub.publish(Float64(data=yaw))
        self.roll_pub.publish(Float64(data=roll))
        self.pitch_pub.publish(Float64(data=-pitch))

    def pressure_callback(self, msg):
        # pressure = msg.fluid_pressure
        # depth = (pressure - 101300) / (997.0474*9.80665)
        depth = msg.data
        self.depth_pub.publish(Float64(depth))


if __name__ == "__main__":
    StatePublisher()
    rospy.spin()
