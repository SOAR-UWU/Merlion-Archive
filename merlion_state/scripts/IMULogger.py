import rospy
import numpy as np
import pandas as pd


class IMULogger:
    def __init__(self):
        rospy.init_node('IMU_logger')

        rospy.Subscriber('input/imu', Imu, self.imu_callback)

        self.imu_data = []

    def imu_callback(self, msg):
        data = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
                msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
                msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]

        self.imu_data.append(data)
        print(self.imu_data)

    def save_data(self):
        dataframe = pd.DataFrame(self.imu_data)
        dataframe.to_csv(r"home\soar\IMU.csv")


if __name__ == '__main__':
    IMULogger()
    rospy.spin()